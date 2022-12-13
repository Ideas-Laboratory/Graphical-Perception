# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer
import logging
from util.debug_utils import check
from torch.nn.utils.rnn import pad_sequence

class DETROpt(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, mid_dim, num_queries, label_quant):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            mid_dim    : transform -> FFN -> output feature dim
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = MLP(hidden_dim, hidden_dim, label_quant, 2)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, mid_dim, 2)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.public_layer = MLP(hidden_dim,hidden_dim,hidden_dim,2)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.rcn = RCN(mid_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        '''
            About dimension:
                B:batch
                F:Features = num_queries
                T:Lable types 
                M:Feature Content = mid_dim
        '''
        hs = self.public_layer(hs).squeeze(0)
        check(hs,"hs") # check nan
        outputs_class = self.class_embed(hs).squeeze(1) # [B*F*T]
        outputs_feature = self.bbox_embed(hs).relu().squeeze(1) # [B*F*M]

        feature_shape = outputs_feature.shape
        feature_dim = feature_shape[2]
        batch_size = feature_shape[0]

        
        outputs_class_label = outputs_class.argmax(dim=2) #[B*F]
        outputs_class_valid = outputs_class_label.bool()
        outputs_feature_mask = outputs_class_valid.unsqueeze(2).expand(outputs_feature.shape) #[B*F]->[B*F*M]
        feature_useful=torch.masked_select(outputs_feature,outputs_feature_mask).reshape(-1,feature_dim) #[?*M]

        # now, stack features together
        outputs_count = torch.sum(outputs_class_valid,dim=1) #[B*F]->[B]
        index = 0
        feature_stack = torch.Tensor().to(outputs_feature)
        if feature_useful.shape[0]==0:
            out={}
            #[B*?*2M] --> [B*?]
            out['pred_v']=feature_useful 
            out['pred_c']=outputs_count #[B]
            out['pred_class'] = outputs_class.squeeze(1) #[B*F*T]
            out['pred_label'] =  outputs_class_label#[B*F]
            out['mask']=outputs_class_valid
            out['pred_count']=outputs_count
            return out
        feature_final = feature_useful #[?*M]
        out={}
        
        out['pred_v']=self.rcn(feature_final).squeeze(1)
        out['pred_c']=outputs_count #[B]
        out['pred_class'] = outputs_class.squeeze(1) #[B*F*T]
        out['mask']=outputs_class_valid
        out['pred_label'] =  outputs_class_label
        out['pred_count']=outputs_count
        return out

class RCN(nn.Module):
    
    def __init__(self, input_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim,input_dim)
        self.relu_layer1 = nn.LeakyReLU()

        self.linear2 = nn.Linear(input_dim,input_dim)
        self.linear2_1 = nn.Linear(input_dim,input_dim)
        self.relu_layer2 = nn.LeakyReLU()
        # use leaky relu to avoid dead cells
        self.linear3 = nn.Linear(input_dim,1)
        pass

    #features: B*?*2M
    def forward(self,features):
        x = features
        y0 = self.relu_layer1(self.linear1(x))+x
        y1 = self.relu_layer2(self.linear2(y0))
        y1_1 = self.relu_layer2(self.linear2_1(y1))+y1
        y2 = self.relu_layer2(self.linear3(y1_1))
        return y2

def replaceNan(a):
    return torch.where(torch.isnan(a), torch.full_like(a, 0), a)



class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, lamda_1, lamda_class,lamda_v, loss_type):
        """ Create the criterion.
        Parameters:
        """
        super().__init__()
        self.entropy_loss = nn.CrossEntropyLoss()
        if loss_type == "L2":
            self.mse_loss = nn.MSELoss(reduction="none")
            self.mse_loss_avg = nn.MSELoss()
        elif loss_type == "L1": #smooth L1
            self.mse_loss = nn.SmoothL1Loss(reduction="none")
            self.mse_loss_avg = nn.SmoothL1Loss()
        self.lamda_1 = torch.Tensor([lamda_1])
        self.lamda_class = torch.Tensor([lamda_class])
        self.lamda_v = torch.Tensor([lamda_v])
        self.one = torch.Tensor([1.0])
        self.zero = torch.Tensor([0.0])

    def setConfig(self,config,device):
        self.lamda_1 = self.lamda_1.to(device)
        self.lamda_class = self.lamda_class.to(device)
        self.lamda_v = self.lamda_v.to(device)
        self.one = self.one.to(device)
        self.zero = self.zero.to(device)
        self.device=device

    # outputs: [B*F*T]
    # targets: [B*F]
    def loss_classify(self, outputs, targets):
        type_dim = outputs.shape[2]
        # convert to [BF*T] & [BF]
        return self.entropy_loss(outputs.reshape(-1,type_dim),targets.reshape(-1))*self.lamda_class
    
    def loss_result(self, outputs, target_result, output_mask, output_count, pred_label, target_label):
        # outputs: [?]
        # outputs_class: [B*F*T]
        # target_result: [B*F] (Padding to F)
        # output_mask: [B*F] (bool)
        if outputs.shape[0]>0:
            target_mask_eq = target_label.eq(pred_label)
            target_useful_mask = target_label.gt(0)
            target_mask =  torch.bitwise_and(target_mask_eq,target_useful_mask)
            #target_mask = target_result.eq(pred_label) #[B*F] bool
            public_mask = torch.bitwise_and(output_mask, target_mask)
            public_count = torch.sum(public_mask,dim=1) #[B]
            batch_size = output_mask.shape[0]
            batch_size_tensor = torch.Tensor([batch_size]).to(public_count[0]).float()
            weight = public_count.pow(-1.0)
            weight = torch.where(weight.isinf(),torch.full_like(weight,0),weight) # replace inf
            weight = weight.unsqueeze(1).expand(target_result.shape) * public_mask.float() #[B*F]
            filter_target = torch.masked_select(target_result,output_mask)
            filter_weight = torch.masked_select(weight,output_mask)
            normal_loss = torch.sum(self.mse_loss(filter_target,outputs) * filter_weight) / batch_size_tensor*self.lamda_v
            return normal_loss
        else:
            return torch.Tensor([0]).to(target_result)
    
    def get_loss(self, outputs, targets):
        # TODO: modify
        loss_map={}
        loss_map["label_l"] = self.loss_classify(outputs["pred_class"],targets["label_l"])
        loss_map["r"], = self.loss_result(outputs["pred_v"],targets["num"],outputs['mask'],outputs["pred_count"],outputs['pred_label'],targets["label_l"])
        return loss_map

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        loss_map = self.get_loss(outputs,targets)
        loss_map["total_loss"] = loss_map["label_l"]+loss_map["r"]
        return loss_map

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x