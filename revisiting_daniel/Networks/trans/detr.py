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

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, mid_dim, num_queries):
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
        self.class_embed = MLP(hidden_dim, hidden_dim, 2, 2)
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

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the o above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        #print(("------------")
        ##print((samples.tensors)
        #print((f"samples t {samples.tensors.shape}")
        #print((f"samples m {samples.mask.shape}")
        features, pos = self.backbone(samples)
        #print((f"feature count {len(features)}")
        #print((f"feature shapet {features[0].tensors.shape}")
        #print((f"feature shapem {features[0].mask.shape}")
        #print((f"pos c {len(pos)}")
        #print((f"pos t {pos[0].shape}")
        src, mask = features[-1].decompose()
        #print((f"src {src.shape}")
        #print((f"mask {mask.shape}")
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        hs = self.public_layer(hs)
        outputs_class = self.class_embed(hs) # [B*1*F*2]
        outputs_feature = self.bbox_embed(hs).relu() # [B*1*F*M]
        
        outputs_class_label = outputs_class.argmax(dim=3) #[B*1*F]
        output_shape = outputs_feature.shape
        batch_size = output_shape[0]
        feature_shape = outputs_feature[0].shape #[1*F*M]
        feature_list=[]
        #check(samples,"samples")
        #check(features,"features")
        #check(pos,"pos")
        #check(src,"src")
        #check(mask,"mask")
        check(hs,"hs")
        #check(outputs_class,"oclass")
        #check(outputs_feature,"ofeatures")
        for batchIndex in range(batch_size):
            features=[]
            outputs_feature[batchIndex]#[1*F*M]
            feature_mask = outputs_class_label[batchIndex].unsqueeze(-1).expand(feature_shape).bool()#[1*F*M]
            feature_filter=torch.masked_select(outputs_feature[batchIndex],feature_mask)
            feature_useful = feature_filter.reshape(-1,feature_shape[2]) #[1*M]
            feature_list.append(feature_useful)
        out={}
        out['pred_v']=self.rcn(feature_list)
        out['pred_class'] = outputs_class
        
        #logging.info("Debug out--------------------")
        #logging.info(out)
        return out

class RCN(nn.Module):
    
    def __init__(self, input_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim*2,input_dim*2)
        self.relu_layer1 = nn.LeakyReLU()

        self.linear2 = nn.Linear(input_dim*2,input_dim)
        self.linear2_1 = nn.Linear(input_dim,input_dim)
        self.relu_layer2 = nn.LeakyReLU(inplace=True)
        # use leaky relu to avoid dead cell
        self.linear3 = nn.Linear(input_dim,1)
        pass

    # input: [count*feature_dim]
    # output:  [count]
    # count should be higher than 2
    # if ==1, use self twice
    # if ==0, return empty
    def _processBatch(self,x):
        ##print((f"x shape {x.shape}")
        feature_count = x.shape[0]
        feature_size = x.shape[1]
        lastIndex= feature_count-1
        tensor_list=[]
        if feature_count==0:
            return None
        for i in range(lastIndex):
            tensor_list.append(self._processPair(x[i],x[i+1]))
        tensor_list.append(self._processPair(x[-1],x[0]))
        return torch.cat(tensor_list,dim=0)

    # input: [feature_dim],[feature_dim]
    # output: [1]
    def _processPair(self,x1,x2):
        x = torch.cat([x1,x2],dim=0)
        ##print((f"Batch cat x shape {x.shape}")
        y0 = self.relu_layer1(self.linear1(x))+x
        ##print((f"Batch cat y0 shape {y0.shape}")
        y1 = self.relu_layer2(self.linear2(y0))
        y1_1 = self.relu_layer2(self.linear2_1(y1))+y1
        ##print((f"Batch cat y1 shape {y1.shape}")
        y2 = self.relu_layer2(self.linear3(y1_1))
        ##print((f"Batch cat y2 shape {y2.shape}")
        return y2.reshape(-1)

    # input: Batch(list), count*feature_dim
    # output: Batch(list), count
    def forward(self,xList):
        yList=[]
        for x in xList:
            result = self._processBatch(x)
            yList.append(result)
        return yList


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

    # outputs: [B*1*F*2]
    # 
    def loss_classify(self, outputs, targets):
        #todo: gen ground truth, and 
        batch_size = outputs.shape[0]
        feature_count = outputs.shape[2]

        true_result = torch.zeros((batch_size,feature_count),dtype=torch.long).to(self.device)
        for i in range(batch_size):
            for j in range(len(targets[i])):
                true_result[i][j]=1

        loss = self.zero.clone()
        for i in range(batch_size):
            loss+=self.entropy_loss(outputs.squeeze(1)[i],true_result[i])
        loss/=torch.Tensor([batch_size]).to(self.device)
        return loss*self.lamda_class

    # outputs: (list)batch_size
    def retainOrgValue(self,outputs):
        batch_size = len(outputs)
        new_outputs=[]
        for i in range(batch_size):
            output = outputs[i]
            if output is None:
                new_outputs.append(output)
                continue
            output_count = output.shape[0]
            multi = self.one.clone()
            for i in output:
                multi*=i
            new_output=multi.expand(output_count)#torch.Tensor([multi]*output_count)
            #new_output[0]=1.0
            new_output=new_output.to(self.device)
            #new_output[-1]=output[-1]
            for j in range(1,output_count):
                for j2 in range(j,output_count):
                    new_output[j] *= output[j2]
            new_output = new_output/torch.max(new_output)
            new_outputs.append(new_output)
            #print(output)
            #print(new_output)
            #print("------------")
        return new_outputs
    
    def preprocessTargetValue(self,true_result):
        newList=[]
        for true_result_c in true_result:
            r=true_result_c/torch.max(true_result_c)
            newList.append(r)
        return newList
    
    # outputs: (list)batch_size, 1*F*M varied count
    # true_result
    def loss_result(self,outputs,outputs_class, true_result):
        batch_size = len(outputs)
        all_loss = torch.zeros(1).to(self.device)
        outputs_class_labels = outputs_class.argmax(dim=3).squeeze(1) #[B*F]
        loss_multi_total = self.zero.clone()
        loss_result_true_total = self.zero.clone()
        total_quant_correct=0.0
        for i in range(batch_size):
            output = outputs[i]
            if output is None:
                continue
            output_count = output.shape[0]
            multi_all = self.one.clone()
            for j in range(output_count):
                multi_all*=output[j]
            loss_multi = self.mse_loss(multi_all,self.one.clone())

            final_output=output
            true_result_c = true_result[i]
            # check cover, otherwise, all of the data should be zero
            lossv1 = self.zero.clone()
            should_be_true=int(len(true_result_c))
            t_count=0
            f_count=0
            for j in range(should_be_true):
                if outputs_class_labels[i][j]==1:
                    lossv1+=self.mse_loss(final_output[t_count],true_result_c[j])
                    t_count+=1
                else:
                    f_count+=1
            '''
            Truth:    1 1 1 1 1 0 0 0 ...
            Ours :    1 1 0 1 0 0 1 1 
                      + +   +                   (true)  v1
                          +   +   + +           (false) v2
            '''
            countt= torch.Tensor([f_count+t_count]).to(self.device)
            loss_multi*=self.lamda_1
            loss_multi_total+=loss_multi
            pre = t_count/(t_count+f_count)
            recall = t_count/should_be_true
            if t_count>0:
                total_quant_correct+=2*pre*recall/(pre+recall)
            if t_count>0:
                loss_result_true_total+=lossv1/countt
        total_quant_correct/=batch_size
        bs = torch.tensor([batch_size]).to(self.device)
        loss_multi_total/=bs
        loss_result_true_total/=bs
        loss_result_true_total*=self.lamda_v
        all_loss = loss_multi_total + loss_result_true_total
        return all_loss, loss_multi_total, loss_result_true_total, torch.Tensor([total_quant_correct]).to(self.device)

    def get_loss(self, outputs, targets, predName="pred_v"):
        # TODO: modify
        loss_map={}
        loss_map["label"] = self.loss_classify(outputs["pred_class"],targets)
        loss_map["r"], loss_map["r_mult"], loss_map["r_comp"], loss_map["f1"] = self.loss_result(outputs[predName],outputs["pred_class"],targets)
        return loss_map

    def forward(self, outputs, targets, predName="pred_v"):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        loss_map = self.get_loss(outputs,targets, predName)
        loss_map["total_loss"] = loss_map["label"]+loss_map["r"]
        return loss_map


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


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


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors
