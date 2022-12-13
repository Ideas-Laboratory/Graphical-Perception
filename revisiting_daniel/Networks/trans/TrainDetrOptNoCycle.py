from ..trans import detrOptNoCycle as detr
from ..trans import backbone
from ..trans import transformer
from ..trans import position_encoding
import torch.nn as nn
import torch
from .. import NetUtils
from util.Config import obj2dic

class DetrTrainer(nn.Module):
    def __init__(self,network,loss):
        super(DetrTrainer,self).__init__()
        self.network=network
        self.loss=loss
    
    def forward(self,x,returnLoss=False):
        input = x["input"]["img"]
        if returnLoss:
            target = x["target"]
            y = self.network(input)
            loss = self.loss(y,target)
            return y, loss
        else:
            return self.network(input)

class TrainDetrOptNoCycle(nn.Module):

    def __init__(self,param):
        super(TrainDetrOptNoCycle,self).__init__()
        self.param=param

        # build backbone

        position_embedding=None
        if param.joiner.positionEmbeddingType in ('v2','sine','PositionEmbeddingSine'):
            position_embedding = position_encoding.PositionEmbeddingSine(**obj2dic(param.joiner.positionEmbeddingSine))
        elif param.joiner.positionEmbeddingType in ('v3','learned','PositionEmbeddingLearned'):
            position_embedding = position_encoding.PositionEmbeddingLearned(**obj2dic(param.joiner.positionEmbeddingLearned))
        else:
            print("Unknown embedding type %s"%param.joiner.positionEmbeddingType)
            raise ValueError(f"not supported {param.joiner.positionEmbeddingType}")
        backbone_model = backbone.Backbone(**obj2dic(param.joiner.backbone))
        joiner = backbone.Joiner(backbone_model, position_embedding)
        joiner.num_channels = backbone_model.num_channels

        # transformer
        trans_model = transformer.Transformer(**obj2dic(param.transformer))

        self.network = detr.DETROpt(
            joiner,
            trans_model,
            mid_dim=param.detr.mid_dim,
            num_queries=param.detr.num_queries,
            label_quant=param.detr.label_quant
        )

        self.ignoreLable = param.ignoreLable
        if not isinstance(self.ignoreLable, bool):
            self.ignoreLable = False
            param.ignoreLable= False
        self.lossfunc = detr.SetCriterion(**obj2dic(param.lossFunction))
        
        self.moduleNet = DetrTrainer(self.network,self.lossfunc)


    def setConfig(self,config,device):
        self.config=config
        self.device=device
        self.opt = NetUtils.getOpt(self.network.parameters(),config)
        self.learnScheduler = NetUtils.getSch(self.opt, config)
        self.moduleNet.device = device
        if config.cuda.parallel:
            self.network = nn.DataParallel(self.moduleNet,device_ids=config.cuda.use_gpu)
        self.network = self.network.to(device)
        self.lossfunc.setConfig(config, device)
        self.lossfunc = self.lossfunc.to(device)
        self.max_norm = self.config.trainParam.clipNorm
        #self.useOrgData= self.config.model.param.useOrgData

    def _convert(self,x,containTarget=False):
        x["input"]["img"] = x["input"]["img"].to(self.device).float()
        if containTarget:
            x["target"]["num"] = x["target"]["num"].to(self.device).float()
            if self.ignoreLable:
                x["target"]["label_l"] = x["target"]["label_l"].to(self.device).gt(0.5).long()
            else:
                x["target"]["label_l"] = x["target"]["label_l"].to(self.device)
        return x

    def forward(self,x, returnLoss=False):
        return self.moduleNet(self._convert(x,returnLoss),returnLoss)

    def getLR(self):
        return self.opt.param_groups[0]['lr']

    def trainData(self,x):

        self.opt.zero_grad()
        y, loss = self.moduleNet(self._convert(x,True),True)

        loss["total_loss"].backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_norm)
        self.opt.step()

        return loss, y

    def onEpochComplete(self,epochIndex):
        self.learnScheduler.step()

    @torch.no_grad()
    def test(self,x):
        y, loss = self.moduleNet(self._convert(x,True),True)

        v = {}
        v["loss"] = loss
        v["result"] = {'pred_v':y['pred_v']}
        v["result"]["label_l"]=y["pred_label"]
        return v