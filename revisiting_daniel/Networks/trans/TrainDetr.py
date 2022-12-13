from ..trans import detr
from ..trans import backbone
from ..trans import transformer
from ..trans import position_encoding
import torch.nn as nn
import torch
from .. import NetUtils
from Config import obj2dic

class TrainDetr(nn.Module):

    def __init__(self,param):
        super(TrainDetr,self).__init__()
        self.param=param

        # build backbone
        #backbone = backbone.build_backbone(param)

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

        self.network = detr.DETR(
            joiner,
            trans_model,
            mid_dim=param.detr.mid_dim,
            num_queries=param.detr.num_queries
        )

        self.lossfunc = detr.SetCriterion(**obj2dic(param.lossFunction))
        
        pass

    def setConfig(self,config,device):
        self.config=config
        self.device=device
        self.opt = NetUtils.getOpt(self.network.parameters(),config)
        self.learnScheduler = NetUtils.getSch(self.opt, config)
        if config.cuda.parallel:
            self.network = nn.DataParallel(self.network,device_ids=config.cuda.use_gpu)
        self.network = self.network.to(device)
        self.lossfunc.setConfig(config, device)
        self.lossfunc = self.lossfunc.to(device)
        self.max_norm = self.config.trainParam.clipNorm
        self.useOrgData= self.config.model.param.useOrgData

    def forward(self,x):
        return self._postprocess(self.network(x))

    def getLR(self):
        return self.opt.param_groups[0]['lr']

    def _getTarget(self,x):
        if self.useOrgData:
            target_raw = x["target"]["num"][0]
            target = [torch.Tensor(i).to(self.device) for i in target_raw]
            return self.lossfunc.preprocessTargetValue(target)
        else:
            target_raw = x["target"]["ratio"]
            target = [torch.Tensor(i).to(self.device) for i in target_raw]
            return target

    def _postprocess(self,y):
        if self.useOrgData:
            y["pred_v2"] = self.lossfunc.retainOrgValue(y["pred_v"])
        return y

    def trainData(self,x):
        #with torch.autograd.set_detect_anomaly(True):
        input = x["input"]["img"].to(self.device)
        target = self._getTarget(x)

        self.opt.zero_grad()

        y = self._postprocess(self.network(input))

        loss = self.lossfunc(y,target,"pred_v2" if self.useOrgData else "pred_v")
        loss["total_loss"].backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_norm)
        self.opt.step()

        return loss, y

    def onEpochComplete(self,epochIndex):
        self.learnScheduler.step()

    @torch.no_grad()
    def test(self,x):
        input = x["input"]["img"].to(self.device)
        target = self._getTarget(x)

        y = self._postprocess(self.network(input))
        loss = self.lossfunc(y,target,"pred_v2" if self.useOrgData else "pred_v")

        v = {}
        v["loss"] = loss
        v["result"] = {'pred_v':y['pred_v'][0]}
        return v