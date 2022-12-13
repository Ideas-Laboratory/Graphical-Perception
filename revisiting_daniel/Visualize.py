import os
import logging
import util
import time
import random
import util.pyutils as upy
from util import Config
import util.shareCode
from util.shareCode import programInit, globalCatch
from visdom import Visdom
import Dataset
import Dataset.UtilIO as uio


class VisualizeNetwork:

    def __init__(self, device, config):
        self.config=config
        self.device=device
        ConfigObj.default(self.config,"utility.visdomPort",8097)
        #ConfigObj.default(self.config,"saveFolder","")
        pass

    def loadModel(self,modelPath,modelClassName):
        singleModelName = modelClassName.split(".")[-1]
        modelName = "Networks.%s.%s"%(self.config.model.name,singleModelName)
        logging.info("Create Model %s"%modelName)
        modelClass=multiImport(modelName)
        self.model=None

        self.model = self.model.cpu()
        logging.info("Load model %s"%modelName)
        self.model.load_state_dict(torch.load(modelName))
        self.model=self.model.to(self.device)

    def loadData(self,dataListFile):
        uio.load()
        pass






def main():
    #parse params
    config = programInit()

    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda.detectableGPU
    device = torch.device("cpu")
    if torch.cuda.is_available():
        logging.info("Detect GPU, Use gpu to train the model")
        device = torch.device("cuda")
    else:
        logging.warning("Cannot detect gpu, use cpu")
    
    pass

if __name__ == '__main__':
    globalCatch(main)
