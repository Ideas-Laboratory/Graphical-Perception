import numpy as np
import torch
import util.shareCode
from util.shareCode import programInit, globalCatch
from util import Config
from util.Config import ConfigObj
from util.Config import setConfig
import os
import logging
import argparse
import MainTrain
import RunTest
import copy

# use program: https://www.gerad.ca/nomad/Downloads/user_guide.pdf
# for NOMAD 3.9
class HyperparamManager:

    def __init__(self,config,device):
        self.config =config
        self.device = device
        self.params = self.config.MADS.params

    def setParam(self,values):
        tempConfig = copy.deepcopy(self.config)
        if len(values)!=len(self.params):
            logging.warning("unmatch vector length!")
        for i in range(len(values)):
            attr = self.params[i]
            if not self._checkRange(attr.range,values[i]):
                return None
            v = self._castType(values[i],attr.type)
            setConfig(tempConfig,attr.name,v)
        return tempConfig

    def _list2str(self,l):
        s=""
        for i in l:
            s+=" "+i
        s="(%s )"%s
        return s

    def genParamFile(self,path):
        s="DIMENSION %d\n"%len(self.params)
        s+="BB_EXE \"$%s\"\n"%self.MADS.invokeProgram
        s+="BB_OUTPUT_TYPE OBJ\n"
        # generate lower bound
        lowBound=[]
        upBound=[]
        x0=[]
        for attr in self.params:
            r = attr.range
            if r is None:
                r=["-inf","+inf"]
            if r[0]=="-inf":
                lowBound.append("-")
            else:
                lowBound.append(str(r[1]))
            if r[1] in ["inf", "+inf"]:
                upBound.append("-")
            else:
                upBound.append("-")
            x0.append(str(attr.initValue))

        s+="X0 %s\n"%self._list2str(x0)
        s+="LOWER_BOUND %s\n"%self._list2str(lowBound)
        s+="UPPER_BOUND %s\n"%self._list2str(upBound)
        s+="MAX_BB_EVAL %d\n"%self.config.MADS.maxEval

        f = open(path,"w")
        f.write(s)
        f.close()
        
        pass

    def _checkRange(self,ranges,value):
        if ranges is None:
            return True
        if ranges[0] == "-inf" and ranges[1] in ["inf","+inf"]:
            return True
        a1 = ranges[0]=="-inf"
        if not a1:
            a1 = ranges[0]<=value
        a2 = ranges[1] in ["inf","+inf"]
        if not a2:
            a2 = ranges[1]>=value
        return a1 and a2
    
    def _castType(self,v,tp):
        if tp=="int":
            return int(v)
        elif tp=="float":
            return float(v)
        elif tp=="string":
            return str(v)
        elif tp.startswith("categorical"):
            v=int(round(v)+1e-5)
            ls = self.config.MADS.categoricalList
            return getattr(ls,tp)[v]
        else:
            logging.warning("Incorrect cast for %s to %s"%(str(v),tp))
        return None

    def runProgram(self):
        tempDir = self.config.MADS.tempDir
        os.makedirs(tempDir,exist_ok=True)
        os.system(self.config.MADS.programPath+" "+)

def blackFunction(config,device):
    th = MainTrain.TrainHelper(device,config.trainConfig)
    th.train()
    logging.info("Train Complete")
    th.save()
    RunTest.runTest(config.testConfig,device)

def main():
    config = programInit()

    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda.detectableGPU
    device = torch.device("cpu")
    if torch.cuda.is_available():
        logging.info("Detect GPU, Use gpu to train the model")
        device = torch.device("cuda")
    else:
        logging.warning("Cannot detect gpu, use cpu")
    
    

if __name__=="__main__":
    globalCatch(main)