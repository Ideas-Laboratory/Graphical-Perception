import numpy as np 
import random

class MADSParam:
    def __init__(self,tp,initValue,lowBound,upBound):
        self.tp=tp
        self.initValue=tp(initValue)
        self.lowBound = lowBound
        self.upBound = upBound

# reference: https://arxiv.org/pdf/2104.11627.pdf
# https://epubs.siam.org/doi/pdf/10.1137/080716980
# We use https://github.com/bbopt/nomad
"""
class MADS:

    # param list:
    #   each item: MADSParam
    # blackFunction: accept param indicated in paramList
    def __init__(self, paramList, blackFunction):
        self.paramList = paramList
        self.blackFunction = blackFunction
        self.paramCount = len(self.paramList)

    def _getInitParams(self):
        ls = []
        for i in self.paramList:
            ls.append(i.initValue)
        return ls

    def _addParam(self,params,addValues, scalar=1.0):
        newParams=[]
        for i in range(self.paramCount):
            newv = params[i]+addValues[i]*sclar
            newParams.append(newv)
        return newParams

    def _genParamDirectionSet(self,params):
        # +1, -1 for each direction

    def run(self,frameSize = 1.0,sizeAdjustInit=0.5,stopCondition=1e-5):
        delta=frameSize
        param = self._getInitParams()
        sizeAdjust = sizeAdjustInit
        iterCount=0
        while True:
            meshSizeParam = min(delta,delta**2)
            pass
        pass
"""