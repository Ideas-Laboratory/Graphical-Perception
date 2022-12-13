import logging
import os
from . import UtilIO as uio
from . import VisAbstractGen
import numpy as np
import cv2
import random
import math
from util.Config import ConfigObj
import skimage

class ShadingGenerator(VisAbstractGen.VisAbstractGen):

    def __init__(self, config):
        super().__init__(config,False)
        ConfigObj.default(self.param,"color.lineColor",(0,0,0))
        ConfigObj.default(self.param,"color.backColor",(255,255,255))
        ConfigObj.default(self.param,"label",1)
        ConfigObj.default(self.param,"covered",[1,101])

    def getMaxValue(self): # the maximum value of outputs
        return uio.fetchMaxValue(self.param.covered)

    def gen(self,index,isTrainData=True):
        width = uio.fetchValue(self.param.outputWidth,1)
        height = uio.fetchValue(self.param.outputHeight,1)
        x = uio.fetchValue(self.param.x)
        y = uio.fetchValue(self.param.y)
        lineColor = tuple(self.param.color.lineColor)
        covered = uio.fetchValue(self.param.covered)
        backColor = tuple(self.param.color.backColor)

        image = np.ones(shape=(width,height,3),dtype=np.int8)*255
        image[:,:] = backColor
        
        step = max(1,100-covered)
        for i in range(width):
            for j in range(height):
                if ((i+j+x)%step==0) or ((i-j+y) % step == 0):
                    image[i,j]=lineColor 
        v = covered

        # save
        inputFilePath,outputFilePath,orgFilePath = self._getFilePath(isTrainData)

        fileName = self.param.fileName%index
        inputFilePath = os.path.join(inputFilePath,fileName)
        outputFilePath = os.path.join(outputFilePath,fileName)
        orgFilePath = os.path.join(orgFilePath,fileName)

        cv2.imwrite(inputFilePath+".png",image)
        
        uio.save(outputFilePath,[v],"json")
        uio.save(outputFilePath+"_r",[v],"json")
        uio.save(outputFilePath+"_l",[self.param.label],"json")
        uio.save(outputFilePath+"_ll",[self.param.label],"json")