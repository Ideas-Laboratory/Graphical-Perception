import logging
import os
from . import UtilIO as uio
from . import VisAbstractGen
import numpy as np
import cv2
import random
from util.Config import ConfigObj
'''
{
    "xAxis":10,
    "x":[20,80],
    "yRange":[20,80],
    "lineWidth":[1,11],
    "offset":0 # aligned or not aligned
}
'''
class PositionScaleGenerator(VisAbstractGen.VisAbstractGen):

    def __init__(self, config):
        super().__init__(config,False)
        ConfigObj.default(self.param,"color.lineColor",(0,0,0))
        ConfigObj.default(self.param,"color.spotColor",(0,0,0))
        ConfigObj.default(self.param,"color.backColor",(255,255,255))
        ConfigObj.default(self.param,"label",1)
        
    def getMaxValue(self): # the maximum value of outputs
        return abs(self.param.yRange[1]-self.param.yRange[0])

    def spot(self,image,x,y,spotSize):
        half = spotSize//2
        image[y-half:y+half+1,x-half:x+half+1] = tuple(self.param.color.spotColor)

    def gen(self,index,isTrainData=True):
        width = uio.fetchValue(self.param.outputWidth,1)
        height = uio.fetchValue(self.param.outputHeight,1)
        xAxis = uio.fetchValue(self.param.xAxis)
        x = uio.fetchValue(self.param.x)
        offset = uio.fetchValue(self.param.offset,-height)
        ya=self.param.yRange[0]-offset
        yb=self.param.yRange[1]-offset
        y = uio.fetchValue([ya,yb])
        lineWidth = uio.fetchValue(self.param.lineWidth)
        backColor = tuple(self.param.color.backColor)
        
        spotSize = 0
        spotSize = uio.fetchValue(self.param.spotSize)
        while spotSize%2==0:
            spotSize = uio.fetchValue(self.param.spotSize)

        image = np.ones(shape=(width,height,3),dtype=np.int8)*255
        image[:,:] = backColor
        cv2.line(image,(xAxis,ya),(xAxis,yb),tuple(self.param.color.lineColor),lineWidth)

        half = spotSize//2
        self.spot(image,x,y,spotSize)


        v = y-ya

        
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