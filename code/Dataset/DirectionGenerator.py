import logging
import os
from . import UtilIO as uio
from . import VisAbstractGen
import numpy as np
import cv2
import random
import math
from util.Config import ConfigObj

class DirectionGenerator(VisAbstractGen.VisAbstractGen):

    def __init__(self, config):
        super().__init__(config,False)
        ConfigObj.default(self.param,"color.lineColor",(0,0,0))
        ConfigObj.default(self.param,"color.backColor",(255,255,255))
        ConfigObj.default(self.param,"color.spotColor",(0,0,0))
        ConfigObj.default(self.param,"label",1)
        ConfigObj.default(self.param, "bgcolor", 'color_pool')
        ConfigObj.default(self.param, "barcolor", 'same')
        ConfigObj.default(self.param, "barcolordark", 'no')
        ConfigObj.default(self.param, "linecolor", 'color_pool')
        ConfigObj.default(self.param, "train", True)
        
    def getMaxValue(self): # the maximum value of outputs
        return 360.0

    def spot(self,image,x,y,spotSize):
        half = spotSize//2
        image[y-half:y+half+1,x-half:x+half+1] = tuple(self.param.color.spotColor)

    def gen(self,index,isTrainData=True):
        width = uio.fetchValue(self.param.outputWidth,1)
        height = uio.fetchValue(self.param.outputHeight,1)
        x = uio.fetchValue(self.param.x)
        y = uio.fetchValue(self.param.y)
        length = uio.fetchValue(self.param.length)
        lineWidth = uio.fetchValue(self.param.lineWidth)
        spotSize = uio.fetchValue(self.param.spotSize)
        backColor = tuple(self.param.color.backColor)
        dirVary = uio.fetchValue(self.param.dirVary)

        
        deg = int(random.random()*360*dirVary)/dirVary
        rad = math.radians(deg)

        image = np.ones(shape=(width,height,3),dtype=np.int8)*255

        # backColor,strokeColor,dotColor = self._genColor_element_angle()
        if self.param.train:
            backColor,strokeColor,dotColor = self._genColor_element_angle()
        else:
            backColor,strokeColor,dotColor = self._genTestColor_element_angle()
        backColor=uio.RGB2BGR(backColor)
        strokeColor=uio.RGB2BGR(strokeColor)
        dotColor=uio.RGB2BGR(dotColor)
        self.param.color.backColor=backColor
        self.param.color.lineColor=strokeColor
        self.param.color.spotColor=dotColor

        image[:,:] = backColor
        cv2.line(image,(x,y),(x-int(math.cos(rad)*length),y-int(math.sin(rad)*length)),tuple(self.param.color.lineColor),lineWidth)

        self.spot(image,x,y,spotSize)

        v = deg
        
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