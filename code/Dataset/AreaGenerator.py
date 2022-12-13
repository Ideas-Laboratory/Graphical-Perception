import logging
import os
from . import UtilIO as uio
from . import VisAbstractGen
import numpy as np
import cv2
import random
import math
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
class AreaGenerator(VisAbstractGen.VisAbstractGen):

    def __init__(self, config):
        super().__init__(config,False)
        ConfigObj.default(self.param,"color.lineColor",(0,0,0))
        ConfigObj.default(self.param,"color.backColor",(255,255,255))
        ConfigObj.default(self.param,"color.circleColor",(0,0,0))
        ConfigObj.default(self.param,"label",1)
        ConfigObj.default(self.param, "bgcolor", 'color_pool')
        ConfigObj.default(self.param, "barcolordark", 'no')
        ConfigObj.default(self.param, "barcolor", 'same')
        ConfigObj.default(self.param, "linecolor", 'color_pool')
        ConfigObj.default(self.param, "train", True)
        
    def getMaxValue(self): # the maximum value of outputs
        return (uio.fetchMaxValue(self.param.radius)**2)*3.14159265358979

    def gen(self,index,isTrainData=True):
        width = uio.fetchValue(self.param.outputWidth,1)
        height = uio.fetchValue(self.param.outputHeight,1)
        x = uio.fetchValue(self.param.x)
        y = uio.fetchValue(self.param.y)
        lineWidth = uio.fetchValue(self.param.lineWidth)
        radius = uio.fetchValue(self.param.radius)
        backColor = tuple(self.param.color.backColor)

        image = np.ones(shape=(width,height,3),dtype=np.int8)*255

        if self.param.train:
            backColor,barColor,lineColor,fill = self._genColor_element_area()
        else:
            backColor,barColor,lineColor,fill = self._genTestColor_element_area()
        # backColor,barColor,lineColor,fill = self._genColor_element_area()
        backColor=uio.RGB2BGR(backColor)
        barColor=uio.RGB2BGR(barColor)
        lineColor=uio.RGB2BGR(lineColor)
        self.param.color.backColor=backColor
        self.param.color.circleColor=lineColor

        image[:,:] = backColor
        cv2.ellipse(image,(x,y),(radius,radius),0,0,360,self.param.color.circleColor,-1 if fill else lineWidth)

        v = radius*radius*3.14159265358979
        
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