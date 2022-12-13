import logging
import os
from . import UtilIO as uio
from . import VisAbstractGen
import numpy as np
import cv2
import random
import math
from util.Config import ConfigObj

class VolumeGenerator(VisAbstractGen.VisAbstractGen):

    def __init__(self, config):
        super().__init__(config,False)
        ConfigObj.default(self.param,"color.lineColor",(0,0,0))
        ConfigObj.default(self.param,"color.backColor",(255,255,255))
        ConfigObj.default(self.param,"label",1)
        ConfigObj.default(self.param,"depth",[1,21])
        ConfigObj.default(self.param, "bgcolor", 'color_pool')
        ConfigObj.default(self.param, "barcolor", 'same')
        ConfigObj.default(self.param, "barcolordark", 'no')
        ConfigObj.default(self.param, "linecolor", 'color_pool')
        ConfigObj.default(self.param, "train", True)

    def getMaxValue(self): # the maximum value of outputs
        return uio.fetchMaxValue(self.param.depth)**3

    def obliqueProjection(self, point):
        angle = -45.0
        alpha = (np.pi / 180.0) * angle
        
        P = [[1, 0, (1/2.)*np.sin(alpha)],
             [0, 1, (1/2.)*np.cos(alpha)],
             [0, 0, 0]]

        ss = np.dot(P, point)
        
        return (int(np.round(ss[1])), int(np.round(ss[0])))


    def gen(self,index,isTrainData=True):
        width = uio.fetchValue(self.param.outputWidth,1)
        height = uio.fetchValue(self.param.outputHeight,1)
        x = uio.fetchValue(self.param.x)
        y = uio.fetchValue(self.param.y)
        depth = uio.fetchValue(self.param.depth)
        lineWidth = uio.fetchValue(self.param.lineWidth)
        lineColor = tuple(self.param.color.lineColor)
        backColor = tuple(self.param.color.backColor)

        image = np.ones(shape=(width,height,3),dtype=np.int8)*255
        if self.param.train:
            backColor,lineColor,dotColor = self._genColor_element_angle()
        else:
            backColor,lineColor,dotColor = self._genTestColor_element_angle()
        # backColor,lineColor,dotColor = self._genColor_element_angle()
        backColor=uio.RGB2BGR(backColor)
        lineColor=uio.RGB2BGR(lineColor)
        self.param.color.backColor=backColor
        self.param.color.lineColor=lineColor
        image[:,:] = backColor
        
        front_bottom_left = (x, y)

        front_bottom_right = (front_bottom_left[0]+depth, front_bottom_left[1])
        front_top_left = (front_bottom_left[0],front_bottom_left[1]-depth)
        front_top_right = ( front_bottom_right[0], front_bottom_right[1]-depth)
        back_bottom_right = self.obliqueProjection([front_bottom_right[1], front_bottom_right[0], depth])
        back_top_right = (back_bottom_right[0], back_bottom_right[1]-depth)
        back_top_left = (back_top_right[0]-depth, back_top_right[1])

        cv2.line(image,front_bottom_left,front_bottom_right,lineColor,lineWidth)
        cv2.line(image,front_top_left,front_top_right,lineColor,lineWidth)
        cv2.line(image,front_top_left,front_bottom_left,lineColor,lineWidth)
        cv2.line(image,front_top_right,front_bottom_right,lineColor,lineWidth)
        cv2.line(image,front_bottom_right,back_bottom_right,lineColor,lineWidth)
        cv2.line(image,back_bottom_right,back_top_right,lineColor,lineWidth)
        cv2.line(image,back_top_right,back_top_left,lineColor,lineWidth)
        cv2.line(image,back_top_left,front_top_left,lineColor,lineWidth)
        cv2.line(image,back_top_right,front_top_right,lineColor,lineWidth)

        v = depth**3

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