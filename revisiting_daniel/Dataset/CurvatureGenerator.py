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
import skimage.draw

class CurvatureGenerator(VisAbstractGen.VisAbstractGen):

    def __init__(self, config):
        super().__init__(config,False)
        ConfigObj.default(self.param,"color.lineColor",(0,0,0))
        ConfigObj.default(self.param,"color.backColor",(255,255,255))
        ConfigObj.default(self.param,"label",1)
        ConfigObj.default(self.param,"depth",[1,81])
        ConfigObj.default(self.param,"curveWidth",[20,60])

    def getMaxValue(self): # the maximum value of outputs
        maxDepth=uio.fetchMaxValue(self.param.depth)
        minDepth=uio.fetchMinValue(self.param.depth)
        minWidth=uio.fetchMinValue(self.param.curveWidth)
        maxY=uio.fetchMaxValue(self.param.y)
        minY=uio.fetchMinValue(self.param.y)
        maxDelta = max([abs(minY-maxDepth),abs(minY-minDepth),abs(maxY-maxDepth),abs(maxY-minDepth)])

        curv = self.curvature((0,0),(maxDelta,minWidth//2),(0,minWidth))
        return curv

    def curvature(self,start,mid,end):
        t = 0.5
        P10 = (mid[0] - start[0], mid[1] - start[1])
        P21 = (end[0] - mid[0], end[1] - mid[1])
        dBt_x = 2*(1-t)*P10[1] + 2*t*P21[1]
        dBt_y = 2*(1-t)*P10[0] + 2*t*P21[0]
        dBt2_x = 2*(end[1] - 2*mid[1] + start[1])
        dBt2_y = 2*(end[0] - 2*mid[0] + start[0])
        v = np.abs((dBt_x*dBt2_y - dBt_y*dBt2_x) / ((dBt_x**2 + dBt_y**2)**(3/2.)))
        return v

    def gen(self,index,isTrainData=True):
        width = uio.fetchValue(self.param.outputWidth,1)
        height = uio.fetchValue(self.param.outputHeight,1)
        x = uio.fetchValue(self.param.x)
        y = uio.fetchValue(self.param.y)
        depth = uio.fetchValue(self.param.depth)
        #lineWidth = uio.fetchValue(self.param.lineWidth)
        lineColor = tuple(self.param.color.lineColor)
        curveWidth = uio.fetchValue(self.param.curveWidth)
        backColor = tuple(self.param.color.backColor)

        image = np.ones(shape=(width,height,3),dtype=np.int8)*255
        image[:,:] = backColor
        
        start = (y,x)
        mid = (depth,x+curveWidth//2)
        end = (y,x+curveWidth)
        

        rr, cc = skimage.draw.bezier_curve(start[0], start[1], mid[0], mid[1], end[0], end[1], 1)
        image[rr, cc] = lineColor
        
        v = self.curvature(start,mid,end)

        v = np.round(v, 3)

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