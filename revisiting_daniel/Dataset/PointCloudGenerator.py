import logging
import os
from . import UtilIO as uio
from . import VisAbstractGen
import numpy as np
import cv2
import random
from util.Config import ConfigObj

class PointCloudGenerator(VisAbstractGen.VisAbstractGen):

    def __init__(self, config):
        super().__init__(config,False)
        ConfigObj.default(self.param,"color.dotColor",[0,0,0])
        ConfigObj.default(self.param,"color.backColor",[255,255,255])
        ConfigObj.default(self.param,"label",1)
        ConfigObj.default(self.param,"outputAddValue",True)
        ConfigObj.default(self.param,"preprocess.enable",False)

    def getMaxValue(self): # the maximum value of outputs
        v=uio.fetchMaxValue(self.param.pointCount)
        if self.param.outputAddValue:
            v = uio.fetchMaxValue(self.param.pointCountAdd)
        return v
        
    def gen(self,index,isTrainData=True):
        width = uio.fetchValue(self.param.outputWidth,1)
        height = uio.fetchValue(self.param.outputHeight,1)
        pointCount = uio.fetchValue(self.param.pointCount)
        pointCountAdd = random.randint(0,self.param.pointCountAdd+1)
        dotColor = self.param.color.dotColor
        backColor = self.param.color.backColor

        totalCount = pointCount+pointCountAdd

        image = np.ones(shape=(width,height,3),dtype=np.int8)*255
        image[:,:] = backColor
        imageTag = np.zeros(shape=(width,height),dtype=np.bool)


        if width*height<totalCount:
            logging.error("Cannot generate point cloud, pixel %d, points %d"%(width*height,totalCount))

        curCount=0
        while curCount<totalCount:
            x = random.randint(0,width-1)
            y = random.randint(0,height-1)
            if imageTag[y,x]:
                continue
            image[y,x]=dotColor
            imageTag[y,x]=True
            curCount+=1
        
        # save
        inputFilePath,outputFilePath,orgFilePath = self._getFilePath(isTrainData)

        fileName = self.param.fileName%index
        inputFilePath = os.path.join(inputFilePath,fileName)
        outputFilePath = os.path.join(outputFilePath,fileName)
        orgFilePath = os.path.join(orgFilePath,fileName)

        cv2.imwrite(inputFilePath+".png",image)
        
        v=totalCount
        if self.param.outputAddValue:
            v = pointCountAdd
        uio.save(outputFilePath,[v],"json")
        uio.save(outputFilePath+"_r",[v],"json")
        uio.save(outputFilePath+"_l",[self.param.label],"json")
        uio.save(outputFilePath+"_ll",[self.param.label],"json")