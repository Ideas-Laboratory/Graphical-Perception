import logging
import os
from . import UtilIO as uio
from . import VisAbstractGen
import numpy as np
import cv2
import random
import math
from math import radians, pi
from util.Config import ConfigObj

class PieGenerator(VisAbstractGen.VisAbstractGen):

    def __init__(self, config):
        super().__init__(config)
        
        
    def mark(self,image,fromRad,toRad,midPos,radius,dotColor):
        while abs(toRad-fromRad)>2*3.14159265358979:
            if toRad<fromRad:
                toRad+=2*3.14159265358979
            else:
                fromRad+=2*3.14159265358979
        midRad = (toRad-fromRad)*0.5+fromRad
        midRadius = radius * 0.5
        y = int(midPos[1]+math.sin(midRad)*midRadius)
        x = int(midPos[0]+math.cos(midRad)*midRadius)
        if y<0 or x<0 or x+1>=image.shape[0] or y+1>=image.shape[0]:
            logging.warning("Pie Generator: Wrong dot Position %d %d"%(x,y))
        else:
            image[y:y+1,x:x+1] = (dotColor[0],dotColor[1],dotColor[2])

    def gen(self,index,isTrainData=True):
        width = uio.fetchValue(self.param.outputWidth,1)
        height = uio.fetchValue(self.param.outputHeight,1)
        pieCount = uio.fetchValue(self.param.pieCount)
        pieRadius = uio.fetchValue(self.param.pieRadius,1)
        lineThickness = uio.fetchValue(self.param.lineThickness,-1)
        centerPosX = uio.fetchValue(self.param.centerPosX)
        centerPosY = uio.fetchValue(self.param.centerPosY)
        pieInitDegreeRatio = uio.fetchValue(self.param.pieInitDegreeRatio,types=float)

        if pieRadius*2>width or pieRadius*2>height:
            #logging.warning("Wrong Parameters! Too large pie radius!")
            pieRadius = min(width,height)//2
        
        if centerPosX<pieRadius:
            centerPosX=pieRadius
        if centerPosX+pieRadius>width:
            centerPosX = width-pieRadius
        
        if centerPosY<pieRadius:
            centerPosX=pieRadius
        if centerPosY+pieRadius>height:
            centerPosY = height-pieRadius

        # process color
        colorLists,backColor,fill, strokeColor = self._genColor(pieCount)

        
        pieWeights = self._genValues(pieCount)
        if not self.param.values.pixelValue:
            quantv = 3.5/2.0/3.141592655358979/pieRadius #avoid lines overlaps, compute min arc length
            if lineThickness>0:
                quantv*=lineThickness
            quantv = int(1.0/quantv)

            maxv = max(pieWeights)
            pieWeights = [i/maxv for i in pieWeights]
            totalV = sum(pieWeights)
            pieWeights = [max(1,int(i/totalV*quantv))/quantv for i in pieWeights]
            totalV = sum(pieWeights)
            pieWeights = [i/totalV for i in pieWeights]
        else:
            totalV = sum(pieWeights)
            pieWeights = [i/totalV for i in pieWeights]
        markList = self._mark(pieWeights)

        image = np.ones(shape=(width,height,3),dtype=np.int8)*255
        image[:,:] = backColor
        
        initDeg = -pieInitDegreeRatio*pieWeights[0]*360
        for i in range(pieCount):
            endDeg = pieWeights[i]*360+initDeg
            useColor = colorLists[i]
            #if fill:
            cv2.ellipse(image,(centerPosX,centerPosY),(pieRadius,pieRadius),0,initDeg,endDeg,useColor,-1 if fill else lineThickness)
            if lineThickness>0: # non-fill mode
                useColor=strokeColor[0]
                cv2.line(image,
                    (centerPosX,centerPosY),
                    (int(centerPosX+pieRadius*math.cos(math.radians(initDeg))),
                    int(centerPosY+pieRadius*math.sin(math.radians(initDeg)))),useColor,lineThickness)
                cv2.line(image,
                    (centerPosX,centerPosY),
                    (int(centerPosX+pieRadius*math.cos(math.radians(endDeg))),
                    int(centerPosY+pieRadius*math.sin(math.radians(endDeg)))),useColor,lineThickness)
                cv2.ellipse(image,(centerPosX,centerPosY),(pieRadius,pieRadius),0,initDeg,endDeg,useColor, lineThickness)
            if i in markList:
                self.mark(image,math.radians(initDeg),math.radians(endDeg),(centerPosX,centerPosY),pieRadius,self.param.mark.dotColor)
            initDeg=endDeg
            
            
        # save
        inputFilePath,outputFilePath,orgFilePath = self._getFilePath(isTrainData)

        fileName = self.param.fileName%index
        inputFilePath = os.path.join(inputFilePath,fileName)
        outputFilePath = os.path.join(outputFilePath,fileName)
        orgFilePath = os.path.join(orgFilePath,fileName)

        self._preprocess(inputFilePath,image)

        pieWeights = self._processValues(pieWeights,markList)

        if self.param.mark.ratio.ratioNotMarkOnly:
            ind = markList[0]
            pieWeights = pieWeights[ind:]+pieWeights[0:ind]
            uio.save(outputFilePath,pieWeights,"json")
            uio.save(outputFilePath+"_ll",[self.param.labelValue]*len(pieWeights),"json")
        elif self.param.mark.ratio.ratio2Only:
            uio.save(outputFilePath,[pieWeights[0]/pieWeights[1]],"json")
            uio.save(outputFilePath+"_ll",[self.param.labelValue],"json")
        else:
            uio.save(outputFilePath,pieWeights,"json")
            uio.save(outputFilePath+"_ll",[self.param.labelValue]*len(pieWeights),"json")

        ratio = self._genRatio(pieWeights)
        uio.save(outputFilePath+"_r",ratio,"json")

        labels = [self.param.labelValue]*len(ratio)
        if self.param.mark.ratio.ratio2Only:
            uio.save(outputFilePath+"_l",[labels[0]],"json")
        else:
            uio.save(outputFilePath+"_l",labels,"json")

