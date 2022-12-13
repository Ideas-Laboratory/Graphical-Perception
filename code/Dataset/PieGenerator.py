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
from PIL import Image, ImageFont, ImageDraw
import sys
sys.path.append("..")
from util.color_pool import *

class PieGenerator(VisAbstractGen.VisAbstractGen):

    def __init__(self, config):
        super().__init__(config)
        ConfigObj.default(self.param, "TitlePosition", 'mid')
        ConfigObj.default(self.param, "TitleFontSize", 12)
        ConfigObj.default(self.param, "TitleFontType", 'arial')
        ConfigObj.default(self.param, "Direction", 'vertical')
        ConfigObj.default(self.param, "bgcolor", 'color_pool')
        ConfigObj.default(self.param, "barcolor", 'same')
        ConfigObj.default(self.param, "barcolordark", 'no')
        ConfigObj.default(self.param, "linecolor", 'color_pool')
        ConfigObj.default(self.param, "xTickNumber", 'retain')
        ConfigObj.default(self.param, "TitleLength", 1)
        ConfigObj.default(self.param, "train", True)
        ConfigObj.default(self.param, "TitlePaddingLeft", 0.1)
        ConfigObj.default(self.param, "lightness_pertubation", 0)
        ConfigObj.default(self.param, "bgcolor_pertubation", 0)
        ConfigObj.default(self.param, "barcolor_pertubation", 0)
        ConfigObj.default(self.param, "strokecolor_pertubation", 0)
        ConfigObj.default(self.param, "mark.markAdjancy", False)
        ConfigObj.default(self.param, "mark.markStackedAdjancy", False)
        ConfigObj.default(self.param, "mark.randPos", False)
        ConfigObj.default(self.param, "mark.markSize", 1)
        ConfigObj.default(self.param, "mark.dotDeviation", 0)
        ConfigObj.default(self.param, "mask.isFlag", False)
        ConfigObj.default(self.param, "mask.type", "contour")
        ConfigObj.default(self.param, "changeTargetOnly", False)
        ConfigObj.default(self.param, "bgcolorL_perturbation", 0)
        ConfigObj.default(self.param, "bgcolorA_perturbation", 0)
        ConfigObj.default(self.param, "bgcolorB_perturbation", 0)
        ConfigObj.default(self.param, "LABperturbation", False)
        ConfigObj.default(self.param, "strokecolorL_perturbation", 0)
        ConfigObj.default(self.param, "strokecolorA_perturbation", 0)
        ConfigObj.default(self.param, "strokecolorB_perturbation", 0)
        ConfigObj.default(self.param, "strokeLABperturbation", False)
        ConfigObj.default(self.param, "barcolorL_perturbation", 0)
        ConfigObj.default(self.param, "barcolorA_perturbation", 0)
        ConfigObj.default(self.param, "barcolorB_perturbation", 0)
        ConfigObj.default(self.param, "barLABperturbation", False)
        
        
    def mark(self,image,fromRad,toRad,midPos,radius,dotColor):
        while abs(toRad-fromRad)>2*3.14159265358979:
            if toRad<fromRad:
                toRad+=2*3.14159265358979
            else:
                fromRad+=2*3.14159265358979
        midRad = (toRad-fromRad)*0.5+fromRad
        midRadius = radius * 0.5
        y = int(midPos[1]+math.sin(midRad)*midRadius)
        x = int(midPos[0]+math.cos(midRad)*midRadius) + self.param.mark.dotDeviation
        if y<0 or x<0 or x+1>=image.shape[0] or y+1>=image.shape[0]:
            logging.warning("Pie Generator: Wrong dot Position %d %d"%(x,y))
        else:
            image[y:y+self.param.mark.markSize,x:x+self.param.mark.markSize]=(dotColor[0],dotColor[1],dotColor[2])
            # image[y:y+1,x:x+1] = (dotColor[0],dotColor[1],dotColor[2])
        # return x,y

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
        # colorLists,backColor,fill, strokeColor = self._genColor(pieCount)
        if self.param.train:
            colorLists, backColor, fill, strokeColor = self._genTrainColor(pieCount,index)
        else:
            colorLists, backColor, fill, strokeColor = self._genTestColor_pie(pieCount,index)
        # print(strokeColor)
        # colorLists=uio.RGB2BGR(colorLists)
        # backColor=uio.RGB2BGR(backColor)
        # strokeColor=uio.RGB2BGR(strokeColor)
        
        lightness_pertubation=self.param.lightness_pertubation
        bgcolor_pertubation=self.param.bgcolor_pertubation
        barcolor_pertubation=self.param.barcolor_pertubation
        strokecolor_pertubation=self.param.strokecolor_pertubation

        backColor=np.array(backColor)-lightness_pertubation+bgcolor_pertubation
        backColor[backColor<0]=0
        backColor[backColor>255]=255
        backColor=tuple(backColor.tolist())
        if self.param.LABperturbation:
            temp_=uio.RGB2Lab(backColor)
            temp_[0]=temp_[0]+self.param.bgcolorL_perturbation
            temp_[1]=temp_[1]+self.param.bgcolorA_perturbation
            temp_[2]=temp_[2]+self.param.bgcolorB_perturbation
            backColor=uio.Lab2RGB(temp_)
        # fill=tuple((np.array(fill)-100).tolist())
        strokeColor=np.array(strokeColor)-lightness_pertubation+strokecolor_pertubation
        strokeColor[strokeColor<0]=0
        strokeColor[strokeColor>255]=255
        strokeColor=tuple(strokeColor.tolist())
        if self.param.strokeLABperturbation:
            # print(strokeColor)
            temp_=uio.RGB2Lab(strokeColor[0])
            temp_[0]=temp_[0]+self.param.strokecolorL_perturbation
            temp_[1]=temp_[1]+self.param.strokecolorA_perturbation
            temp_[2]=temp_[2]+self.param.strokecolorB_perturbation
            strokeColor=tuple([int(xxx) for xxx in uio.RGB2BGR(uio.Lab2RGB(temp_))])
        for ccc in range(len(colorLists)):
            # colorLists[ccc]=tuple((np.array(colorLists[ccc])-lightness_pertubation+barcolor_pertubation).tolist())
            # colorLists[ccc]=np.array(colorLists[ccc])-lightness_pertubation+barcolor_pertubation
            # colorLists[ccc][colorLists[ccc]<0]=0
            # colorLists[ccc][colorLists[ccc]>255]=255
            # colorLists[ccc]=tuple(colorLists[ccc].tolist())
            if self.param.barLABperturbation:
                temp_=uio.RGB2Lab(colorLists[ccc])
                temp_[0]=temp_[0]+self.param.barcolorL_perturbation
                temp_[1]=temp_[1]+self.param.barcolorA_perturbation
                temp_[2]=temp_[2]+self.param.barcolorB_perturbation
                colorLists[ccc]=tuple([int(xxx) for xxx in uio.RGB2BGR(uio.Lab2RGB(temp_))])
        
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
        
        mask_image=Image.fromarray(np.zeros((100,100)))
        mask_draw = ImageDraw.ImageDraw(mask_image)

        # image = Image.new('RGB', (width,height), backColor)
        # draw = ImageDraw.ImageDraw(image)
        # # draw.ellipse(((centerPosX-pieRadius,centerPosY-pieRadius), (centerPosX+pieRadius,centerPosY+pieRadius)), fill=useColor)

        # initDeg = -pieInitDegreeRatio*pieWeights[0]*360
        # uselineColor=tuple(strokeColor[0])
        # for i in range(pieCount):
        #     endDeg = pieWeights[i]*360+initDeg
        #     useColor = tuple(colorLists[i])
        #     # print(useColor)
        #     if not fill:
        #         draw.pieslice(((centerPosX-pieRadius,centerPosY-pieRadius), (centerPosX+pieRadius,centerPosY+pieRadius)), start=initDeg,end=endDeg,fill=None)
        #     else:
        #         draw.pieslice(((centerPosX-pieRadius,centerPosY-pieRadius), (centerPosX+pieRadius,centerPosY+pieRadius)), start=initDeg,end=endDeg,fill=useColor)
        #     if lineThickness>0:
        #         # useColor=tuple(strokeColor[0])
        #         uselineColor=tuple(strokeColor[0])
        #         print(useColor)
        #         draw.line(((centerPosX,centerPosY),
        #             (int(centerPosX+pieRadius*math.cos(math.radians(initDeg))),
        #             int(centerPosY+pieRadius*math.sin(math.radians(initDeg))))),fill=uselineColor,width=lineThickness)
        #         draw.line(((centerPosX,centerPosY),
        #             (int(centerPosX+pieRadius*math.cos(math.radians(endDeg))),
        #             int(centerPosY+pieRadius*math.sin(math.radians(endDeg))))),fill=uselineColor,width=lineThickness)
        #         if index==13:
        #             print(initDeg)
        #             print(endDeg)
        #         draw.arc(((centerPosX-pieRadius,centerPosY-pieRadius), (centerPosX+pieRadius,centerPosY+pieRadius)), start=initDeg,end=endDeg,fill=uselineColor, width=lineThickness)
        #         if i in markList:
        #             mask_draw.line([(centerPosX,centerPosY),
        #                 (int(centerPosX+pieRadius*math.cos(math.radians(initDeg))),
        #                 int(centerPosY+pieRadius*math.sin(math.radians(initDeg))))],fill=0,width=1)
        #     if i in markList:
        #         dotColor=self.param.mark.dotColor
        #         mask_draw.pieslice([(10,10),(90,90)],initDeg,endDeg, fill =255)
        #         # mark_x,mark_y=self.mark(image,math.radians(initDeg),math.radians(endDeg),(centerPosX,centerPosY),pieRadius,self.param.mark.dotColor)
        #         # draw.point((mark_x, mark_y), fill = (dotColor[0], dotColor[1], dotColor[2]))
        #     initDeg=endDeg
        self.param.mark.randPos=False
        
        initDeg = -pieInitDegreeRatio*pieWeights[0]*360
        for i in range(pieCount):
            endDeg = pieWeights[i]*360+initDeg
            useColor = colorLists[i]
            #if fill:
            cv2.ellipse(image,(centerPosX,centerPosY),(pieRadius,pieRadius),0,initDeg,endDeg,useColor,-1 if fill else lineThickness)
            
            if lineThickness>0: # non-fill mode
                useColor=strokeColor
                cv2.line(image,
                    (centerPosX,centerPosY),
                    (int(centerPosX+pieRadius*math.cos(math.radians(initDeg))),
                    int(centerPosY+pieRadius*math.sin(math.radians(initDeg)))),useColor,lineThickness)
                cv2.line(image,
                    (centerPosX,centerPosY),
                    (int(centerPosX+pieRadius*math.cos(math.radians(endDeg))),
                    int(centerPosY+pieRadius*math.sin(math.radians(endDeg)))),useColor,lineThickness)
                cv2.ellipse(image,(centerPosX,centerPosY),(pieRadius,pieRadius),0,initDeg,endDeg,useColor, lineThickness)
                # if i in markList:
                #     mask_draw.line([(centerPosX,centerPosY),
                #         (int(centerPosX+pieRadius*math.cos(math.radians(initDeg))),
                #         int(centerPosY+pieRadius*math.sin(math.radians(initDeg))))],fill=0,width=3)
            if i in markList:
                mask_draw.pieslice([(10,10),(90,90)],initDeg,endDeg, fill =255)
                if self.param.mark.randPos:
                    fromRad,toRad,midPos=math.radians(initDeg),math.radians(endDeg),(centerPosX,centerPosY)
                    while abs(toRad-fromRad)>2*3.14159265358979:
                        if toRad<fromRad:
                            toRad+=2*3.14159265358979
                        else:
                            fromRad+=2*3.14159265358979
                    randomRad=(toRad-fromRad)*round(random.uniform(0.3,0.8),1)+fromRad
                    randomRadius=pieRadius * round(random.uniform(0.3,0.8),1)
                    # y = int(midPos[1]+math.sin(midRad)*midRadius)
                    # x = int(midPos[0]+math.cos(midRad)*midRadius)
                    y = int(midPos[1]+math.sin(randomRad)*randomRadius)
                    x = int(midPos[0]+math.cos(randomRad)*randomRadius)
                    if y<0 or x<0 or x+1>=image.shape[0] or y+1>=image.shape[0]:
                        logging.warning("Pie Generator: Wrong dot Position %d %d"%(x,y))
                    else:
                        image[y:y+1,x:x+1] = (self.param.mark.dotColor[0],self.param.mark.dotColor[1],self.param.mark.dotColor[2])
                    # self.mark(image,math.radians(initDeg),math.radians(endDeg),(centerPosX,centerPosY),pieRadius,self.param.mark.dotColor)
                    pass
                else:
                    # if self.param.mask.isFlag:
                    #     pass
                    # else:
                    self.mark(image,math.radians(initDeg),math.radians(endDeg),(centerPosX,centerPosY),pieRadius,self.param.mark.dotColor)
            initDeg=endDeg

        TitlePosition=self.param.TitlePosition
        TitleFontSize=self.param.TitleFontSize
        TitleFontType=self.param.TitleFontType
        TitleLength=self.param.TitleLength
        TitlePaddingLeft=self.param.TitlePaddingLeft
        image=uio.add_padding(image,backColor)
        image = Image.fromarray(np.uint8(image)).convert("RGB")
        image=uio.add_title(image,backColor,TitlePosition,TitleFontSize,TitleFontType,TitleLength,TitlePaddingLeft)
        if self.param.mask.isFlag:
            if self.param.mask.type=='contour':
                mask_image=np.array(mask_image)
                mask_image=np.uint8(mask_image)
                mask_image=np.pad(mask_image,((25,25),(25,25)),'constant',constant_values = (0,0))
                # image.paste(Image.fromarray(mask_image),(0,0))
                mask_image=np.expand_dims(mask_image,axis=2)
                image=np.array(image)
                image=np.concatenate((image, mask_image), axis=2)
                image=Image.fromarray(image)
        
        # image.paste(mask_image,(25,25))
        # image=np.array(image)
        # image=uio.add_pie_legend(image,backColor,colorLists)
            
        # save
        inputFilePath,outputFilePath,orgFilePath = self._getFilePath(isTrainData)

        fileName = self.param.fileName%index
        inputFilePath = os.path.join(inputFilePath,fileName)
        outputFilePath = os.path.join(outputFilePath,fileName)
        orgFilePath = os.path.join(orgFilePath,fileName)

        # self._preprocess_numpy(inputFilePath,image)
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
        elif self.param.mark.markAngle:
            uio.save(outputFilePath,[pieWeights[0]*360],"json")
            uio.save(outputFilePath+"_ll",[self.param.labelValue]*len(pieWeights),"json")
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

