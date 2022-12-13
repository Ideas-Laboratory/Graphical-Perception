import logging
import os
from . import UtilIO as uio
from . import VisAbstractGen
import numpy as np
import cv2
import random
from util.Config import ConfigObj

class FramedBarGenerator(VisAbstractGen.VisAbstractGen):

    def __init__(self, config):
        super().__init__(config)
        ConfigObj.default(self.param,"fixBarGap",0)
        ConfigObj.default(self.param,"values.useSpecialGen",False)
        ConfigObj.default(self.param,"outputAll",False)
        ConfigObj.default(self.param, "bgcolor", 'color_pool')
        ConfigObj.default(self.param, "barcolor", 'same')
        ConfigObj.default(self.param, "barcolordark", 'no')
        ConfigObj.default(self.param, "linecolor", 'color_pool')
        ConfigObj.default(self.param, "train", True)

    def mark(self,image,center,dotColor):
        y=int(center[1])
        x=int(center[0])
        image[y:y+1,x:x+1]=(dotColor[0],dotColor[1],dotColor[2])

    def gen(self,index,isTrainData=True):
        width = uio.fetchValue(self.param.outputWidth,1)
        height = uio.fetchValue(self.param.outputHeight,1)
        barCount = uio.fetchValue(self.param.barCount)
        barWidth = uio.fetchValue(self.param.barWidth,1)
        lineThickness = uio.fetchValue(self.param.lineThickness)
        spacePaddingLeft = uio.fetchValue(self.param.spacePaddingLeft)
        spacePaddingRight = uio.fetchValue(self.param.spacePaddingRight)
        spacePaddingTop = uio.fetchValue(self.param.spacePaddingTop)
        spacePaddingBottom = uio.fetchValue(self.param.spacePaddingBottom)
        barTotalHeight = uio.fetchValue(self.param.barTotalHeight)

        # colorLists,backColor,fill,storkeColor = self._genColor(barCount)
        if self.param.train:
            colorLists, backColor, fill, strokeColor = self._genTrainColor(barCount,index)
        else:
            colorLists, backColor, fill, strokeColor = self._genTestColor(barCount,index)
        # colorLists,backColor,fill,strokeColor = self._genTrainColor(barCount,index)
        colorLists=uio.RGB2BGR(colorLists)
        backColor=uio.RGB2BGR(tuple(backColor))
        strokeColor=uio.RGB2BGR(strokeColor)
        if fill:
            lineThickness = -1

        '''
        <       ><-------><        ><--------><          >
        padding  barWidth   empty    barWidth   padding
        '''

        horSpace = width - spacePaddingLeft - spacePaddingRight
        
        verSpace = height - spacePaddingTop - spacePaddingBottom
        if verSpace<=20:
            logging.error("Wrong Parameters! Vertical Padding is too large! Set 20 instead.")
            verSpace=20

        leftHorEmptySpace = horSpace - barWidth*barCount
        if lineThickness>0:
            leftHorEmptySpace-= barCount*lineThickness*4
        # avoid overlapping
        if leftHorEmptySpace<0:
            leftHorEmptySpace=0
            barWidth = int(horSpace/barCount)
            if lineThickness>0:
                barWidth-=lineThickness*2
                if barWidth<=2:
                    barWidth+=lineThickness*2-2
                    lineThickness=1
                    leftHorEmptySpace+=barCount*(lineThickness-1)*4
            if barWidth<=2:
                lineThickness=1
                barWidth=2
                emptyPadding = width-barWidth*barCount
                spacePaddingLeft = int((np.random.rand()*emptyPadding))
                spacePaddingRight = int(emptyPadding-spacePaddingLeft)
                leftHorEmptySpace = width-emptyPadding
        horEmptySpace = 0
        if barCount>1: 
            horEmptySpace = leftHorEmptySpace // (barCount-1)

        if lineThickness>0:
            horEmptySpace+=lineThickness*2

        if self.param.fixBarGap>0:
            horEmptySpace = self.param.fixBarGap

        barHeights = []
        maxBarHeight = 0


        image = np.ones(shape=(width,height,3),dtype=np.int8)*255
        image[:,:] = backColor
        startOffsetX = int(spacePaddingLeft)
        startOffsetY = int(height-spacePaddingBottom)

        quant=verSpace
        if lineThickness>0:
            quant = verSpace//lineThickness
        if self.param.values.pixelValue:
            quant = 0
        values = self._genValues(barCount,quant)

        yOffsets = [uio.fetchValue(self.param.yOffset) for i in range(len(values))]

        valueMax = max(values)
        for i in range(barCount):
            if self.param.values.pixelValue:
                barHeight = max(1,int(values[i]))
            else:
                barHeight = max(1,int(verSpace*values[i]/valueMax))
            barHeights.append(barHeight)
            
            maxBarHeight = max(maxBarHeight,barHeight)
            curY=startOffsetY-yOffsets[i]
            cv2.rectangle(image,
                (startOffsetX,curY),
                (startOffsetX+barWidth,curY-barHeight),
                colorLists[i],
                -1
                )
            if self.param.isFramed:
                cv2.rectangle(image,
                    (startOffsetX,curY),
                    (startOffsetX+barWidth,curY-barTotalHeight),
                    colorLists[i],
                    lineThickness
                    )
            startOffsetX += barWidth + horEmptySpace

        # if preprocess is enabled, preprocess input data
        
        # save
        inputFilePath,outputFilePath,orgFilePath = self._getFilePath(isTrainData)

        fileName = self.param.fileName%index
        inputFilePath = os.path.join(inputFilePath,fileName)
        outputFilePath = os.path.join(outputFilePath,fileName)
        orgFilePath = os.path.join(orgFilePath,fileName)

        self._preprocess_numpy(inputFilePath,image)
        # print(barHeights)
        barHeights = self._processValues(barHeights,[0,1])
        # print(barHeights)
        if self.param.mark.ratio.ratio2Only:
            uio.save(outputFilePath,[barHeights[0]/barHeights[1]],"json")
            uio.save(outputFilePath+"_ll",[self.param.labelValue],"json")
        else:
            if not self.param.outputAll:
                barHeights = [min(barHeights)]
            uio.save(outputFilePath,barHeights,"json")
            uio.save(outputFilePath+"_ll",[self.param.labelValue]*len(barHeights),"json")

        
        ratio = self._genRatio(barHeights)
        uio.save(outputFilePath+"_r",ratio,"json")

        labels = [self.param.labelValue]*len(ratio)
        uio.save(outputFilePath+"_l",labels,"json")
        


        

