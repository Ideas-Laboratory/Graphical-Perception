import logging
import os
from . import UtilIO as uio
from . import VisAbstractGen
import numpy as np
import cv2
import random
from util.Config import ConfigObj

class GroupBarGenerator(VisAbstractGen.VisAbstractGen):

    def __init__(self,config):
        super().__init__(config)
        ConfigObj.default(self.param,"fixStackGap",0)
        ConfigObj.default(self.param,"groupInnerPadding",0)
        ConfigObj.default(self.param,"mark.fix",[])
        ConfigObj.default(self.param,"values.maxGroupPixelHeight",0)

    def mark(self,image,center,dotColor):
        y=int(center[1])
        x=int(center[0])
        image[y:y+1,x:x+1]=(dotColor[0],dotColor[1],dotColor[2])

    def gen(self,index,isTrainData=True):
        width=uio.fetchValue(self.param.outputWidth,1)
        height=uio.fetchValue(self.param.outputWidth,1)
        stackCount = uio.fetchValue(self.param.stackCount) # each group has how many types?
        stackWidth = uio.fetchValue(self.param.stackWidth,1)
        stackGroup = uio.fetchValue(self.param.stackGroup) # how many groups?
        lineThickness = uio.fetchValue(self.param.lineThickness,-1)
        spacePaddingLeft = uio.fetchValue(self.param.spacePaddingLeft)
        spacePaddingRight = uio.fetchValue(self.param.spacePaddingRight)
        spacePaddingTop = uio.fetchValue(self.param.spacePaddingTop)
        spacePaddingBottom = uio.fetchValue(self.param.spacePaddingBottom)
        groupInnerPadding = uio.fetchValue(self.param.groupInnerPadding)

        
        # process color for stackCount
        colorLists,backColor,fill,strokeColor = self._genColor(stackCount)

        '''
        <       ><-------><        ><--------><          >
        padding  barWidth   empty    barWidth   padding
        '''

        horSpace = width-spacePaddingLeft-spacePaddingRight
        verSpace = height-spacePaddingTop-spacePaddingBottom
        if verSpace<=20:
            logging.warning("Wrong Parameters! Vertical Padding is too large! Set 20 instead.")
            verSpace = 20

        leftHorEmptySpace = horSpace - stackWidth*stackGroup
        if lineThickness>0:
            leftHorEmptySpace-= stackGroup*lineThickness*2
        # avoid overlapping
        if leftHorEmptySpace<0:
            leftHorEmptySpace=0
            stackWidth = int(horSpace/stackGroup)
            if lineThickness>0:
                stackWidth-=lineThickness*2
                if stackWidth<=2:
                    stackWidth+=lineThickness*2-2
                    lineThickness=1
            if stackWidth<=2:
                lineThickness=1
                stackWidth=2
                groupInnerPadding=1
                emptyPadding = width-stackWidth*stackGroup
                spacePaddingLeft = int((np.random.rand()*emptyPadding))
                spacePaddingRight = int(emptyPadding-spacePaddingLeft)
                leftHorEmptySpace = width-emptyPadding

        horEmptySpace = 0
        if stackGroup>1:
            horEmptySpace=leftHorEmptySpace // (stackGroup-1)

        if lineThickness>0:
            horEmptySpace+=lineThickness*2

        if self.param.fixStackGap>0:
            horEmptySpace = self.param.fixStackGap

        stackHeights=[]
        maxStackBarHeight = 0



        image = np.ones(shape=(width,height,3),dtype=np.int8)*255
        image[:,:]=backColor
        startOffsetX=int(spacePaddingLeft)
        startOffsetY =int(height-spacePaddingBottom)

        quant=verSpace
        if lineThickness>0:
            quant = verSpace//lineThickness
        if self.param.values.pixelValue:
            quant = 0
        values = self._genValues(stackGroup*stackCount,quant)

        markList=[]
        if self.param.mark.markAlwaysSameGroup:
            groupID = random.randint(0,stackGroup-1)
            startIndex = groupID*stackCount
            endIndex = startIndex+stackCount
            markList = self._mark(values[startIndex:endIndex])
            markList = [ i+startIndex for i in markList]
        elif self.param.mark.markAlwaysDiffGroup:
            indexList=[]
            for i in range(stackGroup):
                indPart = random.randint(0,stackCount-1)
                indexList.append(indPart+i*stackCount)
            filterValues = [values[i] for i in indexList]
            markList = self._mark(filterValues)
            markList = [indexList[i] for i in markList]
        else:
            markList = self._mark(values)


        barWidth = (stackWidth-groupInnerPadding*(stackCount-1))//stackCount
        if barWidth<=1:
            barWidth=max(2,stackWidth//stackCount)
            groupInnerPadding=0
        for i in range(stackGroup):
            tmpOffsetX=startOffsetX
            stackBarHeight=0
            for j in range(stackCount):
                curIndex=i*stackCount+j
                stackHeight=0
                if self.param.values.pixelValue:
                    stackHeight = max(1,int(values[curIndex]))
                else:
                    stackHeight = max(1,int(values[curIndex] * verSpace))
                stackHeights.append(stackHeight)
                useColor=colorLists[j]
                useColor2=strokeColor[0]
                if lineThickness>0:
                    useColor = colorLists[0]
                cv2.rectangle(image,
                    (tmpOffsetX, startOffsetY),
                    (tmpOffsetX + barWidth, startOffsetY - stackHeight),
                    useColor,
                    -1 if fill else lineThickness
                )
                if lineThickness>0:
                    cv2.rectangle(image,
                        (tmpOffsetX, startOffsetY),
                        (tmpOffsetX + barWidth, startOffsetY - stackHeight),
                        useColor2,
                        lineThickness
                    )
                if curIndex in markList:
                    self.mark(image,(startOffsetX+stackWidth*0.5,startOffsetY-stackHeight*0.5),self.param.mark.dotColor)
                tmpOffsetX+=barWidth+groupInnerPadding

            startOffsetX+=horEmptySpace

        # if preprocess is enabled, preprocess input data

        #save
        inputFilePath,outputFilePath,orgFilePath = self._getFilePath(isTrainData)

        fileName = self.param.fileName%index
        inputFilePath=os.path.join(inputFilePath,fileName)
        outputFilePath=os.path.join(outputFilePath,fileName)
        orgFilePath =os.path.join(orgFilePath,fileName)

        self._preprocess(inputFilePath,image)

        stackHeights = self._processValues(stackHeights,markList)
        if self.param.mark.ratio.ratio2Only:
            uio.save(outputFilePath,[stackHeights[0]/stackHeights[1]],"json")
            uio.save(outputFilePath+"_ll",[self.param.labelValue],"json")
        else:
            uio.save(outputFilePath,stackHeights,"json")
            uio.save(outputFilePath+"_ll",[self.param.labelValue]*len(stackHeights),"json")

        ratio = self._genRatio(stackHeights)
        uio.save(outputFilePath + "_r", ratio, "json")

        labels = [self.param.labelValue]*len(ratio)
        if self.param.mark.ratio.ratio2Only:
            uio.save(outputFilePath+"_l",[labels[0]],"json")
        else:
            uio.save(outputFilePath+"_l",labels,"json")