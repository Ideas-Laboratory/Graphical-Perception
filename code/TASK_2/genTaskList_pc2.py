import os
import platform
import json

modelFolder = "model"
dataFolder = "data/set"

modelFileList = []
for root,dirs,files in os.walk(modelFolder):
    for f in files:
        if ".json" in f:
            modelFileList.append(f.replace(".json",""))
    break

print("Model Count: %d"%len(modelFileList))
dataFileList = []
for root,dirs,files in os.walk(dataFolder):
    for f in files:
        if ".json" in f:
            dataFileList.append(f.replace(".json",""))
    break
print("Data Count: %d"%len(dataFileList))

pythonName = "python3"
#if platform.system()=='Windows':
    #pythonName = "python"

taskFolder = "TASK_2/tasks"

dataCommandTemplate = pythonName + " GenDataset.py --config_file TASK_2/genData.json --datasetName %s"
trainCommandTemplate = pythonName+" MainTrain.py --config_file TASK_2/train.json --datasetName %s --modelName %s"
testCommandTemplate = pythonName+" RunTest.py --config_file TASK_2/test.json --trainDatasetName %s --modelName %s --datasetName "

cmdList=[]


import hashlib
def md5(s):
    md = hashlib.md5()
    md.update(s.encode("utf-8"))
    return md.hexdigest()

class TaskInfo:
    def __init__(self,cmd,res=0,waitID=[],rerun=True,weight=0):
        self.id=md5(cmd.strip())
        self.waitID=waitID
        self.res=res
        self.cmd=cmd
        self.rerun=rerun
        self.weight=weight

tasks=[]
dataTasks = {}
for dataName in dataFileList:
    if dataName not in ["posLen_tp_1_v1","posAngle_bar_v1","posLen_tp_mix_rand","posLen_tp_mix","pointCloud_10","pointCloud_100","pointCloud_1000","posAngle_bar_v2","posLen_tp_1_v2","posAngle_bar_nodot","posAngle_bar_v1_nodot","posAngle_bar_v2_nodot","posAngle_bar"]:
        continue
    cmd = dataCommandTemplate % dataName
    dataTask=None
    dataTask = TaskInfo(cmd,{"disk":40,"gpu":0},[],True,999)
    tasks.append(dataTask)
    dataTasks[dataName]=dataTask.id

def addTrainTask(tasks,dataName,modelName,dataCommandTemplate, trainCommandTemplate, testCommandTemplate,weight=0):
        rerunTest=True
        rerunTrain=False
        trainCommand = trainCommandTemplate%(dataName,modelName)
        trainTask = TaskInfo(trainCommand,{"disk":25,"gpu":40},[dataTasks[dataName]],rerunTrain,weight)
        trainID = trainTask.id
        tasks.append(trainTask)

        testCommand = testCommandTemplate%(dataName,modelName)
        testCommand+="%s"
        tasks.append(TaskInfo(testCommand%dataName,{"disk":20,"gpu":20},[trainID,dataTasks[dataName]],rerunTest,weight))
        #special
        if dataName == "posAngle_mix":
            tasks.append(TaskInfo(testCommand%"posAngle_bar",{"disk":20,"gpu":20},[trainID,dataTasks[dataName]],rerunTest,weight))
            tasks.append(TaskInfo(testCommand%"posAngle_pie",{"disk":20,"gpu":20},[trainID,dataTasks[dataName]],rerunTest,weight))
        elif dataName == "posLen_tp_mix":
            tasks.append(TaskInfo(testCommand%"posLen_tp_1",{"disk":20,"gpu":20},[trainID,dataTasks[dataName]],rerunTest,weight))
            tasks.append(TaskInfo(testCommand%"posLen_tp_2",{"disk":20,"gpu":20},[trainID,dataTasks[dataName]],rerunTest,weight))
            tasks.append(TaskInfo(testCommand%"posLen_tp_3",{"disk":20,"gpu":20},[trainID,dataTasks[dataName]],rerunTest,weight))
            tasks.append(TaskInfo(testCommand%"posLen_tp_4",{"disk":20,"gpu":20},[trainID,dataTasks[dataName]],rerunTest,weight))
            tasks.append(TaskInfo(testCommand%"posLen_tp_5",{"disk":20,"gpu":20},[trainID,dataTasks[dataName]],rerunTest,weight))
        elif dataName == "posLen_tp_mix_rand":
            tasks.append(TaskInfo(testCommand%"posLen_tp_1_rand",{"disk":20,"gpu":20},[trainID,dataTasks[dataName]],rerunTest,weight))
            tasks.append(TaskInfo(testCommand%"posLen_tp_2_rand",{"disk":20,"gpu":20},[trainID,dataTasks[dataName]],rerunTest,weight))
        elif dataName == "visCue_mix":
            tasks.append(TaskInfo(testCommand%"visCue_nonframed",{"disk":20,"gpu":20},[trainID,dataTasks[dataName]],rerunTest,weight))
            tasks.append(TaskInfo(testCommand%"visCue_framed",{"disk":20,"gpu":20},[trainID,dataTasks[dataName]],rerunTest,weight))
        elif dataName == "pointCloud_mix":
            tasks.append(TaskInfo(testCommand%"pointCloud_10",{"disk":20,"gpu":20},[trainID,dataTasks[dataName]],rerunTest,weight))
            tasks.append(TaskInfo(testCommand%"pointCloud_100",{"disk":20,"gpu":20},[trainID,dataTasks[dataName]],rerunTest,weight))
            tasks.append(TaskInfo(testCommand%"pointCloud_1000",{"disk":20,"gpu":20},[trainID,dataTasks[dataName]],rerunTest,weight))


# single task:
trainCommandTemplateLowDecay = pythonName+" MainTrain.py --config_file TASK_2/train_low_weight_decay.json --datasetName %s --modelName %s"
testCommandTemplateLowDecay = pythonName+" RunTest.py --config_file TASK_2/test_low_weight_decay.json --trainDatasetName %s --modelName %s --datasetName "

trainCommandTemplateLowDecayHighLearn = pythonName+" MainTrain.py --config_file TASK_2/train_low_weight_decay.json --datasetName %s --modelName %s"
testCommandTemplateLowDecayHighLearn = pythonName+" RunTest.py --config_file TASK_2/test_low_weight_decay.json --trainDatasetName %s --modelName %s --datasetName "

trainCommandTemplateSGD = pythonName+" MainTrain.py --config_file TASK_2/train_SGD.json --datasetName %s --modelName %s"
testCommandTemplateSGD = pythonName+" RunTest.py --config_file TASK_2/test_SGD.json --trainDatasetName %s --modelName %s --datasetName "
#addTrainTask(tasks, "posAngle_bar", "vgg19",dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay)
#addTrainTask(tasks, "posAngle_pie", "vgg19",dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay)
"""
addTrainTask(tasks, "posLen_tp_1_nodot", "vgg19",dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay)
addTrainTask(tasks, "posLen_tp_2_nodot", "vgg19",dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay)
addTrainTask(tasks, "posLen_tp_3_nodot", "vgg19",dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay)
addTrainTask(tasks, "posLen_tp_4_nodot", "vgg19",dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay)
addTrainTask(tasks, "posLen_tp_5_nodot", "vgg19",dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay)
addTrainTask(tasks, "posLen_tp_mix_nodot", "vgg19",dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay)


for network in ["vgg19","resnet152"]:
    addTrainTask(tasks, "posLen_tp_mix_rand", network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay)
    addTrainTask(tasks, "posLen_tp_1_rand", network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay)
    addTrainTask(tasks, "posLen_tp_2_rand", network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay)


for network in ["vgg19","resnet152"]:
    baseWeight=0
    if network=="vgg19":
        baseWeight+=100
    #addTrainTask(tasks, "posLen_tp_mix_fixm", network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,350+baseWeight)
    #addTrainTask(tasks, "posLen_tp_mix_nonfixm", network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,350+baseWeight)
    ##addTrainTask(tasks, "posLen_tp_mix_rand_5n", network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,340+baseWeight)
    #addTrainTask(tasks, "posLen_tp_mix_rand_20n", network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,340+baseWeight)

    ###addTrainTask(tasks, "posLen_tp_mix_rand_nodot_cl", network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,250+baseWeight)
    #addTrainTask(tasks, "posLen_tp_mix_rand_cl", network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,205+baseWeight)
    ###addTrainTask(tasks, "posLen_tp_mix_rand_nodot", network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,200+baseWeight)
    addTrainTask(tasks, "posLen_tp_mix_rand", network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,203+baseWeight)
    addTrainTask(tasks, "posLen_tp_mix", network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,200+baseWeight)

for network in ["vgg19"]:
    addTrainTask(tasks, "pointCloud_10_lowc", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,101)
    addTrainTask(tasks, "pointCloud_100_lowc", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,102)
    addTrainTask(tasks, "pointCloud_1000_lowc", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,103)

    #addTrainTask(tasks, "pointCloud_10_lowc",   network,dataCommandTemplate, trainCommandTemplateLowDecayHighLearn, testCommandTemplateLowDecayHighLearn,101)
    #addTrainTask(tasks, "pointCloud_100_lowc",  network,dataCommandTemplate, trainCommandTemplateLowDecayHighLearn, testCommandTemplateLowDecayHighLearn,102)
    #addTrainTask(tasks, "pointCloud_1000_lowc", network,dataCommandTemplate, trainCommandTemplateLowDecayHighLearn, testCommandTemplateLowDecayHighLearn,103)

    #addTrainTask(tasks, "pointCloud_10",   network,dataCommandTemplate, trainCommandTemplateLowDecayHighLearn, testCommandTemplateLowDecayHighLearn,101)
    #addTrainTask(tasks, "pointCloud_100",  network,dataCommandTemplate, trainCommandTemplateLowDecayHighLearn, testCommandTemplateLowDecayHighLearn,102)
    #addTrainTask(tasks, "pointCloud_1000", network,dataCommandTemplate, trainCommandTemplateLowDecayHighLearn, testCommandTemplateLowDecayHighLearn,103)



for network in ["vgg19"]:
    baseWeight=0
    if network=="vgg19":
        baseWeight+=100
    addTrainTask(tasks, "posLen_tp_1_v1", network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,500+baseWeight)
    addTrainTask(tasks, "posAngle_bar_v1", network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,500+baseWeight)

"""

for network in ["vgg19","resnet152"]:
    weight=0
    if network=="vgg19":
        weight=100
    addTrainTask(tasks, "posAngle_bar"     , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,1105+weight)
    addTrainTask(tasks, "posAngle_bar_nodot"     , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,1105+weight)
    addTrainTask(tasks, "posAngle_bar_v1_nodot"     , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,1105+weight)
    addTrainTask(tasks, "posAngle_bar_v2_nodot"     , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,1105+weight)

    addTrainTask(tasks, "posAngle_bar_v2"     , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,1005+weight)
    addTrainTask(tasks, "posLen_tp_1_v2"     , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,1005+weight)
    addTrainTask(tasks, "posLen_tp_1_v1"     , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,1000+weight)

for network in ["vgg19","resnet152"]:
    weight=200
    if network=="vgg19":
        weight=300
    addTrainTask(tasks, "pointCloud_10"     , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,105+weight)
    addTrainTask(tasks, "pointCloud_100"    , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,105+weight)
    addTrainTask(tasks, "pointCloud_1000"   , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,105+weight)
    #addTrainTask(tasks, "posAngle_bar"      , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,101+weight)
    #addTrainTask(tasks, "posAngle_pie"      , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,101+weight)
    #addTrainTask(tasks, "posAngle_mix"      , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,101+weight)
    #addTrainTask(tasks, "posLen_tp_1"       , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,102+weight)
    #addTrainTask(tasks, "posLen_tp_2"       , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,102+weight)
    #addTrainTask(tasks, "posLen_tp_3"       , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,102+weight)
    #addTrainTask(tasks, "posLen_tp_4"       , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,102+weight)
    #addTrainTask(tasks, "posLen_tp_5"       , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,102+weight)
    #addTrainTask(tasks, "posLen_tp_mix"     , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,102+weight)
    #addTrainTask(tasks, "visCue_framed"     , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,99+weight)
    #addTrainTask(tasks, "visCue_nonframed"  , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,99+weight)
    #addTrainTask(tasks, "ele_angle"         , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,98+weight)
    #addTrainTask(tasks, "ele_area"          , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,98+weight)
    #addTrainTask(tasks, "ele_curvature"     , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,98+weight)
    #addTrainTask(tasks, "ele_direction"     , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,98+weight)
    #addTrainTask(tasks, "ele_length"        , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,98+weight)
    #addTrainTask(tasks, "ele_posCommon"     , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,98+weight)
    #addTrainTask(tasks, "ele_posNonAlign"   , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,98+weight)
    #addTrainTask(tasks, "ele_shading"       , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,98+weight)
    #addTrainTask(tasks, "ele_volume"        , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,98+weight)

for network in ["vgg19","resnet152"]:
    weight=0
    if network=="vgg19":
        weight=100
    addTrainTask(tasks, "pointCloud_10", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,105+weight)
    addTrainTask(tasks, "pointCloud_100", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,105+weight)
    addTrainTask(tasks, "pointCloud_1000", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,105+weight)
    #addTrainTask(tasks, "posAngle_bar", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,101+weight)
    #addTrainTask(tasks, "posAngle_pie", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,101+weight)
    #addTrainTask(tasks, "posAngle_mix", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,101+weight)
    #addTrainTask(tasks, "posLen_tp_1", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,102+weight)
    #addTrainTask(tasks, "posLen_tp_2", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,102+weight)
    #addTrainTask(tasks, "posLen_tp_3", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,102+weight)
    #addTrainTask(tasks, "posLen_tp_4", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,102+weight)
    #addTrainTask(tasks, "posLen_tp_5", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,102+weight)
    #addTrainTask(tasks, "posLen_tp_mix", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,102+weight)
    #addTrainTask(tasks, "visCue_framed", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,99+weight)
    #addTrainTask(tasks, "visCue_nonframed", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,99+weight)
    #addTrainTask(tasks, "ele_angle", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,98+weight)
    #addTrainTask(tasks, "ele_area", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,98+weight)
    #addTrainTask(tasks, "ele_curvature", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,98+weight)
    #addTrainTask(tasks, "ele_direction", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,98+weight)
    #addTrainTask(tasks, "ele_length", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,98+weight)
    #addTrainTask(tasks, "ele_posCommon", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,98+weight)
    #addTrainTask(tasks, "ele_posNonAlign", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,98+weight)
    #addTrainTask(tasks, "ele_shading", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,98+weight)
    #addTrainTask(tasks, "ele_volume", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,98+weight)
    pass
# normal task
rerunTest=True
rerunTrain=False
#for dataName in dataFileList:
#    for modelName in modelFileList:
#        addTrainTask(tasks, dataName,modelName,dataCommandTemplate, trainCommandTemplate, testCommandTemplate)

tasks = [x.__dict__ for x in tasks]
fp = open("tasks_pc2.json","w")
json.dump(tasks,fp,indent=4)
fp.close()
print("Task Count: %d"%len(tasks))
