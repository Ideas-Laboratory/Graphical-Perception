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
    def __init__(self,cmd,res=0,waitID=[],rerun=False,weight=0):
        self.id=md5(cmd.strip())
        self.waitID=waitID
        self.res=res
        self.cmd=cmd
        self.rerun=rerun
        self.weight=weight

tasks=[]
dataTasks = {}
for dataName in dataFileList:
    #if "posLen" not in dataName and "pointCloud" not in dataName:
    #    continue
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

trainCommandTemplateMidDecay = pythonName+" MainTrain.py --config_file TASK_2/train_mid_weight_decay.json --datasetName %s --modelName %s"
testCommandTemplateMidDecay = pythonName+" RunTest.py --config_file TASK_2/test_mid_weight_decay.json --trainDatasetName %s --modelName %s --datasetName "

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
    addTrainTask(tasks, "posLen_tp_mix_fixm", network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,330+baseWeight)
    addTrainTask(tasks, "posLen_tp_mix_nonfixm", network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,330+baseWeight)
    #addTrainTask(tasks, "posLen_tp_mix_rand_5n", network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,340+baseWeight)
    addTrainTask(tasks, "posLen_tp_mix_rand_20n", network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,340+baseWeight)
    ###addTrainTask(tasks, "posLen_tp_mix_rand_nodot_cl", network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,200)
    ###addTrainTask(tasks, "posLen_tp_mix_rand_cl", network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,250)
    addTrainTask(tasks, "posLen_tp_mix_rand_cl", network,dataCommandTemplate, trainCommandTemplateMidDecay, testCommandTemplateMidDecay,250)
    ###addTrainTask(tasks, "posLen_tp_mix_rand_nodot", network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,200)
    #addTrainTask(tasks, "posLen_tp_mix_rand", network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,203)
    #addTrainTask(tasks, "posLen_tp_mix", network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,200)

for network in ["vgg19"]:
    #addTrainTask(tasks, "pointCloud_10_lowc", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,101)
    #addTrainTask(tasks, "pointCloud_100_lowc", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,102)
    #addTrainTask(tasks, "pointCloud_1000_lowc", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,103)

    addTrainTask(tasks, "pointCloud_10_lowc",   network,dataCommandTemplate, trainCommandTemplateLowDecayHighLearn, testCommandTemplateLowDecayHighLearn,101)
    addTrainTask(tasks, "pointCloud_100_lowc",  network,dataCommandTemplate, trainCommandTemplateLowDecayHighLearn, testCommandTemplateLowDecayHighLearn,102)
    addTrainTask(tasks, "pointCloud_1000_lowc", network,dataCommandTemplate, trainCommandTemplateLowDecayHighLearn, testCommandTemplateLowDecayHighLearn,103)

    addTrainTask(tasks, "pointCloud_10",   network,dataCommandTemplate, trainCommandTemplateLowDecayHighLearn, testCommandTemplateLowDecayHighLearn,101)
    addTrainTask(tasks, "pointCloud_100",  network,dataCommandTemplate, trainCommandTemplateLowDecayHighLearn, testCommandTemplateLowDecayHighLearn,102)
    addTrainTask(tasks, "pointCloud_1000", network,dataCommandTemplate, trainCommandTemplateLowDecayHighLearn, testCommandTemplateLowDecayHighLearn,103)


prefix = "python3 "
tasks.append(TaskInfo("mkdir result",{"disk":1,"gpu":0},[],False,1000))
tasks.append(TaskInfo("mkdir result/org_paper",{"disk":1,"gpu":0},[tasks[-1].id],False,999))
mkid = tasks[-1].id
tasks.append(TaskInfo("python3 run_position_length.py C.Figure4.data_to_type1 VGG19 True 0 --gpu %d > result/org_paper/train_poslen_1.log",{"disk":5,"gpu":40},[mkid],False,500))
tasks.append(TaskInfo("python3 run_position_length.py C.Figure4.data_to_type2 VGG19 True 1 --gpu %d > result/org_paper/train_poslen_2.log",{"disk":5,"gpu":40},[mkid],False,500))
tasks.append(TaskInfo("python3 run_position_length.py C.Figure4.data_to_type3 VGG19 True 2 --gpu %d > result/org_paper/train_poslen_3.log",{"disk":5,"gpu":40},[mkid],False,500))
tasks.append(TaskInfo("python3 run_position_length.py C.Figure4.data_to_type4 VGG19 True 3 --gpu %d > result/org_paper/train_poslen_4.log",{"disk":5,"gpu":40},[mkid],False,500))
tasks.append(TaskInfo("python3 run_position_length.py C.Figure4.data_to_type5 VGG19 True 4 --gpu %d > result/org_paper/train_poslen_5.log",{"disk":5,"gpu":40},[mkid],False,500))
tasks.append(TaskInfo("python3 run_position_length.py C.Figure4.multi VGG19 True 5 --gpu %d > result/org_paper/train_poslen_multi.log",{"disk":5,"gpu":40},[mkid],False,505))
tasks.append(TaskInfo("python3 run_position_angle.py C.Figure3.data_to_barchart VGG19 True 6 --gpu %d > result/org_paper/train_posangle_bar.log",{"disk":5,"gpu":40},[mkid],False,400))
tasks.append(TaskInfo("python3 run_position_angle.py C.Figure3.data_to_piechart VGG19 True 7 --gpu %d > result/org_paper/train_posangle_pie.log",{"disk":5,"gpu":40},[mkid],False,400))
tasks.append(TaskInfo("python3 run_bar_framed_rectangle.py C.Figure12.data_to_framed_rectangles VGG19 True 8 --gpu %d > result/org_paper/train_frame_bar.log",{"disk":5,"gpu":40},[mkid],False,300))
tasks.append(TaskInfo("python3 run_bar_framed_rectangle.py C.Figure12.data_to_bars VGG19 True 9 --gpu %d > result/org_paper/train_frame_non_bar.log",{"disk":5,"gpu":40},[mkid],False,300))
tasks.append(TaskInfo("python3 run_weber.py C.Weber.base10 VGG19 True 10 --gpu %d > result/org_paper/train_point_10.log",{"disk":5,"gpu":40},[mkid],False,200))
tasks.append(TaskInfo("python3 run_weber.py C.Weber.base100 VGG19 True 11 --gpu %d > result/org_paper/train_point_100.log",{"disk":5,"gpu":40},[mkid],False,200))
tasks.append(TaskInfo("python3 run_weber.py C.Weber.base1000 VGG19 True 12 --gpu %d > result/org_paper/train_point_1000.log",{"disk":5,"gpu":40},[mkid],False,200))
"""
for network in ["vgg19","resnet152"]:
    weight=200
    if network=="vgg19":
        weight=300
    #addTrainTask(tasks, "pointCloud_10"     , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,105+weight)
    #addTrainTask(tasks, "pointCloud_100"    , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,105+weight)
    #addTrainTask(tasks, "pointCloud_1000"   , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,105+weight)
    addTrainTask(tasks, "posAngle_bar"      , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,101+weight)
    addTrainTask(tasks, "posAngle_pie"      , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,101+weight)
    addTrainTask(tasks, "posAngle_mix"      , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,101+weight)
    addTrainTask(tasks, "posLen_tp_1"       , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,102+weight)
    addTrainTask(tasks, "posLen_tp_2"       , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,102+weight)
    addTrainTask(tasks, "posLen_tp_3"       , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,102+weight)
    addTrainTask(tasks, "posLen_tp_4"       , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,102+weight)
    addTrainTask(tasks, "posLen_tp_5"       , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,102+weight)
    addTrainTask(tasks, "posLen_tp_mix"     , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,102+weight)
    addTrainTask(tasks, "visCue_framed"     , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,99+weight)
    addTrainTask(tasks, "visCue_nonframed"  , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,99+weight)
    addTrainTask(tasks, "ele_angle"         , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,98+weight)
    addTrainTask(tasks, "ele_area"          , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,98+weight)
    addTrainTask(tasks, "ele_curvature"     , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,98+weight)
    addTrainTask(tasks, "ele_direction"     , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,98+weight)
    addTrainTask(tasks, "ele_length"        , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,98+weight)
    addTrainTask(tasks, "ele_posCommon"     , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,98+weight)
    addTrainTask(tasks, "ele_posNonAlign"   , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,98+weight)
    addTrainTask(tasks, "ele_shading"       , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,98+weight)
    addTrainTask(tasks, "ele_volume"        , network,dataCommandTemplate, trainCommandTemplateLowDecay, testCommandTemplateLowDecay,98+weight)

for network in ["vgg19","resnet152"]:
    weight=0
    if network=="vgg19":
        weight=100
    #addTrainTask(tasks, "pointCloud_10", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,105+weight)
    #addTrainTask(tasks, "pointCloud_100", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,105+weight)
    #addTrainTask(tasks, "pointCloud_1000", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,105+weight)
    addTrainTask(tasks, "posAngle_bar", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,101+weight)
    addTrainTask(tasks, "posAngle_pie", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,101+weight)
    addTrainTask(tasks, "posAngle_mix", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,101+weight)
    addTrainTask(tasks, "posLen_tp_1", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,102+weight)
    addTrainTask(tasks, "posLen_tp_2", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,102+weight)
    addTrainTask(tasks, "posLen_tp_3", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,102+weight)
    addTrainTask(tasks, "posLen_tp_4", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,102+weight)
    addTrainTask(tasks, "posLen_tp_5", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,102+weight)
    addTrainTask(tasks, "posLen_tp_mix", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,102+weight)
    addTrainTask(tasks, "visCue_framed", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,99+weight)
    addTrainTask(tasks, "visCue_nonframed", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,99+weight)
    addTrainTask(tasks, "ele_angle", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,98+weight)
    addTrainTask(tasks, "ele_area", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,98+weight)
    addTrainTask(tasks, "ele_curvature", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,98+weight)
    addTrainTask(tasks, "ele_direction", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,98+weight)
    addTrainTask(tasks, "ele_length", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,98+weight)
    addTrainTask(tasks, "ele_posCommon", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,98+weight)
    addTrainTask(tasks, "ele_posNonAlign", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,98+weight)
    addTrainTask(tasks, "ele_shading", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,98+weight)
    addTrainTask(tasks, "ele_volume", network,dataCommandTemplate, trainCommandTemplateSGD, testCommandTemplateSGD,98+weight)
    pass
# normal task
rerunTest=True
rerunTrain=False
#for dataName in dataFileList:
#    for modelName in modelFileList:
#        addTrainTask(tasks, dataName,modelName,dataCommandTemplate, trainCommandTemplate, testCommandTemplate)

tasks = [x.__dict__ for x in tasks]
fp = open("tasks.json","w")
json.dump(tasks,fp,indent=4)
fp.close()
print("Task Count: %d"%len(tasks))
