'''
import json
import os
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--network',type=str, default='resnet152')
parser.add_argument('--datasetName',type=str, default='posLen_tp_1_nonfixm_c')
parser.add_argument('--trainMethod',type=str, default='')
args = parser.parse_args()
network=args.network
dataset_name = args.datasetName
train_method = args.trainMethod

pth = './result/%s_%s' % (network,dataset_name)
print(train_method)
if train_method != '':
    pth=pth+'_%s'
    pth=pth % (train_method)
pth=pth+'/model_%d.pkl'
if pth is None or len(pth)==0:
    print("basicPath is none")
folder, filebasicName = os.path.split(pth)
otherName = filebasicName.strip().replace(".pkl","").replace("%d","")
flist=[]
for root,dirs,files in os.walk(folder):
    for f in files:
        if f.endswith(".pkl"):
            temp = f.replace(otherName,"").replace(".pkl","").strip()
            iterValue = 0
            try:
                iterValue=int(temp)
            except BaseException as e:
                continue
            realPath=os.path.join(root,f)
            print(realPath)
            flist.append((iterValue,realPath))
print("Try to find model with minimum valid loss to test")
testResultOutputPath='./result/%s_%s/test' % (network,dataset_name)
validResultIterOutputFolder='Iter_%d'
testResultIterOutputFolder='Iter_%d_test'
minLoss=None
minIter=0
minIndex=0
i=0
# for iters,modelPath in flist:
#     folder = os.path.join(testResultOutputPath,testResultIterOutputFolder%iters)
#     print(folder)
#     fname = os.path.join(folder,"lossTotalInfo.json")
#     with open(fname,'r') as f:
#         data = json.load(f)
#     maxLoss=-1.0
#     for k,v in data.items():
#         maxLoss=max(v,maxLoss)
#     print("Model loss %f"%maxLoss)
#     if minLoss is None or minLoss > maxLoss:
#         minLoss=maxLoss
#         minIter=iters
#         minIndex=i
#     i+=1
loss=[]
for iters,modelPath in flist:
    folder = os.path.join(testResultOutputPath,validResultIterOutputFolder%iters)
    folder_test = os.path.join(testResultOutputPath,testResultIterOutputFolder%iters)
    print(folder)
    fname = os.path.join(folder,"lossTotalInfo.json")
    fname_test = os.path.join(folder_test,"lossTotalInfo.json")
    with open(fname,'r') as f:
        data = json.load(f)
    with open(fname_test,'r') as f:
        data_test = json.load(f)
    for k,v in zip(data.keys(),data_test.keys()):
        loss.append({'iter':iters,'valid_loss':data[k],'test_loss':data_test[v]})
    print("Model loss %f"%data[k])

if train_method != '':
    with open('%s_%s_%s_loss.json' % (network,dataset_name,train_method), 'w') as f_obj:
        json.dump(loss, f_obj)
else:
    with open('%s_%s_loss.json' % (network,dataset_name), 'w') as f_obj:
        json.dump(loss, f_obj)
# print("Decide to use Model (Iter %d), min loss %f"%(minIter,minLoss))
'''
import os
import json
import re
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--network',type=str, default='resnet152')
parser.add_argument('--datasetName',type=str, default='posLen_tp_1_nonfixm_c')
parser.add_argument('--trainMethod',type=str, default='')

args = parser.parse_args()
network=args.network
dataset_name = args.datasetName
train_method = args.trainMethod
# network='resnet152'
# dataset_name='posLen_tp_1_nonfixm_c'
# train_method='sgd'

pth = './result/%s_%s' % (network,dataset_name)
if train_method != '':
    pth=pth+'_%s'
    pth=pth % (train_method)
pth=pth+'/test'
print(pth)
if pth is None or len(pth)==0:
    print("basicPath is none")
loss=[]
for root,dirs,files in os.walk(pth):
    loss_dict={}
    # print(dirs)
    for dir in dirs:
        if not dir.endswith('test'):

            loss_dict={}

            path=os.path.join(root,dir)
            file_path=os.path.join(path+'_test','lossTotalInfo.json')
            iters=re.findall('\d+',path)[-1]
            loss_dict['iter']=iters
            with open(file_path,'r') as f:
                data = json.load(f)
            loss_dict['test_loss']=data['loss']

            file_path=os.path.join(path,'lossTotalInfo.json')
            iters=re.findall('\d+',path)[-1]
            loss_dict['iter']=iters
            with open(file_path,'r') as f:
                data = json.load(f)
            loss_dict['valid_loss']=data['loss']
            loss.append(loss_dict)
if train_method != '':
    with open('%s_%s_%s_loss.json' % (network,dataset_name,train_method), 'w') as f_obj:
        json.dump(loss, f_obj)
else:
    with open('%s_%s_loss.json' % (network,dataset_name), 'w') as f_obj:
        json.dump(loss, f_obj)