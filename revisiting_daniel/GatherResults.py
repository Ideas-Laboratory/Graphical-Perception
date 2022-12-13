import os
import json
import argparse

print("Usage: --root_path file_name\n\tCollect results from this folder")
print("--output_path file_name\n\tOutput csv file")

#result/raw_result
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", default='.')
parser.add_argument("--output_path", default='sumup.csv')
parser_args, _ = parser.parse_known_args()
rootPath = parser_args.root_path
outputPath = parser_args.output_path

jsonFiles = []
for root, dirs, files in os.walk(rootPath):
    if os.path.split(root)[1]!="final" or "old" in root:
        continue
    for fp in files:
        jsonPath = os.path.join(root,fp)
        print("Detect %s"%jsonPath)
        jsonFiles.append(jsonPath)

print("Find %d files"%len(jsonFiles))

results=[]


for jf in jsonFiles:
    try:
        path, fileName = os.path.split(jf) # xxx/final/xxx.json
        path, _ = os.path.split(path) # xxx/final
        _, folder = os.path.split(path) # xxx

        obj=None
    
        f = open(jf,"r")
        obj = json.load(f)
        f.close()
        minMLAE=10000000
        minIndex=0
        #print(obj)
        #print("-"*16)
        print(jf)
        for i,o in enumerate(obj):
            oloss = o["loss"]
            omlae = oloss["MLAE"]
            v = omlae["avg"]
            if v<minMLAE:
                minMLAE=v
                minIndex=i

        opath = o["path"]
        iter_info = os.path.split(os.path.split(opath)[0])[1]
        oloss = o["loss"]
        omlae = oloss["MLAE"]
        omlaeun = oloss["MLAE_unmatch"]
        oabs = oloss["ABS"]
        oabsun = oloss["ABS_unmatch"]
        test_name = fileName.replace(".json","").replace("mlae_g2_","").replace("_10000","")
        test_type = test_name.strip().split("_")[0]
        is_out=False
        if "_out_" in test_name:
            is_out=True
            test_name = test_name.replace("_out_","_")
        test_obj = test_name.replace(test_type+"_","")
        network_name = folder.split("_")[0]
        network_train = folder.replace(network_name,"")[1:]
        r = (network_name,network_train,iter_info,test_type,is_out,test_obj,omlae["avg"],oabs["avg"],omlaeun["avg"],oabsun["avg"],omlae["std"],oabs["std"])
        results.append(r)
    except BaseException as e:
        print("Error while load %s"%jf)
        print(e)
        continue


results = sorted(results)

csvf = open(outputPath,"w")
csvf.write("net_name,train_data,iter_info,test_attr,out_distribute,test_type,MLAE_avg,ABS_avg,MLAE_unmatch,ABS_unmatch,MLAE_std,ABS_std\n")
for r in results:
    s = str(r)
    s = s.replace("\'","").replace("(","").replace(")","").replace(" ","")
    csvf.write(s+"\n")
csvf.close()

print("Complete, %d results"%len(results))





    
