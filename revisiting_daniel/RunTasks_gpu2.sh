for ((i=0; i<=0; i++))
do
python3 MainTrain.py --config_file TASK_2/train_hlr.json --datasetName posLen_tp_5 --modelName resnet152 --gpu 2
done
for ((i=0; i<=0; i++))
do
python3 MainTrain.py --config_file TASK_2/train_hlr.json --datasetName posAngle_bar --modelName resnet152 --gpu 2
done