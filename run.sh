#!/bin/bash

for model in resnet20 #resnet32 resnet44 resnet56 resnet110 resnet1202
do
    echo "python -u trainer_hktcifar100.py  --arch=$model  --save-dir=save_$model |& tee -a log_$model"
    python -u trainer_hktcifar100.py  --arch=$model  --save-dir=save_$model |& tee -a log_$model
done
