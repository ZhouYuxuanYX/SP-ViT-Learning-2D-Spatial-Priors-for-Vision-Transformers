##!/bin/bash

# sp-vit-s
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#python3 -m torch.distributed.launch --nproc_per_node=8 main.py /ssd2/zhouyuxuan/Dataset/ImageNet \
#--model lvvit_s -b 128 --img-size 224 --local_up_to_layer 14 --native-amp \
#--drop-path 0.1 --token-label --token-label-data /ssd2/zhouyuxuan/Dataset/label_top5_train_nfnet --token-label-size 14 --model-ema --amp

## sp-vit-m
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#python3 -m torch.distributed.launch --nproc_per_node=8 main.py /ssd2/zhouyuxuan/Dataset/ImageNet \
#--model lvvit_m -b 128 --img-size 224 --native-amp \
#--drop-path 0.2 --token-label --token-label-data /ssd2/zhouyuxuan/Dataset/label_top5_train_nfnet --token-label-size 14 --model-ema

## sp-vit-l
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#python3 -m torch.distributed.launch --nproc_per_node=8 main.py /ssd2/zhouyuxuan/Dataset/ImageNet \
#--model lvvit_l -b 100 --img-size 224 --local_up_to_layer 22 --native-amp --lr 6.e-4 --aa rand-n3-m9-mstd0.5-inc1 \
#--drop-path 0.3 --token-label --token-label-data /ssd2/zhouyuxuan/Dataset/label_top5_train_nfnet --token-label-size 14 --model-ema
