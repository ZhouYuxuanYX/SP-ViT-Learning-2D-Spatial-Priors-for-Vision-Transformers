##!/bin/bash

## fine tuning lv-vit-s on 384x384 resolution
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#python3 -m torch.distributed.launch --nproc_per_node=8 main.py /ssd2/zhouyuxuan/Dataset/ImageNet --model lvvit_s -b 32 --native-amp \
#--img-size 384 --drop-path 0.1 --token-label --token-label-data /ssd2/zhouyuxuan/Dataset/label_top5_train_nfnet --token-label-size 24 --lr 5.e-6 \
#--min-lr 5.e-6 --weight-decay 1.e-8 --finetune /ssd2/zhouyuxuan/Repositories/lv-vit/output/train/20220323-154324-lvvit_s-224/model_best.pth.tar \
#--local_up_to_layer 14

## fine tuning lv-vit-m on 384x384 resolution
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#python3 -m torch.distributed.launch --nproc_per_node=8 main.py /ssd2/zhouyuxuan/Dataset/ImageNet --model lvvit_m -b 16 --native-amp \
#--img-size 384 --drop-path 0.2 --token-label --token-label-data /ssd2/zhouyuxuan/Dataset/label_top5_train_nfnet --token-label-size 24 --lr 5.e-6 \
#--min-lr 5.e-6 --weight-decay 1.e-8 --finetune /ssd2/zhouyuxuan/Repositories/lv-vit/output/train/20220323-153351-lvvit_m-224/model_best.pth.tar \

# fine tuning lv-vit-l on 384x384 resolution
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#python3 -m torch.distributed.launch --nproc_per_node=8 main.py /ssd2/zhouyuxuan/Dataset/ImageNet --grad_accu 4 --min-lr 5.e-6 \
#--model lvvit_l -b 16 --img-size 384 --local_up_to_layer 22 --native-amp --aa rand-n3-m9-mstd0.5-inc1 --weight-decay 1.e-8 \
#--drop-path 0.3 --token-label --token-label-data /ssd2/zhouyuxuan/Dataset/label_top5_train_nfnet --token-label-size 24 --model-ema --lr 5.e-6 \
#--finetune /ssd2/zhouyuxuan/Repositories/lv-vit/output/train/20220328-111607-lvvit_l-224/checkpoint-309.pth.tar



