#!/bin/bash

GPU_ID=6

work_dir=`basename $PWD`
experiment_name=slt_base_ft
#train_data='/home7/wjc505/ctc-vc/data/finetune_data/data_fintune_bdl100.txt'
train_data='/student/home/wl/data/vctk_vad_wav/finetune_data/data_fintune_slt100_wl.txt'

# train model
CUDA_VISIBLE_DEVICES=${GPU_ID} python finetune.py \
	--name=${experiment_name} \
	--input=${train_data} \
	--checkpoint_path='output/vqvae_vq40_q64_base/model/model_300000.pt' \
    --gpus=$GPU_ID
