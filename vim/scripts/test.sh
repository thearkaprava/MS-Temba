#!/usr/bin/env bash

export CUDA_HOME=/data/wbondura/cuda-12.1 
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export PATH=/pytorch_env/bin:$PATH

python MSTemba_main.py \
-dataset tsu \
-mode rgb \
-backbone clip  \
-model mstemba \
-train True \
-rgb_root /data/asinha13/projects/MAD/MS-TCT/data/smarthome_features_clip/ \
-num_clips 2500 \
-skip 0 \
--lr 4.5e-4 \
-comp_info False \
-epochs 140 \
-unisize True \
-alpha_l 1 \
-beta_l 0.05 \
-batch_size 1 \
-output_dir /data/asinha13/projects/MSTEMBA_TSU_CLIP_TEST_17Dec25