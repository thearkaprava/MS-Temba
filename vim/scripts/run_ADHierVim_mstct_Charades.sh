#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python AD_hiermamba_mstct_main.py \
-dataset charades \
-mode rgb \
-model hierarchical_vim \
-train True \
-rgb_root /data/asinha13/projects/MAD/MS-TCT/data/charades_i3d_features/ \
-num_clips 256 \
-skip 0 \
-comp_info False \
-epochs 50 \
-unisize True \
-alpha_l 1 \
-beta_l 0.05 \
-batch_size 32 \
-output_dir workdirs/ms_tcfusion_mstct_chmambaint_e512_vim_d111_18Oct24/