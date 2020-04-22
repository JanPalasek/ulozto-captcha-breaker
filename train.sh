#!/bin/bash
# this script can be run on cluster using command e.g.: qsub -l mem_free=8G,act_mem_free=8G,h_vmem=12G -cwd -j y sh train.sh
/usr/bin/python3 ./src/captcha_detection/train.py \
--checkpoint_freq=1 \
--transformed_img_height=60 \
--transformed_img_width=60