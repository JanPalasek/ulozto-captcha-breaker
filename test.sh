#!/bin/bash
# this script can be run on cluster using command e.g.: qsub -l mem_free=8G,act_mem_free=8G,h_vmem=12G -cwd -j y sh train.sh
/usr/bin/python3 ./src/captcha_detection/test.py