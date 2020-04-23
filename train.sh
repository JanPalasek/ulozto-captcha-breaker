#!/bin/bash
venv/bin/python3 src/train.py --weights_file="src/captcha_detection/model.h5" \
--checkpoint_freq=1