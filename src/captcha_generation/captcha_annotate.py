#!/usr/bin/env python

import sys
sys.path.insert(0, "src")

import argparse
import os
import random

from dataset.annotations_generator import AnnotationsGenerator

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--test_split", default=0.1, type=float)
    parser.add_argument("--val_split", default=0.1, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--case_sensitive", action="store_true", default=False, help="Boolean switch that is true when "
                                                                                     "captcha label should be case sensitive.")

    args = parser.parse_args()

    random.seed(args.seed)

    out_dir = os.path.abspath(args.out_dir)
    data_dir = os.path.join(out_dir, "data")

    generator = AnnotationsGenerator(data_dir, out_dir, args.val_split, args.test_split, not args.case_sensitive)
    generator.save_annotations()
