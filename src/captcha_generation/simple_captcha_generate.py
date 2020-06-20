#!/usr/bin/env python

import sys
sys.path.insert(0, "src")

import argparse
import itertools
import os
import random

from captcha.image import ImageCaptcha

from faker import Faker


def generate_randomly(available_chars: str, dataset_size: int, captcha_length: int):
    for i in range(0, dataset_size):
        captcha_code = ""
        for _ in range(captcha_length):
            random_idx = random.randint(0, len(available_chars) - 1)
            captcha_code += available_chars[random_idx]

        yield captcha_code


def generate_systematically(available_chars: str, dataset_size: int, captcha_length: int):
    y = [available_chars for _ in range(captcha_length)]

    available_combinations = itertools.product(*y)

    for x in itertools.islice(available_combinations, dataset_size):
        yield "".join(x)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_size", default=1000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--captcha_length", default=4, type=int)
    parser.add_argument("--available_chars", default="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", type=str)
    parser.add_argument("--generation_type", type=str, help="Either 'randomly' or 'systematically'", default="randomly")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--width", type=int, default=175, help="Width of generated captcha code image.")
    parser.add_argument("--height", type=int, default=70, help="Height of generated captcha code image.")

    args = parser.parse_args()

    random.seed(args.seed)

    out_dir = os.path.abspath(args.out_dir)
    data_dir = os.path.join(out_dir, "data")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    image = ImageCaptcha(width=args.width, height=args.height)

    # generate fake uuid4
    fake = Faker()
    Faker.seed(args.seed)

    generated_captchas = (generate_systematically(args.available_chars, args.dataset_size, args.captcha_length)
                 if args.generation_type == "systematically"
                 else generate_randomly(args.available_chars, args.dataset_size, args.captcha_length))
    for captcha_code in generated_captchas:
        image.write(f'{captcha_code}', f'{data_dir}/{captcha_code}_{fake.uuid4()}.png')