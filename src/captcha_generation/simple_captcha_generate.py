import argparse
import itertools
import os
import random

import sys
import uuid

sys.path.insert(0, "src")

from captcha.image import ImageCaptcha
from dataset.annotations_generator import AnnotationsGenerator

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

    parser.add_argument("--test_split", default=0.1, type=float)
    parser.add_argument("--dataset_size", default=10000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--captcha_length", default=4, type=int)
    parser.add_argument("--available_chars", default="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", type=str)
    parser.add_argument("--generation_type", type=str, help="Either 'randomly' or 'systematically'")

    args = parser.parse_args()

    random.seed(args.seed)

    out_dir = os.path.abspath("out")
    data_dir = os.path.join(out_dir, "data")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    image = ImageCaptcha()

    # generate fake uuid4
    fake = Faker()
    Faker.seed(args.seed)

    for captcha_code in generate_randomly(args.available_chars, args.dataset_size, args.captcha_length):
        image.write(f'{captcha_code}', f'{data_dir}/{captcha_code}_{fake.uuid4()}.png')

    train_annotations_path = os.path.join(out_dir, "annotations-train.txt")
    test_annotations_path = os.path.join(out_dir, "annotations-test.txt")

    generator = AnnotationsGenerator(data_dir, args.test_split, True)
    generator.save_annotations(train_annotations_path, test_annotations_path)