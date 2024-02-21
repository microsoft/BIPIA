# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
import jsonlines
from datasets import load_dataset
import string
from itertools import chain
import json
import hashlib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
args = parser.parse_args()

newsqa = load_dataset("newsqa", "combined-csv", data_dir=args.data_dir)


def process_answer(example):
    example["parsed_answer"] = eval(example["answer_char_ranges"])
    return example


def merge_newlines(string):
    merged_string = re.sub(r"\n{3,}", "\n\n", string)
    return merged_string


def merge_lines_with_spaces(string):
    merged_string = re.sub(r"\n\s*\n", "\n\n", string)
    return merged_string


def process_answer1(example):
    story = example["story_text"]
    char_ranges = example["parsed_answer"][0]
    char_ranges = [i.split("|") for i in char_ranges.split(",")]

    answers = []
    for char_range in chain(*char_ranges):
        if char_range != "None":
            start, end = map(int, char_range.split(":"))
            answer = story[start:end].strip(string.punctuation + string.whitespace)
            answers.append(answer)

    example["answers"] = list(set(answers))
    example["answers"].sort()
    example["story_text"] = merge_lines_with_spaces(example["story_text"])
    example["story_text"] = merge_newlines(example["story_text"])
    return example


newsqa = newsqa.map(process_answer, remove_columns=["answer_char_ranges", "story_id"])

with open("./index.json", "r") as f:
    indexes = json.load(f)

train_data = newsqa["train"].select(indexes["train"])
test_data = newsqa["train"].select(indexes["test"])


train_data = train_data.map(process_answer1, remove_columns=["parsed_answer"])

test_data = test_data.map(process_answer1, remove_columns=["parsed_answer"])

train_objs = []
for sample in train_data:
    new_obj = {}
    new_obj["ideal"] = sample["answers"]
    new_obj["context"] = sample["story_text"]
    new_obj["question"] = sample["question"]
    train_objs.append(new_obj)

test_objs = []
for sample in test_data:
    new_obj = {}
    new_obj["ideal"] = sample["answers"]
    new_obj["context"] = sample["story_text"]
    new_obj["question"] = sample["question"]
    test_objs.append(new_obj)

test_objs[87]["ideal"] = ["Janine Sligar"]


with jsonlines.open("./train.jsonl", "w") as writer:
    writer.write_all(train_objs)

with jsonlines.open("./test.jsonl", "w") as writer:
    writer.write_all(test_objs)

md5_values = {}
md5 = hashlib.md5()
with open("./train.jsonl", "rb") as f:
    md5.update(f.read())
    md5_train = md5.hexdigest()
    md5_values["train.jsonl"] = md5_train

md5 = hashlib.md5()
with open("./test.jsonl", "rb") as f:
    md5.update(f.read())
    md5_test = md5.hexdigest()
    md5_values["test.jsonl"] = md5_test

with open("./md5.txt", "r") as f:
    for line in f:
        md5_old, file_name = line.strip().split("  ")
        assert md5_values[file_name] == md5_old, f"{file_name} md5 mismatch"
        print(f"{file_name} md5 match")
