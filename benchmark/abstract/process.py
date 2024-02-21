# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import jsonlines
from datasets import load_dataset
import json
import hashlib

enter = input(
    "Please follow the guildlines in https://github.com/EdinburghNLP/XSum and https://huggingface.co/datasets/EdinburghNLP/xsum to download and use the XSum dataset. Press [Y] to continue:"
)
if enter.lower() == "y":
    ds = load_dataset("xsum")
else:
    raise ValueError("Please read the guildlines and run the script again.")

with open("./index.json", "r") as f:
    indexes = json.load(f)

train_ds = ds["train"].select(indexes["train"])
test_ds = ds["test"].select(indexes["test"])


train_objs = []
for sample in train_ds:
    new_obj = {}
    new_obj["ideal"] = sample["summary"]
    new_obj["context"] = sample["document"]
    train_objs.append(new_obj)

test_objs = []
for sample in test_ds:
    new_obj = {}
    new_obj["ideal"] = sample["summary"]
    new_obj["context"] = sample["document"]
    test_objs.append(new_obj)


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
