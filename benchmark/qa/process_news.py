# %%
from pathlib import Path
import re
import random
import jsonlines
from datasets import load_dataset
import string
from itertools import chain
import tiktoken
from transformers import LlamaTokenizer

# %%
newsqa = load_dataset("newsqa", "combined-csv", data_dir="../../raw/newsqa")


# %%
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
    example["story_text"] = merge_lines_with_spaces(example["story_text"])
    example["story_text"] = merge_newlines(example["story_text"])
    return example


# %%
updated_dataset = newsqa.map(
    process_answer, remove_columns=["answer_char_ranges", "story_id"]
)

updated_dataset = updated_dataset.filter(
    lambda x: x["parsed_answer"][1] == "0.0" and x["parsed_answer"][2] == "0.0"
)

# %%

enc_35 = tiktoken.encoding_for_model("gpt-3.5-turbo")
enc_4 = tiktoken.encoding_for_model("gpt-3.5-turbo")
enc_lamma = LlamaTokenizer.from_pretrained(
    "decapoda-research/llama-7b-hf", use_fast=False
)


def length_fn(example, max_len=1024):
    text = example["story_text"]
    gpt35_len = len(enc_35.encode(text))
    gpt4_len = len(enc_4.encode(text))
    lamma_len = len(enc_lamma(text)["input_ids"])

    return max(gpt35_len, gpt4_len, lamma_len) <= max_len


updated_dataset = updated_dataset.filter(length_fn)

# %%
updated_dataset1 = updated_dataset.map(
    process_answer1, remove_columns=["parsed_answer"]
)

# %%
rng = random.Random(2023)

# %%
sample_indexes = rng.sample(range(len(updated_dataset1["train"])), 1000)

# %%
train_indexes = sample_indexes[:900]
test_indexes = sample_indexes[900:]

# %%
train_data = updated_dataset1["train"].select(train_indexes)
test_data = updated_dataset1["train"].select(test_indexes)
# %%

train_objs = []
for sample in train_data:
    new_obj = {}
    new_obj["ideal"] = sample["answers"]
    new_obj["context"] = sample["story_text"]
    new_obj["question"] = sample["question"]
    train_objs.append(new_obj)

# %%
test_objs = []
for sample in test_data:
    new_obj = {}
    new_obj["ideal"] = sample["answers"]
    new_obj["context"] = sample["story_text"]
    new_obj["question"] = sample["question"]
    test_objs.append(new_obj)

# %%
with jsonlines.open("./train.jsonl", "w") as writer:
    writer.write_all(train_objs)

with jsonlines.open("./test.jsonl", "w") as writer:
    writer.write_all(test_objs)

# %%
