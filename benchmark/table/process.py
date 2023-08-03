# %%
from pathlib import Path
import pandas as pd
import random
import jsonlines
from datasets import DatasetDict, Dataset
import tiktoken
from transformers import LlamaTokenizer

root_dir = Path("../../raw/WikiTableQuestions")

# %%
data_dir = root_dir / "data"

# %%
train_file = pd.read_csv(data_dir / "training.tsv", sep="\t")
test_file = pd.read_csv(data_dir / "pristine-unseen-tables.tsv", sep="\t")

train_num = 900
test_num = 100

# %%
ds = DatasetDict(
    {"train": Dataset.from_pandas(train_file), "test": Dataset.from_pandas(test_file)}
)

#%%

def get_table(sample):
    p = root_dir / sample["context"]
    p = p.parent / (p.name.split(".")[0] + ".table")

    file = open(p)
    table_content = file.read()
    file.close()

    sample["context"] = table_content
    return sample

ds = ds.map(get_table)

# %%
# filter long context examples
enc_35 = tiktoken.encoding_for_model("gpt-3.5-turbo")
enc_4 = tiktoken.encoding_for_model("gpt-3.5-turbo")
enc_lamma = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf", use_fast=False)


def length_fn(example, max_len=1024):
    text = example["context"]
    gpt35_len = len(enc_35.encode(text))

    if gpt35_len > max_len:
        return False
    gpt4_len = len(enc_4.encode(text))

    if gpt4_len > max_len:
        return False
    lamma_len = len(enc_lamma(text)["input_ids"])

    return max(gpt35_len, gpt4_len, lamma_len) <= max_len

# %%
filtered_ds = ds.filter(length_fn)

# %%
rng = random.Random(2023)

train_index = rng.sample(list(range(len(filtered_ds["train"]))), train_num)
test_index = rng.sample(list(range(len(filtered_ds["test"]))), test_num)

train_ds = filtered_ds["train"].select(train_index)
test_ds = filtered_ds["test"].select(test_index)

# %%
train_objs = []
for sample in train_ds:
    new_obj = {}
    new_obj["ideal"] = sample["targetValue"]
    new_obj["context"] = sample["context"]
    new_obj["question"] = sample["utterance"]
    train_objs.append(new_obj)

test_objs = []
for sample in test_ds:
    new_obj = {}
    new_obj["ideal"] = sample["targetValue"]
    new_obj["context"] = sample["context"]
    new_obj["question"] = sample["utterance"]
    test_objs.append(new_obj)

# %%
with jsonlines.open("./train.jsonl", "w") as writer:
    writer.write_all(train_objs)

with jsonlines.open("./test.jsonl", "w") as writer:
    writer.write_all(test_objs)

# %%
