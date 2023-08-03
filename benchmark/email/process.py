#%%

from pathlib import Path
import jsonlines
from sklearn.model_selection import train_test_split
import re

data_path = Path(".")
raw_path = data_path / "raw.jsonl"
parse_path = data_path / "data.jsonl"


# %%
def merge_newlines(string):
    merged_string = re.sub(r'\n{3,}', '\n\n', string)
    return merged_string

def merge_lines_with_spaces(string):
    merged_string = re.sub(r'\n\s*\n', '\n\n', string)
    return merged_string

def parse_email(obj):
    prefix = ("Read the provided email, then complete the task. "
    "If the passage does not provide enough information to answer the question, answer with \"unknown || unknown\"."
    "\n-\n"
    "EMAIL\n\n")
    email = obj['input'][0]["content"].strip(prefix)
    email = merge_lines_with_spaces(email)
    email = merge_newlines(email)

    return email

#%%
new_objs = []
with jsonlines.open(raw_path) as reader:
    for obj in reader:
        new_obj = {}
        new_obj["context"] = parse_email(obj)
        new_obj["question"] = obj['input'][1]["content"]
        new_obj["ideal"] = obj['ideal']

        new_objs.append(new_obj)

#%%
with jsonlines.open(parse_path, 'w') as writer:
    writer.write_all(new_objs)

#%%

train_objs, test_objs = train_test_split(new_objs, test_size=0.5, random_state=2023)

with jsonlines.open('./train.jsonl', 'w') as writer:
    writer.write_all(train_objs)

with jsonlines.open('./test.jsonl', 'w') as writer:
    writer.write_all(test_objs)

#%%