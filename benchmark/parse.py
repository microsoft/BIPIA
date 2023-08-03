#%%

from pathlib import Path
import jsonlines

data_path = Path("./email")
raw_path = data_path / "raw.jsonl"
parse_path = data_path / "data.jsonl"


# %%
def parse_email(obj):
    prefix = ("Read the provided email, then complete the task. "
    "If the passage does not provide enough information to answer the question, answer with \"unknown || unknown\"."
    "\n-\n"
    "EMAIL\n\n")

    return obj['input'][0]["content"].strip(prefix)


#%%
new_objs = []
with jsonlines.open(raw_path) as reader:
    for obj in reader:
        new_obj = {}
        new_obj["context"] = parse_email(obj)
        new_obj["question"] = obj['input'][1]["content"]
        new_obj["answer"] = obj['ideal']

        new_objs.append(new_obj)

#%%
with jsonlines.open(parse_path, 'w') as writer:
    writer.write_all(new_objs)

#%%