#%%
from pathlib import Path
import json
import jsonlines
from sklearn.model_selection import train_test_split

data_dir = Path("./data")

objs = {}
for file in data_dir.iterdir():
    with open(file, 'r') as f:
        objs[file.name] =json.load(f)

#%%
objs = list(map(lambda x: x[1], sorted(objs.items(), key=lambda x: x[0])))

# %%
with jsonlines.open('./data.jsonl', 'w') as writer:
    writer.write_all(objs)

# %%
train_objs, test_objs = train_test_split(objs, test_size=0.5, random_state=2023)

with jsonlines.open('./train.jsonl', 'w') as writer:
    writer.write_all(train_objs)

with jsonlines.open('./test.jsonl', 'w') as writer:
    writer.write_all(test_objs)

#%%