#%%
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
from pathlib import Path

#%%
df = pd.read_csv("./bug.csv", sep="\t", names=["index", "package", "url"])


#%%
def fetech_accepted_answer(url):
    def format_p(child):
        if not hasattr(child, "childen"):
            return child.get_text()
        
        formatted_text = []
        for t_child in child.children:
            if t_child.name == "code":
                formatted_text.append("`" + t_child.get_text() + "`")
            else:
                formatted_text.append(t_child.get_text())
        return "".join(formatted_text)

    response = requests.get(url)
    html_content = response.content

    # 解析html内容
    soup = BeautifulSoup(html_content, 'html.parser')

    accepted_answer = soup.find('div', {'class': 'accepted-answer'})
    accepted_answer = accepted_answer.find('div', {'class': 'js-post-body'})

    formatted_answer = []


    for child in accepted_answer.children:
        if child.name == "pre":
            formatted_answer.append("```\n" + child.get_text() + "```\n")
        elif child.name == "blockquote":
            formatted_text = []
            for t_child in child.children:
                formatted_text.append("> " + format_p(t_child)) 

            formatted_answer.append("".join(formatted_text))  
        elif not hasattr(child, "childen"):
            formatted_answer.append(child.get_text())
        else:
            formatted_answer.append(format_p(child))  

    answer = "".join(formatted_answer)
    return answer
#%%

rslt = fetech_accepted_answer(df.loc[7, "url"])

# %%
# 31 use the first answer instead of accepeted answer
fail_indexes = [31]
for i in tqdm(df.index):
# for i in tqdm([34]):
    try:
        df.loc[i, "acc_answer"] = fetech_accepted_answer(df.loc[i, "url"])
    except:
        fail_indexes.append(i)

#%%
def parse_md(data):
    data = data.split("\n")
    data = [i.strip("\n") for i in data]
    return data


#%%
for i in tqdm(df.index):
# for i in tqdm([34]):
    if i in fail_indexes:
        continue

    sample = df.loc[i]
    package = sample["package"]
    url = sample["url"]
    index = sample["index"]
    acc_answer = sample["acc_answer"]
    file = Path(f"./data/{package}{index}.json")

    if file.exists():
        with open(file, 'r') as f:
            data = json.load(f)
    else:
        data = {
            "error": [],
            "code": [],
            "context": []
        }

    data["context"] = parse_md(acc_answer)
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

#%%