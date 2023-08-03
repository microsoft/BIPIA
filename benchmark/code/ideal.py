# %%
import json
import traceback

# pickle 1 and 8 are based on warning
run_indexes = {
    "json_pickle": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "torch": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "nltk": [1, 2, 3],
    "numpy": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "utils": list(range(1, 16)) + list(range(17, 27)),
    "seaborn_matplotlib": [1, 2, 3, 4, 5, 6, 7, 8],
    "sklearn": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17],
    "pandas": list(range(1, 12)),
}

# %%
# 1 and 8 return warning
for key in ["nltk"]:
    for i in run_indexes[key]:
    # for i in [13]:
        try:
            with open(f"./data/{key}{i}.json", "r") as f:
                data = json.load(f)

            print(key, i)
            exec("\n".join(data["ideal"]))
            if "test" in "\n".join(data["ideal"]):
                test()
        except:
            print(f"=================== {i} ========================")
            trace = traceback.format_exc()
            print(trace)
            trace = trace.split("\n")
            # data["error"] = [i.strip("\n") for i in trace]
