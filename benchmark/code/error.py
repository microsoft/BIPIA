# %%
import json
import traceback

# pickle 1 and 8 are based on warning
run_indexes = {
    "pickle": [2, 3, 4, 5, 6, 7, 9, 10],
    "json_pickle": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "torch": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "nltk": [1, 2, 3],
    "numpy": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "utils": list(range(1, 16)) + list(range(17, 28)),
    "seaborn_matplotlib": [1, 2, 3, 4, 5, 6, 7, 8],
    "sklearn": list(range(1, 19)),
    "pandas": list(range(1, 12)),
}

# %%
# 1 and 8 return warning
for key in ["nltk"]:
    # for i in run_indexes[key]:
    for i in [1]:
        try:
            with open(f"./data/{key}{i}.json", "r") as f:
                data = json.load(f)

            # print("\n".join(data["code"]))
            exec("\n".join(data["code"]))
            if "test" in "\n".join(data["code"]):
                test()
        except:
            print(f"=================== {i} ========================")
            trace = traceback.format_exc()
            print(trace)
            
            trace = trace.split("\n")
            data["error"] = [i.strip("\n") for i in trace]

        with open(f"./data/{key}{i}.json", "w") as f:
            data = json.dump(data, f, indent=4)
