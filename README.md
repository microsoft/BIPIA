# BIPIA

Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models.

## Contents
- [Requirements](#requirement)
- [Benchmark](#benchmark)
- [Citation](#citation)

## Requirements
### Step 1
conda envirment preparation

``` pip
pip3 install --upgrade pip
pip install -r requirements.txt
```

## Benchmark

We prepare two pipline to load or bipia dataset. The first one is load with huggingface repo and the other one is set up with the current repo.

### Method 1 (load with HuggingFace)

```

```


### Method 2 (load with Github)

1. Clone this repository and navigate to the BIPIA folder
```bash
git clone https://github.com/yjw1029/BIPIA.git
cd BIPIA
```
2. Setup the package
```bash
pip3 install --upgrade pip
pip3 install -e
```

3. Load dataset with python
``` python
pia_builder = AutoPIABuilder.from_name(dataset_name)(seed)
pia_samples = pia_builder(
    # the path of context text, e.g, './benchmark/abstract/train.jsonl'
    context_data_file,
    # attack text, e.g, './benchmark/text_attack_train.json'
    attack_data_file,
    enable_stealth=enable_stealth,
)
pia_dataset = Dataset.from_pandas(pia_samples)
```

## Defense
We 

## Evaluation

## Citation