# BIPIA

Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models.

## Contents
- [BIPIA](#bipia)
  - [Contents](#contents)
  - [Requirements](#requirements)
  - [Benchmark](#benchmark)
    - [Preparation](#preparation)
    - [Method 1](#method-1)
    - [Method 2](#method-2)
  - [Evaluation](#evaluation)
      - [Evaluation Metrcs](#evaluation-metrcs)
    - [Supported Models](#supported-models)
      - [1. Write a model config](#1-write-a-model-config)
      - [2. Definition a new py **model class** in bipia.model](#2-definition-a-new-py-model-class-in-bipiamodel)
      - [3. Set up package again](#3-set-up-package-again)
  - [Defense](#defense)
  - [Citation](#citation)

## Requirements
conda envirment preparation

``` pip
pip3 install --upgrade pip
pip install -r requirements.txt
```

## Benchmark

BIPIA, which is designed to evaluate defenses against indirect prompt injection attacks on LLMs. 

We release the benchmark and details of this benchmark can be found in the [huggingface repo](https://huggingface.co/datasets/yjw1029/bipia).

We prepare two pipline to load or bipia dataset. 

The first one is load with [huggingface repo](https://huggingface.co/datasets/yjw1029/bipia) and the other one is set up with the [current repo](https://github.com/yjw1029/BIPIA.git).

### Preparation
We need setup the BIPIA package manually and we will release a package soon for easily install.

- **Clone this repository and navigate to the BIPIA folder**
```bash
git clone https://github.com/yjw1029/BIPIA.git
cd BIPIA
```
- **Setup the package**
```bash
pip3 install --upgrade pip
pip3 install -e
```

### Method 1

- **Load dataset with huggingface:**
```python
from datasets import load_dataset

# Take email QA as example
dataset = load_dataset("bipia", "email")
```

### Method 2

- **Load dataset with python**
``` python
pia_builder = AutoPIABuilder.from_name(dataset_name)(seed)
pia_samples = pia_builder(
    # the path of context text, e.g, './benchmark/abstract/train.jsonl'
    context_data_file,
    # the path of attack text, e.g, './benchmark/text_attack_train.json'
    attack_data_file,
    enable_stealth=enable_stealth,
)
pia_dataset = Dataset.from_pandas(pia_samples)
```

## Evaluation

We relase our evaluation code for results reproducibly.

See instructions for running evaluation at [examples/bipia_eval](examples/bipia_eval.md).

#### Evaluation Metrcs
We use the **attack success rate (ASR)** as the primary metric to evaluate an LLM’s susceptibility. 

We design three approachesto verify the success of the attack for different attacks:
  - **Rule-base Match**
    - We check if the model’s answer contains pre-defined malicious strings.
  - **LLM-as-judge**
    - We utilize another advanced large language model, such as GPT-4, to assess if the attack is successful based on the model’s output. 
  - **Langdetect**
    - We utilize the langdetect package14 to judge whether a response is in the language specified in the prompt.

### Supported Models

BIPIA supports a wide range of models, including **GPT-4**, **GPT-3.5-turbo**, **Vicuna-33B**, **Vicuna-13B**, **MPT-30B-Chat**, **Guanaco-33B**, **WizardLM-13B**, **Vicuna-7B**, **Koala-13B**, **GPT4All-13B-Snoozy**, **MPT-7B-Chat**, **RWKV-4-raven-14B**, **Alpaca-13B**, **OpenAssistant-Pythia-12B**, **ChatGLM-6B**, **FastChat-T5-3B**, **StableLM-Tuned-Alpha-7B**, and **Dolly-V2-12B**, and more.

We can easily add a new model by writing a model config and rewrite a new class in model in bipia.model.

#### 1. Write a model config

We give a config example of Vicuna 7b model.
``` json
load_8bit: False # wether use quantiation
model_name: lmsys/vicuna-7b-v1.5 # model id in huggingface.
llm_name: "vicuna" # customized model name 
```

#### 2. Definition a new py **model class** in bipia.model

You can define a any function in base class to fit the model and finall add customized model name in `__all__` tuple in PY file.

We giva a class example of llama2 model.
``` python
# XXXX can be any customized model name
class XXXX(LLAMAModel):
    require_system_prompt = False
```

#### 3. Set up package again
```bash
pip3 install -e.
```

## Defense
We also give propose two type of defense methods, regarding to the white-box and black-box large language model.

- **Black-Box Defense**
  - Border Strings
  - In-context Learning
  - Multi-turn Dialogue

- **White-box Defense**
  - Speical Tokens
  - Type Tokens Embedding

Meanwhile, we relase our defense code for results reproducibly. 

See instructions for running defense at [defense/bipia_defense](defense/bipia_defense.md).

## Citation
Please cite the following paper if you find the code or datasets helpful.
