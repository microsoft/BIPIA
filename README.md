# BIPIA
The data and code of our work "Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models".

Recent remarkable advancements in large language models (LLMs) have led to their widespread adoption in various applications. A key feature of these applications is the combination of LLMs with external content, where user instructions and third-party content are combined to create prompts for LLM processing. These applications, however, are vulnerable to indirect prompt injection attacks, where malicious instructions embedded within external content compromise LLM's output, causing their responses to deviate from user expectations. Despite the discovery of this security issue, no comprehensive analysis of indirect prompt injection attacks on different LLMs is available due to the lack of a benchmark. Furthermore, no effective defense has been proposed.

In this work, we introduce the first **b**enchmark of **i**ndirect **p**rompt **i**njection **a**ttack, BIPIA, to measure the robustness of various LLMs and defenses against indirect prompt injection attacks. Our experiments reveal that LLMs with greater capabilities exhibit more vulnerable to indirect prompt injection attacks for text tasks, resulting in a higher ASR. We hypothesize that indirect prompt injection attacks are mainly due to the LLMs' inability to distinguish between instructions and external content. Based on this conjecture, we propose four black-box methods based on prompt learning and a white-box defense methods based on fine-tuning with adversarial training to enable LLMs to distinguish between instructions and external content and ignore instructions in the external content. Our experimental results show that our black-box defense methods can effectively reduce ASR but cannot completely thwart indirect prompt injection attacks, while our white-box defense method can reduce ASR to nearly zero with little adverse impact on the LLM's performance on general tasks. We hope that our benchmark and defenses can inspire future work in this important area.

## Paper
If you use this code in your research please cite the following [publication](https://arxiv.org/abs/2312.14197):
```
@article{yi2023benchmarking,
  title={Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models},
  author={Yi, Jingwei and Xie, Yueqi and Zhu, Bin and Hines, Keegan and Kiciman, Emre and Sun, Guangzhong and Xie, Xing and Wu, Fangzhao},
  journal={arXiv preprint arXiv:2312.14197},
  year={2023}
}
```

-----
[LICENSE](https://github.com/microsoft/BIPIA/blob/main/LICENSE)


[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

-----

## Responsible AI Transparency Information 

An AI system includes not only the technology, but also the people who will use it, the people who will be affected by it, and the environment in which it is deployed. Creating a system that is fit for its intended purpose requires an understanding of how the technology works, its capabilities and limitations, and how to achieve the best performance. Microsoft has a broad effort to put our AI principles into practice. To find out more, see [Responsible AI principles from Microsoft](https://www.microsoft.com/en-us/ai/responsible-ai).


### Use of this code

Our goal in publishing this code is to facilitate reproducibility of our paper in hopes of motivating further research in defending against indirect prompt injection attacks. 
Our goal is to enhance the reliable and secure utilization of powerful LLMs and to inspire further research on this crucial issue.
This code should only be used for research on indirect prompt injection attacks.


### Project data 

This project builds a dataset based on data from [OpenAI Evals](https://github.com/openai/evals), [NewsQA](https://arxiv.org/abs/1611.09830), [WikiTableQuestions](https://arxiv.org/abs/1508.00305), [XSum](https://arxiv.org/abs/1808.08745v1), [Stack Exchange](https://archive.org/details/stackexchange).
For more information see `Download the dataset` from the `How to use` section bellow.


### Fairness and Responsible AI testing

At Microsoft, we strive to empower every person on the planet to do more. An essential part of this goal is working to create technologies and products that are fair and inclusive. Fairness is a multi-dimensional, sociotechnical topic and impacts many different aspects of our work.  

When systems are deployed, Responsible AI testing should be performed to ensure safe and fair operation for the specific use case. No Responsible AI testing has been done to evaluate this method including validating fair outcomes across different groups of people. Responsible AI testing should be done before using this code in any production scenario. 

> Note: The documentation included in this ReadMe file is for informational purposes only and is not intended to supersede the applicable license terms. 

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.g


## Requirements
conda envirment preparation

```bash
pip3 install --upgrade pip
pip install -r requirements.txt
```

## How to use

### Download the dataset
In our work, we realse the first **b**enchmark of **i**ndirect **p**rompt **i**njection attack, named BIPIA.
There are two methods to load the dataset.


- Load dataset with huggingface:
```python
from datasets import load_dataset

dataset = load_dataset("bipia", "email")
```

- Load dataset with python
```Python
from bipia import AutoPIABuilder

pia_builder = AutoPIABuilder.from_name(dataset_name)(seed)
pia_samples = pia_builder(
    context_data_file,
    attack_data_file,
    enable_stealth=enable_stealth,
)
pia_dataset = Dataset.from_pandas(pia_samples)
```


### Evaluation
In our work, we evaluate the robustness of 25 existing large language models to indirect prompt injection attacks on BIPIA.
To reproduce the evaluation results in our paper, execute the following commands.

```bash
cd examples

# generate respones
python run.py --seed {seed} --dataset_name {task} \
--context_data_file path/of/external/conten/file \
--attack_data_file path/of/attack/file \
--llm_config_file config/{llm_name}.yaml \
--batch_size 20 --output_path path/of/output/file \
--log_steps 10 --resume

# compute attack success rate
python run.py --mode evaluate --seed {seed} \
--dataset_name {task} \
--response_path path/of/output/file \
--output_path path/of/asr/file \
--gpt_config_file config/{evaluate_llm_name}.yaml \
--batch_size 20 --log_steps 10 --resume
```


### Defense
We also propose two type of defense methods.

- Meta-prompting Defenses
  - Border Strings
  - In-context Learning
  - Multi-turn Dialogue
  - Datamarking

- Finetuning Defenses
  - Speical Tokens

Meanwhile, we relase our defense code for reproducing our results. 

See instructions for running defense at [defense/bipia_defense](defense/README.md).

