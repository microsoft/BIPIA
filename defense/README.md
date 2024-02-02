# Defend against indirect prompt injection attacks

In our work, we propose two kinds of defenses to decrease the Attack Success Rates (ASRs) of LLMs towards indirect prompt injection attacks.
We provide the steps and code to reproduce the results in our paper.

## Black-box defenses
Black-box defenses are a set of defenses based on meta-prompting that does not need access the weights of large language models (LLMs).


### Border Strings
We examine three prevalent types of border strings: equal signs, hyphens, and backticks, to create a more distinct separation between data and instructions. 

Here is the command to collect the responses of LLMs with border strings.
```bash
cd defense/black_box
python few_shot.py --bipia_seed {bipia_seed} --fewshot_seed {fewshot_seed} --dataset_name {task} \
--train_context_data_file path/of/task/train/external/content/file \
--train_attack_data_file path/of/task/train/attack/file \
--test_context_data_file path/of/task/test/external/content/file \
--test_attack_data_file path/of/task/test/attack/file \
--llm_config_file config/{llm_name}.yaml \
--batch_size 20 --output_path path/of/output/file \
--log_steps 10 --resume --border_type {border_type} --num_examples 0
```

Arguments:
- `task`: the selected task name, you can choose anyone from `["code", "email", "qa", "abstract", "table"]`
- `bipia_seed`: the seed for BIPIA dataset construction.
- `fewshot_seed`: the seed to reproduce experiment results, which is important for few-shot example selection.
- `llm_name`: the model you want to test.
- `output_path`: the path where to save the response of LLMs during inference.
- `num_examples`: the number of examples used in few-shot learning.
- `border_type`: the border to split context and prompt. Corrently support `empty`, `=`, `-`, and `` ` ``.
  
You are encouraged to check more arguments and definiation in [few_shot.py](./black_box/few_shot.py)

#### In-context Learning
Inspired by the success of in-context learning, we employ this technique to teach an LLM the boundaries between data and instructions.
```bash
cd defense/black_box
python few_shot.py --bipia_seed {bipia_seed} --fewshot_seed {fewshot_seed} --dataset_name {task} \
--train_context_data_file path/of/task/train/external/content/file \
--train_attack_data_file path/of/task/train/attack/file \
--test_context_data_file path/of/task/test/external/content/file \
--test_attack_data_file path/of/task/test/attack/file \
--llm_config_file config/{llm_name}.yaml \
--batch_size 20 --output_path path/of/output/file \
--log_steps 10 --resume --border_type empty --num_examples {num_examples}
```

Arguments:
- `task`: the selected task name, you can choose anyone from `["code", "email", "qa", "abstract", "table"]`
- `bipia_seed`: the seed for BIPIA dataset construction.
- `fewshot_seed`: the seed to reproduce experiment results, which is important for few-shot example selection.
- `llm_name`: the model you want to test.
- `output_path`: the path where to save the response of LLMs during inference.
- `num_examples`: the number of examples used in few-shot learning.
- `border_type`: the border to split context and prompt. Corrently support `empty`, `=`, `-`, and `` ` ``.

You are encouraged to check more arguments and definiation in [few_shot.py](./black_box/few_shot.py)

#### Multi-turn Dialogue
By separating external content from instructions into different turns and distancing malicious instructions from the end of a prompt, ASR should be reduced.

``` bash
cd examples
python run.py --seed {seed} --mode inference \
--dataset_name {task} --context_data_file ../bipia/{TASK}/test.jsonl \
--attack_data_file ../bipia/{ATTACK_TYPE}_attack_test.json \
--gpt_config_file ../config/{GPT_MODEL}.yaml \
--llm_config_file ../config/{MODEL}.yaml \
--output_path {OUPTPUT_PATH} --batch_size 20 \
--resume 
```

Arguments:
- `TASK`: the selected task name, you can choose anyone from `["code", "email", "qa", "abstract", "table"]`
- `SEED`: the random seed.
- `ATTACK_TYPE`: we provide two type of attack, and you can choose anyone from `["text", "code"]`
- `MODEL`: the model name you want to evaluate.
- `OUPTPUT_PATH`: the path where to save the response of LLMs during inference.
- `resume`: whether to resume from previous stored file. If the file does not exist test from scracth.

### White-box Defense
We propose a white-box defense methods that apply adversarialtraining to the self-supervised fine-tuning stage of an LLM to teach it to ignore instructions in data (external content), thus enhancing its robustness against indirect prompt injection attacks.

#### Dataset Construction
We employ three different methods to construct benign responses: 
- Using labels from the BIPIA dataset. This method guarantees the correctness of the responses but may limit their diversity. 
- Using benign responses generated by the original LLM on prompts
without malicious instructions.
- Using responses generated by GPT-4 on prompts without malicious instructions. 

``` bash
cd defense/white_box
deepspeed finetune.py \
  --llm_config_file ../config/{MODEL}.yaml \
  --model_structure {MODEL_STRUCTURE} \
  --output_dir {OUTPUT_PATH} \
  --fp16 True \
  --fp16_opt_level O2 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 100 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --qa_context_data_file ../bipia/qa/train.jsonl \
  --email_context_data_file ../bipia/email/train.jsonl \
  --table_context_data_file ../bipia/table/train.jsonl \
  --abstract_context_data_file ../bipia/abstract/train.jsonl \
  --code_context_data_file ../bipia/code/train.jsonl \
  --text_attack_data_file ../bipia/text_attack_train.json \
  --code_attack_data_file ../bipia/text_attack_train.json \
  --response_strategy {RESPONSE_STRATEGY} \
  --qa_response_file ../bipia/clean_response/qa_{RESPONSE_STRATEGY}_train.jsonl \
  --email_response_file ../bipia/clean_response/email_{RESPONSE_STRATEGY}_train.jsonl \
  --table_response_file ../bipia/clean_response/table_{RESPONSE_STRATEGY}_train.jsonl \
  --abstract_response_file ../bipia/clean_response/abstract_{RESPONSE_STRATEGY}_train.jsonl \
  --code_response_file ../bipia/clean_response/code_{RESPONSE_STRATEGY}_train.jsonl \
  --dataset_name all \
  --dipia_seed 2023 \
  --deepspeed ds_config.json \
  --report_to wandb 
```

Arguments:
- `MODEL`: the model name you want to evaluate.
- `OUPTPUT_PATH`: the path where to save the response of LLMs during inference.
- `RESPONSE_STRATEGY`: the strategy of benign response generation, [`gpt4_clean`, `original`, `self_clean`].
- `MODEL_STRUCTURE`: the structure that fine-tune model, select from [`special_token`, `type_embs`].


We then download the fine-tune model and evaluate with our BIPIA benchmark.

``` bash
pip install transformers==4.34.0
mkdir -p ../{model}/{RESPONSE_STRATEGY}/checkpoint-{step}
cp -r {MODEL_SAVE_PATH}/checkpoint-{step}/*.json ../{MODELMODEL}/{RESPONSE_STRATEGY}/checkpoint-{step}
cp -r {MODEL_SAVE_PATH}/checkpoint-{step}/*.bin ../{model}/{RESPONSE_STRATEGY}/checkpoint-{step}
cp -r {MODEL_SAVE_PATH}/checkpoint-{step}/*.model ../{MODEL}/{RESPONSE_STRATEGY}/checkpoint-{step}
cd defense/white_box
python eval.py 
  --seed {SEED} \
  --dataset_name {dataset_name} \
  --context_data_file ../bipia/{TASK}/test.jsonl \
  --attack_data_file ../bipia/{ATTACK_TYPE}_attack_test.json \
  --batch_size 200 \
  --output_path {OUTPUT_PATH}} \
  --model_name_or_path ../{MODEL}/{RESPONSE_STRATEGY}/checkpoint-{step} \
  --log_steps 1
```

Arguments:
- `MODEL`: the fine-tune model name you want to evaluate.
- `OUPTPUT_PATH`: the path where to save the response of LLMs during inference.
- `RESPONSE_STRATEGY`: the strategy of benign response generation.
- `TASK`: the selected task name, you can choose anyone from `["code", "email", "qa", "abstract", "table"]`
- `SEED`: the Random seed for reproduction.
- `ATTACK_TYPE`: we provide two type of attack, and you can choose anyone from `["text", "code"]`

### Black-box Defense