# BIPIA Evaluation

This folder contains the code to evaluate the robustness of large language models to indirect prompt injection attacks on BIPIA.

## ASR Evaluation

Run the following command to evaluate the attack success rate (ASR) of large language models on BIPIA.
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

## ROUGE Evaluation

We also provide the code to evaluate the performance of large language models on the original tasks (w/o attacks).
```bash
cd examples

# generate respones
python collect_clean_response.py --seed {seed} --dataset_name {task} \
--llm_config_file config/{llm_name}.yaml \
--batch_size 20 --output_path path/of/output/file \
--log_steps 10 --resume

# compute ROUGE score (output a jsonline file with ROUGE scores of all samples)
python run.py --mode capability --seed {seed} \
--dataset_name {task} \
--response_path path/of/output/file \
--output_path path/of/rouge/file \
--batch_size 20 --log_steps 10 --resume
```