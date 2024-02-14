# BIPIA Dataset

This directory contains the files for constructing BIPIA dataset.

## Structure

- `email`: the context file of the EmailQA task
    - test.jsonl: the test context file the EmailQA task
    - train.jsonl: the train context file of the EmailQA task
- `table`: the context file of the TableQA task
    - test.jsonl: the test context file the TableQA task
    - train.jsonl: the train context file of the TableQA task
- `code`: the code context file of the CodeQA task
    - test.jsonl: the test context file the CodeQA task
    - train.jsonl: the train context file of the CodeQA task
- `qa`: the context file of the WebQA task. Due to the license issue, please follow the [Generate Context Data](#webqa-task) section to generate the context files `test.jsonl` and `train.jsonl`
    - md5.txt: the md5 file of the context files
    - index.json: the sample indexes of the context files
    - process.py: the script to process the context files
- `abstract`: the context file of the Summarization task. Due to the license issue, please follow the [Generate Context Data](#summarization-task) section to generate the context files `test.jsonl` and `train.jsonl`
    - md5.txt: the md5 file of the context files
    - index.json: the sample indexes of the context files
    - process.py: the script to process the context files
- `code_attack_test.json`: the test attack file of the code tasks
- `code_attack_train.json`: the train attack file of the code tasks
- `text_attack_test.json`: the test attack file of the text tasks
- `text_attack_train.json`: the train attack file of the text tasks

## Generate Context Data

### WebQA Task

Read and follow the guidlines in [newsqa official repo](https://github.com/Maluuba/newsqa). If you fail to build the docker, consider use [bryant1410/newsqa](https://hub.docker.com/r/bryant1410/newsqa) instead. 
```bash
docker pull bryant1410/newsqa
docker run --rm -it -v ${PWD}:/usr/src/newsqa --name newsqa bryant1410/newsqa
```
After obtaining the `combined-newsqa-data-v1.csv` and `combined-newsqa-data-v1.json`, run the following command to generate the context files.
```bash
cd qa
python process.py --data_dir /path/to/newsqa
```

### Summarization Task

Read and follow the guildlines in [XSum dataset](https://github.com/EdinburghNLP/XSum). Run the following command to generate the context files.
```bash
cd abstract
python process.py
```