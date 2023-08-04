# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from typing import List, Dict, Optional, Sequence
from dataclasses import dataclass, field
import jsonlines
import wandb
import pathlib
from copy import deepcopy

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import datasets
from datasets import concatenate_datasets

import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from bipia.data import AutoPIABuilder

from utils import DATA_INFO, DataCollatorWithPaddingAndLabel
from type_emb_model import LlamaModelWTypeEmbedCausalLM

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The system prompt
    system: str
    # Two roles
    roles: List[str]
    # All messages. Each item is (role, message).
    messages: List[List[str]]
    # The number of few shot examples
    offset: int
    sep: str
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        seps = [self.sep, self.sep2]
        ret = self.system + seps[0]
        for i, (role, message) in enumerate(self.messages):
            if message:
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ":"
        return ret

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-13b-v1.3")


@dataclass
class DataArguments:
    dataset_name: str = field(
        default=None,
        metadata={"help": "Which BIPIA dataset used for training and evaluation."},
    )
    qa_context_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "The context file used for qa in BIPIA dataset."},
    )
    email_context_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "The context file used for email in BIPIA dataset."},
    )
    abstract_context_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "The context file used for abstract in BIPIA dataset."},
    )
    table_context_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "The context file used for table in BIPIA dataset."},
    )
    code_context_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "The context file used for code in BIPIA dataset."},
    )

    text_attack_data_file: Optional[str] = field(
        default=None, metadata={"help": "The text attack file used in BIPIA dataset."}
    )
    code_attack_data_file: Optional[str] = field(
        default=None, metadata={"help": "The code attack file used in BIPIA dataset."}
    )
    bipia_seed: Optional[int] = field(
        default=None,
        metadata={
            "help": "The seed used in data generation (especially the middle position insertation)."
        },
    )
    add_special_context_token: bool = field(
        default=False,
        metadata={"help": "Whether to add special tokens for context boarder."},
    )
    add_context_type_embedding: bool = field(
        default=False,
        metadata={"help": "Whether to add type embeddings for context tokens."},
    )

    response_strategy: str = field(
        default="original",
        metadata={"help": "The strategy to construct robust response for training. Currently support original, self_clean and gpt4_clean."}
    )

    qa_response_file: Optional[str] = field(
        default=None,
        metadata={"help": "The response file used for qa if response strategy is self_clean and gpt4_clean."},
    )
    email_response_file: Optional[str] = field(
        default=None,
        metadata={"help": "The response file used for email if response strategy is self_clean and gpt4_clean."},
    )
    abstract_response_file: Optional[str] = field(
        default=None,
        metadata={"help": "The response file used for abstract if response strategy is self_clean and gpt4_clean."},
    )
    table_response_file: Optional[str] = field(
        default=None,
        metadata={"help": "The response file used for table if response strategy is self_clean and gpt4_clean."},
    )
    code_response_file: Optional[str] = field(
        default=None,
        metadata={"help": "The response file used for code if response strategy is self_clean and gpt4_clean."},
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    mode: str = field(default="train")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    wandb_project_name: str = field(
        default=os.environ.get("WANDB_PROJECT", ""),
        metadata={"help": "The project name of wandb."},
    )
    wandb_run_name: str = field(
        default=os.environ.get("WANDB_RUN", ""),
        metadata={"help": "The run name of wandb."},
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_list: List,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_tokens(special_tokens_list)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def response_fn(example, pia_builder, responses, response_strategy: str):

    if response_strategy == "original":
        return pia_builder.construct_response(example)
    elif response_strategy == "self_clean":
        attack = pia_builder.attacks[example["attack_name"]]
        context = example["context"]
        
        attack_start_index = context.find(attack)
        attack_end_index = attack_start_index + len(attack)
        if example["position"] == "end":
            attack_start_index = attack_start_index - 1
        elif example["position"] == "start":
            attack_end_index = attack_end_index + 1
        else:
            attack_start_index = attack_start_index - 1
            attack_end_index = attack_end_index + 1

        clean_context = "".join([context[:attack_start_index], context[attack_end_index:]])
        new_example = {
            "context": clean_context,
            "question": example["question"] if "question" in example else "",
            "error": example["error"] if "error" in example else "",
            "code": example["code"] if "code" in example else "",
            }
        message = pia_builder.construct_prompt(new_example, False)

        clean_response = responses[message]   
        return clean_response
    elif response_strategy == "gpt4_clean":
        attack = pia_builder.attacks[example["attack_name"]]
        context = example["context"]
        
        attack_start_index = context.find(attack)
        attack_end_index = attack_start_index + len(attack)
        if example["position"] == "end":
            attack_start_index = attack_start_index - 1
        elif example["position"] == "start":
            attack_end_index = attack_end_index + 1
        else:
            attack_start_index = attack_start_index - 1
            attack_end_index = attack_end_index + 1

        clean_context = "".join([context[:attack_start_index], context[attack_end_index:]])
        new_example = {
            "context": clean_context,
            "question": example["question"] if "question" in example else "",
            "error": example["error"] if "error" in example else "",
            "code": example["code"] if "code" in example else "",
            }
        system_prompt, _ = pia_builder.construct_prompt(new_example, True)

        try:
            clean_response = responses[system_prompt]   
        except:
            import pdb; pdb.set_trace()
        return clean_response


def load_bipia_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    context_data_files = {
        "qa": (data_args.qa_context_data_file, data_args.text_attack_data_file),
        "abstract": (data_args.abstract_context_data_file, data_args.text_attack_data_file),
        "email": (data_args.email_context_data_file, data_args.text_attack_data_file),
        "table": (data_args.table_context_data_file, data_args.text_attack_data_file),
        "code": (data_args.code_context_data_file, data_args.code_attack_data_file)
    }

    response_files = {
        "qa": data_args.qa_response_file,
        "abstract": data_args.abstract_response_file,
        "email": data_args.email_response_file,
        "table": data_args.table_response_file,
        "code": data_args.code_response_file
    }
    if data_args.dataset_name == "all":
        processed_datasets = {}
        for dataset_name in ["qa", "abstract", "email", "table", "code"]:
            pia_builder = AutoPIABuilder.from_name(dataset_name)(data_args.bipia_seed)
            pia_samples = pia_builder(
                context_data_files[dataset_name][0], context_data_files[dataset_name][1]
            )
            pia_dataset = datasets.Dataset.from_pandas(pia_samples)

            responses = {}
            if data_args.response_strategy == "self_clean":
                with jsonlines.open(response_files[dataset_name], 'r') as reader:
                    for obj in reader:
                        message = obj["message"][161:-11]
                        responses[message] = obj["response"]
            elif data_args.response_strategy == "gpt4_clean":
                with jsonlines.open(response_files[dataset_name], 'r') as reader:
                    for obj in reader:
                        message = obj["message"][0]["content"]
                        responses[message] = obj["response"]

            def process_fn(example):
                user_prompt = pia_builder.construct_prompt(example, require_system_prompt=False)
                response = response_fn(example, pia_builder, responses, data_args.response_strategy)
                source = (user_prompt, response)

                example["conversation"] = source
                return example

            remove_columns = DATA_INFO[dataset_name]
            remove_columns.remove("context")

            processed_datasets[dataset_name] = pia_dataset.map(
                process_fn,
                remove_columns = remove_columns,
                desc="Processing Indirect PIA datasets.",
            )
        
        processed_datasets = concatenate_datasets(list(processed_datasets.values()))
    else:
        pia_builder = AutoPIABuilder.from_name(data_args.dataset_name)(
            data_args.bipia_seed
        )
        pia_samples = pia_builder(
            context_data_files[data_args.dataset_name][0], context_data_files[data_args.dataset_name][1]
        )
        pia_dataset = datasets.Dataset.from_pandas(pia_samples)
    
        def process_fn(example):
            user_prompt = pia_builder.construct_prompt(example, require_system_prompt=False)
            response = pia_builder.construct_response(example)
            source = (user_prompt, response)

            example["conversation"] = source
            return example

        remove_columns = DATA_INFO[data_args.dataset_name]
        remove_columns.remove("context")
        processed_datasets = pia_dataset.map(
            process_fn,
            remove_columns = remove_columns,
            # num_proc=4,
            desc="Processing Indirect PIA datasets.",
        )

    system = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    )
    system_input_ids = tokenizer(system, add_special_tokens=False).input_ids

    human = "USER: "
    human_input_ids = tokenizer(human, add_special_tokens=False).input_ids

    assistant = " ASSISTANT: "
    assistant_input_ids = tokenizer(assistant, add_special_tokens=False).input_ids

    def tokenize_fn(example):
        user_prompt, response = example["conversation"]
        context_str = example["context"]

        start_index = user_prompt.find(context_str)
        end_index = start_index + len(context_str)

        user_prompt = [
            user_prompt[:start_index],
            user_prompt[start_index:end_index],
            user_prompt[end_index:],
        ]

        tokens = [
            tokenizer.tokenize(t, is_split_into_words=True)
            for t in user_prompt + [response]
        ]

        if data_args.add_special_context_token:
            user_conv = tokens[0] + ["<data>"] + tokens[1] + ["</data>"] + tokens[2]
        else:
            user_conv = tokens[0] + tokens[1] + tokens[2]

        user_conv_input_ids = tokenizer.convert_tokens_to_ids(user_conv)
        response_input_ids = tokenizer.convert_tokens_to_ids(tokens[-1])

        input_ids = (
            [tokenizer.bos_token_id]
            + system_input_ids
            + human_input_ids
            + user_conv_input_ids
            + assistant_input_ids
            + response_input_ids
            + [tokenizer.eos_token_id]
        )
        target = deepcopy(input_ids)
        target[: 1 + len(system_input_ids + human_input_ids + user_conv_input_ids + assistant_input_ids)] = [
            IGNORE_TOKEN_ID
        ] * (1 + len(system_input_ids + human_input_ids + user_conv_input_ids + assistant_input_ids))

        example["input_ids"] = input_ids
        example["attention_mask"] = [1] * len(input_ids)
        example["labels"] = target

        if data_args.add_context_type_embedding:
            example["type_ids"] = (
                [0]
                * (1 + len(system_input_ids) + len(human_input_ids) + len(tokens[0]))
                + [1] * len(tokens[1])
                + [0] * (len(tokens[2]) + len(assistant_input_ids) + len(response_input_ids) + 1)
            )

        return example

    processed_datasets = processed_datasets.map(
        tokenize_fn,
        remove_columns=[
            "conversation",
            "attack_name",
            "task_name",
            "position",
            "context"
        ],
        desc="Tokenize Indirect PIA datasets.",
    )

    processed_datasets = processed_datasets.filter(
        lambda x: len(x["input_ids"]) <= tokenizer.model_max_length,
        desc="Filter long sequences.",
    )

    return processed_datasets


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # load model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # change model achietecture
    if data_args.add_special_context_token:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
        model.config.use_cache = False

        special_tokens = ["<data>", "</data>"]
        smart_tokenizer_and_embedding_resize(special_tokens, tokenizer, model)

    if data_args.add_context_type_embedding:
        model = LlamaModelWTypeEmbedCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
        model.config.use_cache = False

    # load training data from bipia
    train_dataset = load_bipia_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args
    )

    if local_rank == 0:
        wandb.init(
            project=training_args.wandb_project_name,
            name=training_args.wandb_run_name,
        )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorWithPaddingAndLabel(tokenizer),
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
