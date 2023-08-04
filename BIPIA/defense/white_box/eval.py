import argparse

from typing import List
from functools import partial
import jsonlines
import logging
import sys
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import transformers
from transformers import AutoModelForCausalLM

import datasets
from datasets import Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from bipia.model import Vicuna
from bipia.data import AutoPIABuilder, DefaultDataCollator

from utils import DATA_INFO, TEST_ATTACK_INFO as ATTACK_INFO, DataCollatorWithPaddingAndLabel
from type_emb_model import LlamaModelWTypeEmbedCausalLM

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproduction."
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default=None, help="The model name on huggingface."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["code", "email", "qa", "abstract", "table"],
    )
    parser.add_argument("--context_data_file", type=str)
    parser.add_argument("--attack_data_file", type=str)
    parser.add_argument("--llm_config_file", type=str, default=None)
    parser.add_argument(
        "--gpt_config_file",
        type=str,
        default=None,
        help="A config file of GPT for model evaluations.",
    )

    parser.add_argument("--batch_size", type=int, default=None)

    parser.add_argument(
        "--response_path",
        type=str,
        default=None,
        help="Path of responses file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to export responses and evaluation results",
    )
    parser.add_argument(
        "--logging_path",
        type=str,
        default=None,
        help="Path to export logging results",
    )

    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--log_steps", type=int, default=None, help="Log output every N steps."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to resume from previous stored file. If the file does not exist test from scracth.",
    )

    parser.add_argument(
        "--add_context_type_embedding",
        action="store_true",
        help="Whether to add type embeddings.",
    )
    parser.add_argument(
        "--add_special_context_token",
        action="store_true",
        help="Whether to add spetial tokens.",
    )

    args = parser.parse_args()

    return args


class VicunaWithSpecialToken(Vicuna):
    require_system_prompt = False

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"],
            load_in_8bit=self.config["load_8bit"],
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        return self.model

    def process_fn(
        self, example, prompt_construct_fn
    ):
        user_prompt = prompt_construct_fn(example)
        example["target"] = example["ideal"]

        message = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            f"USER: {user_prompt} "
            f"ASSISTANT:"
        )

        example["message"] = message

        context_str = example["context"]
        start_index = message.find(context_str)
        end_index = start_index + len(context_str)

        message = [
            message[:start_index],
            message[start_index:end_index],
            message[end_index:],
        ]

        tokens = [
            self.tokenizer.tokenize(t, is_split_into_words=True)
            for t in message
        ]

        tokens = tokens[0] + ["<data>"] + tokens[1] + ["</data>"] + tokens[2]
        example["input_ids"] = [self.tokenizer.bos_token_id] + self.tokenizer.convert_tokens_to_ids(tokens)

        example["attention_mask"] = [1] * len(example["input_ids"])

        return example

class VicunaWithTypeEmb(Vicuna):
    require_system_prompt = False

    def load_model(self):
        self.model = LlamaModelWTypeEmbedCausalLM.from_pretrained(
            self.config["model_name"],
            load_in_8bit=self.config["load_8bit"],
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        return self.model

    def process_fn(
        self, example, prompt_construct_fn
    ):
        user_prompt = prompt_construct_fn(example)
        example["target"] = example["ideal"]

        message = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            f"USER: {user_prompt} "
            f"ASSISTANT:"
        )

        example["message"] = message

        context_str = example["context"]
        start_index = message.find(context_str)
        end_index = start_index + len(context_str)

        message = [
            message[:start_index],
            message[start_index:end_index],
            message[end_index:],
        ]

        tokens = [
            self.tokenizer.tokenize(t, is_split_into_words=True)
            for t in message
        ]

        example["type_ids"] = [0] * (len(tokens[0]) + 1) + [1] * len(tokens[1]) + [0] * len(tokens[2])
        tokens = tokens[0] + tokens[1] + tokens[2]
        example["input_ids"] = [self.tokenizer.bos_token_id] + self.tokenizer.convert_tokens_to_ids(tokens)
        example["attention_mask"] = [1] * len(example["input_ids"])
        
        return example

    @torch.no_grad()
    def generate(self, data):
        input_ids = torch.as_tensor(data["input_ids"]).cuda()
        type_ids = torch.as_tensor(data["type_ids"]).cuda()

        stopping_criteria = self.load_stopping_criteria(input_ids)

        output_ids = self.model.generate(
            input_ids=input_ids,
            type_ids=type_ids,
            generation_config=self.generation_config,
            stopping_criteria=stopping_criteria,
        )

        output_ids = output_ids[:, input_ids.shape[1] :]

        responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        responses = self.post_process(responses)
        return responses

def inference(args):
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.logging_path)
        if args.with_tracking
        else Accelerator()
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    pia_builder = AutoPIABuilder.from_name(args.dataset_name)(args.seed)
    pia_samples = pia_builder(args.context_data_file, args.attack_data_file)
    pia_dataset = Dataset.from_pandas(pia_samples)
    
    config = {"load_8bit": False, "model_name": args.model_name_or_path}
    
    if args.add_special_context_token:
        llm = VicunaWithSpecialToken(
            config=config, accelerator=accelerator
        )
    elif args.add_context_type_embedding:
        llm = VicunaWithTypeEmb(
            config=config, accelerator=accelerator
        )

    with accelerator.main_process_first():
        processed_datasets = pia_dataset.map(
            partial(
                llm.process_fn,
                prompt_construct_fn=partial(
                    pia_builder.construct_prompt,
                    require_system_prompt=llm.require_system_prompt,
                ),
            ),
            remove_columns=DATA_INFO[args.dataset_name],
            desc="Processing Indirect PIA datasets.",
        )
    
    if args.output_path:
        output_path = Path(args.output_path)
        out = []
        if output_path.exists() and args.resume:
            if isinstance(processed_datasets["message"][0], str):
                needed_messages = set(processed_datasets["message"])
            else:
                needed_messages = set(
                    [i[0]["content"] for i in processed_datasets["message"]]
                )

            # read existing results and filter them from datasets
            exist_messages = set()
            with jsonlines.open(output_path, "r") as reader:
                for obj in reader:
                    if obj["attack_name"] in ATTACK_INFO[args.dataset_name]:
                        if isinstance(obj["message"], str):
                            msg = obj["message"]
                        else:
                            msg = obj["message"][0]["content"]

                        if msg in needed_messages:
                            out.append(obj)
                            exist_messages.add(msg)

            def filter_fn(example):
                if isinstance(example["message"], str):
                    return example["message"] not in exist_messages
                else:
                    return example["message"][0]["content"] not in exist_messages

            with accelerator.main_process_first():
                processed_datasets = processed_datasets.filter(
                    filter_fn, desc="Filter pre-requested messages."
                )

            if len(processed_datasets) == 0:
                logger.info("Already Finished. No need to resume.")
                exit(0)

            num_examples = len(processed_datasets)
            logger.info(f"Need to process {num_examples} samples.")
        else:
            output_path.parent.mkdir(exist_ok=True, parents=True)

    if "input_ids" in processed_datasets.column_names:
        # sort by length if based on huggingface transformers models
        def set_length(example):
            example["length"] = len(example["input_ids"])
            return example

        with accelerator.main_process_first():
            processed_datasets = processed_datasets.map(set_length)
            processed_datasets = processed_datasets.sort("length")
            processed_datasets = processed_datasets.remove_columns("length")

        data_collator = DataCollatorWithPaddingAndLabel(llm.tokenizer)
    else:
        data_collator = DefaultDataCollator()

    dataloader = DataLoader(
        processed_datasets, batch_size=args.batch_size, collate_fn=data_collator
    )

    with torch.no_grad():
        for step, data in tqdm(enumerate(dataloader)):
            responses = llm.generate(data)
            if args.output_path:
                for attack_name, task_name, target, response, message, position in zip(
                    data["attack_name"],
                    data["task_name"],
                    data["target"],
                    responses,
                    data["message"],
                    data["position"],
                ):
                    out.append(
                        {
                            "attack_name": attack_name,
                            "task_name": task_name,
                            "response": response,
                            "message": message,
                            "target": target,
                            "position": position,
                        }
                    )

                if args.log_steps:
                    if step % args.log_steps == 0:
                        with jsonlines.open(args.output_path, "w") as writer:
                            writer.write_all(out)

    if args.output_path:
        with jsonlines.open(args.output_path, "w") as writer:
            writer.write_all(out)



if __name__ == "__main__":
    args = parse_args()
    inference(args)
