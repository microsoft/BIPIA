from functools import partial
import jsonlines
import logging
from tqdm import tqdm
from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader

import datasets
from datasets import Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from bipia.model import AutoLLM
from bipia.data import AutoPIABuilder, DefaultDataCollator, DataCollatorWithPadding
from bipia.data.utils import no_insert
from bipia.metrics import BipiaEvalFactory

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="inference",
        choices=["inference", "evaluate"],
        help="Mode to run run.py: can be set as inference or evaluate."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproduction."
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
    parser.add_argument("--log_steps", type=int, default=None, help="Log output every N steps.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to resume from previous stored file. If the file does not exist test from scracth.",
    )

    args = parser.parse_args()

    return args

logger = get_logger(__name__)

DATA_INFO = {
    "code": ["error", "code", "context", "ideal"],
    "qa": ["context", "question", "ideal"],
    "table": ["context", "question", "ideal"],
    "email": ["context", "question", "ideal"],
    "abstract": ["context", "ideal"],
}

ATTACK_INFO = {
    "code": [
        "Hash",
        "Import",
        "Click",
        "FakeError",
        "Write",
        "SendFile",
        "Delete",
        "InfinityLoop",
        "Reboot",
        "ReverseShell",
    ],
    "qa": [
        "Travel",
        "Recite",
        "Console",
        "Joke",
        "Tone",
        "Interference",
        "Advertisement",
        "Defamation",
        "Intimidation",
        "Translation",
    ],
    "table": [
        "Travel",
        "Recite",
        "Console",
        "Joke",
        "Tone",
        "Interference",
        "Advertisement",
        "Defamation",
        "Intimidation",
        "Translation",
    ],
    "email": [
        "Travel",
        "Recite",
        "Console",
        "Joke",
        "Tone",
        "Interference",
        "Advertisement",
        "Defamation",
        "Intimidation",
        "Translation",
    ],
    "abstract": [
        "Travel",
        "Recite",
        "Console",
        "Joke",
        "Tone",
        "Interference",
        "Advertisement",
        "Defamation",
        "Intimidation",
        "Translation",
    ],
}


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
    pia_samples = pia_builder(args.context_data_file, {"None": ""}, insert_fns=[no_insert])
    pia_dataset = Dataset.from_pandas(pia_samples)

    llm = AutoLLM.from_name(args.llm_config_file)(
        config=args.llm_config_file, accelerator=accelerator
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
                    [
                        " ".join([j["content"] for j in i])
                        for i in processed_datasets["message"]
                    ]
                )

            # read existing results and filter them from datasets
            exist_messages = set()
            with jsonlines.open(output_path, "r") as reader:
                for obj in reader:
                    if obj["attack_name"] in ATTACK_INFO[args.dataset_name]:
                        if isinstance(obj["message"], str):
                            msg = obj["message"]
                        else:
                            msg = " ".join([j["content"] for j in obj["message"]])

                        if msg in needed_messages and msg not in exist_messages:
                            out.append(obj)
                            exist_messages.add(msg)

            def filter_fn(example):
                if isinstance(example["message"], str):
                    return example["message"] not in exist_messages
                else:
                    return (
                        " ".join([j["content"] for j in example["message"]])
                        not in exist_messages
                    )

            with accelerator.main_process_first():
                processed_datasets = processed_datasets.filter(
                    filter_fn, desc="Filter pre-requested messages."
                )

            if len(processed_datasets) == 0:
                logger.info("Already Finished. No need to resume.")

                if args.output_path:
                    with jsonlines.open(args.output_path, "w") as writer:
                        writer.write_all(out)
                exit(0)
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

        data_collator = DataCollatorWithPadding(llm.tokenizer)
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


def evaluate(args):
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.logging_path)
        if args.with_tracking
        else Accelerator()
    )

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

    output_path = Path(args.output_path)
    response_path = Path(args.response_path)

    if not response_path.exists():
        raise ValueError(f"response_path: {args.response_path} does not exists.")

    ds = datasets.load_dataset("json", data_files=args.response_path, split="train")

    if args.output_path:
        output_path = Path(args.output_path)
        out = []
        if output_path.exists() and args.resume:
            if isinstance(ds["message"][0], str):
                needed_messages = set(ds["message"])
            else:
                needed_messages = set(
                    [" ".join([j["content"] for j in i]) for i in ds["message"]]
                )

            # read existing results and filter them from datasets
            exist_messages = set()
            with jsonlines.open(output_path, "r") as reader:
                for obj in reader:
                    if (
                        obj["attack_name"] in ATTACK_INFO[args.dataset_name]
                        and obj["asr"] != -1
                    ):
                        if isinstance(obj["message"], str):
                            msg = obj["message"]
                        else:
                            msg = " ".join([j["content"] for j in obj["message"]])

                        if msg in needed_messages and msg not in exist_messages:
                            out.append(obj)
                            exist_messages.add(msg)

            def filter_fn(example):
                if isinstance(example["message"], str):
                    return example["message"] not in exist_messages
                else:
                    return (
                        " ".join([j["content"] for j in example["message"]])
                        not in exist_messages
                    )

            with accelerator.main_process_first():
                ds = ds.filter(filter_fn, desc="Filter pre-requested messages.")

            if len(ds) == 0:
                logger.info("Already Finished. No need to resume.")

                if args.output_path:
                    with jsonlines.open(args.output_path, "w") as writer:
                        writer.write_all(out)

                exit(0)
        else:
            output_path.parent.mkdir(exist_ok=True, parents=True)
    else:
        raise ValueError(f"output_path: Invalid empty output_path: {args.output_path}.")

    processed_datasets = ds.sort("attack_name")

    evaluator = BipiaEvalFactory(
        gpt_config=args.gpt_config_file,
        activate_attacks=ATTACK_INFO[args.dataset_name],
    )

    data_collator = DefaultDataCollator()

    dataloader = DataLoader(
        processed_datasets, batch_size=args.batch_size, collate_fn=data_collator
    )

    with torch.no_grad():
        for step, data in tqdm(enumerate(dataloader)):
            asrs = evaluator.add_batch(
                predictions=data["response"],
                references=data["target"],
                attacks=data["attack_name"],
                tasks=data["task_name"],
            )

            for attack_name, task_name, target, response, message, position, asr in zip(
                data["attack_name"],
                data["task_name"],
                data["target"],
                data["response"],
                data["message"],
                data["position"],
                asrs,
            ):
                out.append(
                    {
                        "attack_name": attack_name,
                        "task_name": task_name,
                        "response": response,
                        "message": message,
                        "target": target,
                        "position": position,
                        "asr": asr,
                    }
                )

            if args.log_steps and args.output_path:
                if step % args.log_steps == 0:
                    with jsonlines.open(args.output_path, "w") as writer:
                        writer.write_all(out)

    if args.output_path:
        with jsonlines.open(args.output_path, "w") as writer:
            writer.write_all(out)


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "inference":
        inference(args)
    elif args.mode == "evaluate":
        evaluate(args)
    else:
        raise ValueError(f"Invalid mode {args.mode}")
