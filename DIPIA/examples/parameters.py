import argparse


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
