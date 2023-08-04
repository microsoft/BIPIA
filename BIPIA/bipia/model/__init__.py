import yaml
from pathlib import Path
from collections import OrderedDict
from accelerate.logging import get_logger

from .gpt import GPT35, GPT4, GPT35WOSystem, GPT4WOSystem
from .llama import Alpaca, Vicuna, Baize, StableVicuna, Koala, GPT4ALL, Wizard, Guanaco
from .llm import Dolly, OASST, ChatGLM, StableLM, MPT, FastChatT5, RwkvModel

logger = get_logger(__name__)

LLM_NAME_TO_CLASS = OrderedDict(
    [
        ("gpt35", GPT35),
        ("gpt4", GPT4),
        ("gpt35_wosys", GPT35WOSystem),
        ("gpt4_wosys", GPT4WOSystem),
        ("alpaca", Alpaca),
        ("vicuna", Vicuna),
        ("baize", Baize),
        ("stablelm", StableLM),
        ("stablevicuna", StableVicuna),
        ("dolly", Dolly),
        ("rwkv", RwkvModel),
        ("oasst", OASST),
        ("chatglm", ChatGLM),
        ("koala", Koala),
        ("mpt", MPT),
        ("t5", FastChatT5),
        ("gpt4all", GPT4ALL),
        ("wizard", Wizard),
        ("guanaco", Guanaco)
    ]
)


class AutoLLM:
    @classmethod
    def from_name(cls, name: str):
        if name in LLM_NAME_TO_CLASS:
            name = name
        elif Path(name).exists():
            with open(name, "r") as f:
                config = yaml.load(f, Loader=yaml.SafeLoader)
            if "llm_name" not in config:
                raise ValueError("llm_name not in config.")
            name = config["llm_name"]
        else:
            raise ValueError(
                f"Invalid name {name}. AutoLLM.from_name needs llm name or llm config as inputs."
            )

        logger.info(f"Load {name} from name.")

        llm_cls = LLM_NAME_TO_CLASS[name]
        return llm_cls
