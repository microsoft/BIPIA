from typing import List, Dict, Any, Tuple

import json
import jsonlines
from collections import defaultdict
import pandas as pd

from .utils import insert_end, insert_start, insert_middle

__all__ = ["BasePIABuilder", "QAPIABuilder"]


class BasePIABuilder:
    name: str
    system_prompt_template: str
    user_prompt_template: str

    def __init__(self, seed: int = None):
        self.seed = seed

    def __call__(self, contexts: str, attacks: str, insert_fns=[insert_end, insert_start, insert_middle], insert_fn_names=["end", "start", "middle"]) -> pd.DataFrame:
        self.context_samples = self.load_context(contexts)
        self.attacks = self.load_attack(attacks)

        return self.construct_samples(insert_fns=insert_fns, insert_fn_names=insert_fn_names)

    def construct_samples(self) -> pd.DataFrame:
        raise NotImplementedError

    def construct_prompt(self, example: Any) -> Tuple[str, str]:
        raise NotImplementedError

    def construct_response(self, example: Any) -> str:
        raise NotImplementedError

    def load_context(self, contexts: str|List) -> List:
        if isinstance(contexts, list):
            return contexts

        with jsonlines.open(contexts, "r") as reader:
            context_samples = list(reader.iter())

        return context_samples

    def load_attack(self, attacks: str|dict) -> Dict:
        if isinstance(attacks, dict):
            return attacks

        with open(attacks, "r") as f:
            attacks = json.load(f)
        return attacks


class QAPIABuilder(BasePIABuilder):
    def construct_samples(self, insert_fns=[insert_end, insert_start, insert_middle], insert_fn_names=["end", "start", "middle"]) -> pd.DataFrame:
        samples = defaultdict(list)

        for insert_fn_name, insert_fn in zip(
            insert_fn_names, insert_fns
        ):
            for normal_sample in self.context_samples:
                context = normal_sample["context"]
                question = normal_sample["question"]
                ideal = normal_sample["ideal"]

                for attack_name in self.attacks:
                    attack_str = self.attacks[attack_name]

                    poisoned_context = insert_fn(
                        context, attack_str, random_state=self.seed
                    )
                    samples["context"].append(poisoned_context)
                    samples["attack_name"].append(attack_name)
                    samples["task_name"].append(self.name)
                    samples["ideal"].append(ideal)
                    samples["question"].append(question)
                    samples["position"].append(insert_fn_name)

        return pd.DataFrame.from_dict(samples)
