# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import OrderedDict
from typing import List

import numpy as np

from .regist import depia_regist_fn

__all__ = ["BipiaEvalFactory"]

attack2eval = OrderedDict()


class BipiaEvalFactory:
    def __init__(
        self,
        *,
        gpt_config: str | dict,
        regist_fn=depia_regist_fn,
        activate_attacks,
        **kwargs,
    ):
        self.attack2eval = regist_fn(gpt_config, **kwargs)

        self.activated_evals = OrderedDict()
        for attack in activate_attacks:
            self.activated_evals[attack] = self.attack2eval[attack]()

    def __len__(self) -> int:
        num_examples = 0
        for attack in self.activated_evals:
            eval_fn = self.activated_evals[attack]
            num_examples += len(eval_fn)
        return num_examples

    def add(
        self,
        *,
        reference: str = None,
        prediction: str = None,
        attack: str = None,
        task: str = None,
        **kwargs,
    ):
        eval_fn = self.activated_evals[attack]
        asr = eval_fn.add(
            reference=reference,
            prediction=prediction,
            attack=attack,
            task=task,
            **kwargs,
        )
        return asr

    def add_batch(
        self,
        *,
        references: List = None,
        predictions: List = None,
        attacks: List = None,
        tasks: List = None,
        **kwargs,
    ):
        asrs = []
        for ref, pred, attack, task in zip(references, predictions, attacks, tasks):
            eval_fn = self.activated_evals[attack]
            asr = eval_fn.add(
                reference=ref, prediction=pred, attack=attack, task=task, **kwargs
            )
            asrs.append(asr)
        return asrs

    def compute(self):
        asr_report = OrderedDict()
        for attack in self.activated_evals:
            eval_fn = self.activated_evals[attack]
            asr_report[attack] = eval_fn.compute()

        macro_asr = np.average(list(asr_report.values()))
        micro_asr = np.average(
            list(asr_report.values()),
            weights=[len(self.activated_evals[attack]) for attack in asr_report],
        )

        asr_report["macro"] = macro_asr
        asr_report["micro"] = micro_asr

        return asr_report
