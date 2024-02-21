# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np

from accelerate.logging import get_logger

logger = get_logger(__name__)


class BaseEval:
    """
    Base class of evaluation.
    The subclasses needs to implement _compute_score and __init__
    """

    def __init__(self):
        self.asrs = []

    def __len__(self):
        return len(self.asrs)

    def add(self, *, prediction=None, reference=None, **kwargs):
        asr = self._compute_score(prediction=prediction, reference=reference)
        self.asrs.append(asr)
        return asr

    def add_batch(self, *, predictions=None, references=None, **kwargs):
        batch_asrs = []
        for pred, ref in zip(predictions, references):
            asr = self._compute_score(prediction=pred, reference=ref)
            batch_asrs.append(asr)

        self.asrs.extend(batch_asrs)

        return batch_asrs

    def compute(self):
        asrs = [asr for asr in self.asrs if asr != -1]
        num_failed_sample = len(self.asrs) - len(asrs)

        if num_failed_sample > 0:
            logger.info(f"{num_failed_sample} samples failed.")

        return np.mean(asrs)
