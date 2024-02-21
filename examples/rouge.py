# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from rouge_score import rouge_scorer, scoring
import evaluate


class Tokenizer:
    """Helper class to wrap a callable into a class with a `tokenize` method as used by rouge-score."""

    def __init__(self, tokenizer_func):
        self.tokenizer_func = tokenizer_func

    def tokenize(self, text):
        return self.tokenizer_func(text)


class RougeRecall(evaluate.Metric):
    def _compute(
        self,
        predictions,
        references,
        rouge_types=None,
        use_aggregator=True,
        use_stemmer=False,
        tokenizer=None,
    ):
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

        multi_ref = isinstance(references[0], list)

        if tokenizer is not None:
            tokenizer = Tokenizer(tokenizer)

        scorer = rouge_scorer.RougeScorer(
            rouge_types=rouge_types, use_stemmer=use_stemmer, tokenizer=tokenizer
        )
        if use_aggregator:
            aggregator = scoring.BootstrapAggregator()
        else:
            scores = []

        for ref, pred in zip(references, predictions):
            if multi_ref:
                score = scorer.score_multi(ref, pred)
            else:
                score = scorer.score(ref, pred)
            if use_aggregator:
                aggregator.add_scores(score)
            else:
                scores.append(score)

        if use_aggregator:
            result = aggregator.aggregate()
            for key in result:
                result[key] = result[key].mid.recall

        else:
            result = {}
            for key in scores[0]:
                result[key] = list(score[key].recall for score in scores)

        return result
