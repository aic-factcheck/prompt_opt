from loguru import logger
import numpy as np
import wandb

from .population import Population
from ..optimizers.split_tracker import SplitTracker
from ..utils import *


class PredictEvaluateAndLogCandidate:
    def __init__(
        self,
        cfg,
        exp_path,
        population: Population,
        output_schema,
        predictors,
        split_tracker: SplitTracker,
        splits,
        candidate2prompt,
    ):
        self.cfg_optimizer = cfg
        self.exp_path = exp_path
        self.population = population
        self.output_schema = output_schema
        self.predictors = predictors
        self.split_tracker = split_tracker
        self.splits = splits
        self.candidate2prompt = candidate2prompt
        self.load_ops()
        self.best_scores = {}
        self.log_step = 1

    def load_ops(self):
        self.predict_op = get_class_instance_by_config(
            self.cfg_optimizer["ops"]["predict_op"], exp_path=self.exp_path, predictors=self.predictors
        )
        self.score_ops = [
            get_class_instance_by_config(score_op_cfg, exp_path=self.exp_path, predictors=self.predictors)
            for score_op_cfg in self.cfg_optimizer["ops"]["score_ops"]
        ]

    def predict_candidate_json(
        self,
        candidate,
        split,
        examples=None,
    ):
        samples = self.split_tracker.get_samples(split, candidate["split2indices"])
        generated_prompt = self.candidate2prompt(candidate)
        if "split" not in candidate:
            candidate["split"] = {}
        if self.population.is_resume() and split in candidate["split"]:
            ld(f"{len(candidate['split'][split])} already prepared, will skip....")
        else:
            candidate["split"][split] = []
        idx = 0
        for sample in samples:
            # check format if skipping
            if len(candidate["split"][split]) >= (idx + 1):
                entry = candidate["split"][split][idx]
                # sanity check for resume: at least check if queries for the already computed samples match
                if entry["query"] != sample["query"]:
                    le("ENTRY")
                    le(entry["query"])
                    le("SAMPLE")
                    le(sample["query"])
                    assert False
                assert entry["gold"] == sample["answer"]
                idx += 1
                ld(f"skipped {idx}/{len(samples)}")
                continue

            response, messages = self.predict_op.predict(
                prompt=generated_prompt, query=sample["query"], output_schema=self.output_schema, examples=examples
            )

            response.update(
                {
                    "gold": sample["answer"],
                    "query": sample["query"],
                    "messages": messages,
                }
            )
            candidate["split"][split].append(response)
            idx += 1
            self.population.save()  # TODO make save incremental (this saves the whole archive all over again)
            li(f"done {idx}/{len(samples)}")

    def evaluate_candidate_json(self, candidate, split):
        evals = candidate["split"][split]
        for eidx, e in enumerate(evals):
            gold = e["gold"]
            pred = e["pred"]

            if self.population.is_resume() and "eval" in e:
                ld(f"eval present for {eidx+1}/{len(evals)}")
            else:
                e["eval"] = {}

            for score_op in self.score_ops:
                score_key = score_op.get_score_key()
                if self.population.is_resume() and score_key in e["eval"]:
                    ld(f"already evaluated for score_key={score_key} skipping...")
                    continue
                scoring = score_op.score_sample(gold, pred)  # score and reasoning
                assert score_key not in e["eval"], f'duplicate score_op key "{score_key}"'
                e["eval"][score_key] = scoring
        self.population.save()  #

    def log_candidate(self, candidate, split, best_scores={}):
        # tracks and logs best scores so far
        # updates `best_scores`
        score_keys = [score_op.get_score_key() for score_op in self.score_ops]
        log_recs = {}

        for score_key in score_keys:
            scores = []
            for sample in candidate["split"][split]:
                scores.append(sample["eval"][score_key]["score"])
            score = np.mean(scores)  # mean over samples
            full_key = f"{split}_{score_key}"
            best_scores[full_key] = best_scores.get(full_key, 0.0)
            if score > best_scores[full_key]:
                li(f"improved {full_key} to {score:.3f}")
                best_scores[full_key] = score
            log_recs[f"{split}_{score_key}"] = best_scores[full_key]

        if wandb.run is not None:
            wandb.log(log_recs, step=self.log_step, commit=False)

    def predict_evaluate_log(self, candidate):
        for split in self.splits:
            li(f'making predictions for "{split}"...')
            self.predict_candidate_json(candidate, split=split, examples=None)

            li(f'evaluating candidate predictions for "{split}"...')
            self.evaluate_candidate_json(candidate, split=split)

            self.log_candidate(candidate, split=split, best_scores=self.best_scores)

        if wandb.run is not None:
            wandb.log({}, step=self.log_step, commit=True)
        self.log_step += 1


def get_candidate_score(candidate, split, score_key):
    # ld(f"canidate id={candidate['id']}({candidate.get('parent_id')}), {candidate.keys()}")
    sample_scores = [sample["eval"][score_key]["score"] for sample in candidate["split"][split]]
    return np.mean(sample_scores)


def get_candidate_score_source(candidate, source_split, score_key):
    target_splits = []
    for tgt_split, tgt_cfg in candidate["split2indices"].items():
        if tgt_cfg["source"] == source_split:
            target_splits.append(tgt_split)

    sample_scores = []
    for split in target_splits:
        sample_scores += [sample["eval"][score_key]["score"] for sample in candidate["split"][split]]
    # ld(f"# sample scores: {len(sample_scores)}")
    return np.mean(sample_scores)


def rank_candidates(candidates, split, score_key):
    candidate_scores = [get_candidate_score(candidate, split, score_key) for candidate in candidates]
    return np.argsort(candidate_scores, kind="stable")[::-1]
