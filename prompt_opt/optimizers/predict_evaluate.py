from loguru import logger
import numpy as np
import wandb

from.population import Population
from ..utils import *

class PredictEvaluateAndLogCandidate:
    def __init__(self, cfg, population: Population, dataset_loader, predictors, splits):
        self.cfg_optimizer = cfg
        self.population = population
        self.dataset_loader = dataset_loader
        self.predictors = predictors
        self.splits = splits
        self.load_ops()
        self.best_scores = {}
        self.log_step = 1
    
    
    def load_ops(self):
        self.predict_op = get_class_instance_by_config(
            self.cfg_optimizer["ops"]["predict_op"],
            dataset_loader=self.dataset_loader,
            predictors=self.predictors)
        self.score_ops = [get_class_instance_by_config(score_op_cfg)
                          for score_op_cfg in self.cfg_optimizer["ops"]["score_ops"]]
        
    
    def predict_candidate_json(
        self,
        candidate,
        split,
        examples=None,
    ):
        samples = self.dataset_loader.get_data()[split]
        eval_key = split
        generated_prompt = candidate2prompt_md(candidate)
        if self.population.is_resume and eval_key in candidate:
            logger.debug(f"{len(candidate[eval_key])} already prepared, will skip....")
        else:
            candidate[eval_key] = []
        idx = 0
        for sample in samples:
            # check format if skipping
            if len(candidate[eval_key]) >= (idx + 1):
                entry = candidate[eval_key][idx]
                if entry["query"] != sample["query"]:
                    logger.error("ENTRY")
                    logger.error(entry["query"])
                    logger.error("SAMPLE")
                    logger.error(sample["query"])
                    assert False
                assert entry["gold"] == sample["answer"]
                idx += 1
                logger.debug(f'skipped {idx}/{len(samples)}')
                continue

            response, messages = self.predict_op.predict(
                prompt=generated_prompt, 
                query=sample["query"], 
                examples=examples)

            response.update(
                {
                    "gold": sample["answer"],
                    "query": sample["query"],
                    "messages": messages,
                }
            )
            candidate[eval_key].append(response)
            idx += 1
            self.population.save() # TODO make save incremental (this save the whole archive all over again)
            logger.info(f'done {idx}/{len(samples)}')
        
            
    def evaluate_candidate_json(self, candidate, split):
        evals = candidate[split]
        for e in evals:
            gold = e["gold"]
            pred = e["pred"]
            
            e["eval"] = {}
            for score_op in self.score_ops:
                scoring = score_op.score_sample(gold, pred) # score and reasoning
                score_key = score_op.get_score_key()
                assert score_key not in e["eval"], f'duplicate score_op key "{score_key}"'
                e["eval"][score_key] = scoring
        self.population.save() #
            
            
    def log_candidate(self, candidate, split, best_scores={}):
        # tracks and logs best scores so far
        # updates `best_scores`
        score_keys = [score_op.get_score_key() for score_op in self.score_ops]
        log_recs = {}
        
        for score_key in score_keys:
            scores = []
            for sample in candidate[split]:
                scores.append(sample["eval"][score_key]["score"])
            score = np.mean(scores) # mean over samples
            full_key = f"{split}_{score_key}"
            best_scores[full_key] = best_scores.get(full_key, 0.0)
            if score > best_scores[full_key]:
                logger.info(f"improved {full_key} to {score:.3f}")
                best_scores[full_key] = score
            log_recs[f"{split}_{score_key}"] = best_scores[full_key]
                
        if wandb.run is not None:
            wandb.log(log_recs, step=self.log_step, commit=False)
        

    def predict_evaluate_log(self, candidate):
        for split in self.splits:
            logger.info(f'making predictions for "{split}"...')
            self.predict_candidate_json(candidate, split=split, examples=None)
            
            logger.info(f'evaluating candidate predictions for "{split}"...')
            self.evaluate_candidate_json(candidate, split=split)
            
            self.log_candidate(candidate, split=split, best_scores=self.best_scores)
        
        if wandb.run is not None:
            wandb.log({}, step=self.log_step, commit=True)
        self.log_step += 1
        
        
def get_candidate_score(candidate, split, score_key):
    logger.debug(f"canidate id={candidate['id']}({candidate.get('parent_id')}), {candidate.keys()}")
    sample_scores = [sample["eval"][score_key]['score'] for sample in candidate[split]]
    return np.mean(sample_scores)


def rank_candidates(candidates, split, score_key):
    candidate_scores = [get_candidate_score(candidate, split, score_key) for candidate in candidates]
    return np.argsort(candidate_scores, kind="stable")[::-1]