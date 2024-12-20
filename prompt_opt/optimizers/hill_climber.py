from copy import deepcopy
from pathlib import Path

from loguru import logger
import numpy as np
import wandb

from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl, process_to_jsonl

from .population import Population
from ..agents.agent_chat import AgentChat
from ..optimizers.predict_evaluate import PredictEvaluateAndLogCandidate, rank_candidates
from ..utils import *


class HillClimber:
    def __init__(self, cfg, exp_path, dataset_loader, predictors, resume:bool=False):
        logger.info("loading HillClimber...")
        self.exp_path = exp_path
        self.cfg_optimizer = cfg
        self.dataset_loader = dataset_loader
        self.predictors = predictors
        self.resume = resume
        self.archive_jsonl = Path(self.exp_path, "archive.jsonl")
        self.population = Population(archive_jsonl=self.archive_jsonl, resume=resume)
        
        self.load_ops()
        eval_splits = self.cfg_optimizer["eval_splits"]
        self.pel = PredictEvaluateAndLogCandidate(cfg=self.cfg_optimizer,
                                                  population = self.population,
                                                  dataset_loader=self.dataset_loader,
                                                  predictors=self.predictors, 
                                                  splits=eval_splits)
        
        
    def load_ops(self):
        self.init_op = get_class_instance_by_config(
            self.cfg_optimizer["ops"]["init_op"],
            population = self.population,
            dataset_loader=self.dataset_loader,
            predictors=self.predictors)
        
        self.mutate_op = get_class_instance_by_config(
            self.cfg_optimizer["ops"]["mutate_op"],
            population = self.population,
            dataset_loader=self.dataset_loader,
            predictors=self.predictors)


    def optimize(self, rng: np.random.RandomState):
        logger.info("starting optimization...")
        
        n_initial = self.cfg_optimizer["n_initial"]
        top_k = self.cfg_optimizer["top_k"]
        n_neighbors = self.cfg_optimizer["n_neighbors"]
        n_iters = self.cfg_optimizer["n_iters"]
        score_key = self.cfg_optimizer["score_key"]
        select_split = self.cfg_optimizer["select_split"]
        
        logger.info(f"generating {n_initial} initial candidates...")
        for idx in range(n_initial):
            op_result = self.init_op.generate_initial(rng)
            candidate = op_result["candidate"]
            if op_result["skipped"]:
                logger.debug(f"skipped candidate id={candidate['id']} {idx+1}/{n_initial}")
            else:
                logger.info(f"generated candidate id={candidate['id']} {idx+1}/{n_initial}")
            self.pel.predict_evaluate_log(candidate)
            
        parents = self.population.get_candidates()
        for it in range(n_iters):
            logger.info(f"iteration {it+1}")
            logger.debug(f"parent ids={[p['id'] for p in parents]}")
            rank_idxs = rank_candidates(parents, split=select_split, score_key=score_key)
            assert top_k <= len(rank_idxs), (top_k, len(rank_idxs))
            selected_parents = [parents[idx] for idx in rank_idxs[:top_k]]
            parents = selected_parents[:]
            for parent_idx, parent in enumerate(selected_parents):
                logger.info(f"selected parent {parent_idx+1}/{top_k} id={parent['id']}")
                op_results = self.mutate_op.mutate(parent, 
                                                  n_neighbors=n_neighbors, 
                                                  rng=rng)
                for neighbor_idx, op_result in enumerate(op_results):
                    neighbor = op_result["candidate"]
                    if op_result["skipped"]:
                        logger.debug(f"mutation skipped for candidate id={neighbor['id']} {neighbor_idx+1}/{n_neighbors}")
                    else:
                        logger.info(f"mutated candidate id={neighbor['id']} {neighbor_idx+1}/{n_neighbors}")
                    self.pel.predict_evaluate_log(neighbor)
                    parents.append(neighbor)
                
        logger.info(f'archive saved to: {self.archive_jsonl}')
        return self.population.get_candidates()

        