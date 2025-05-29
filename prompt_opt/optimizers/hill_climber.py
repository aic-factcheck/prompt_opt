from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from loguru import logger
import numpy as np
import wandb

from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl, process_to_jsonl


from .population import Population
from ..agents.agent_chat import AgentChat
from ..optimizers.abstract_optimizer import AbstractObtimizer
from ..optimizers.predict_evaluate import PredictEvaluateAndLogCandidate, rank_candidates
from ..optimizers.split_tracker import SplitTracker
from ..utils import *


class HillClimber(AbstractObtimizer):
    def __init__(self, cfg, exp_path, dataset_loader, predictors, rng: np.random.RandomState, resume:bool=False):
        li("loading HillClimber...")
        super().__init__(cfg, exp_path, dataset_loader, predictors, rng, resume)
        
        
    def _load_ops(self):
        super()._load_ops()


    def optimize(self):
        li("starting optimization...")
        
        n_initial = self.cfg_optimizer["n_initial"]
        top_k = self.cfg_optimizer["top_k"]
        n_neighbors = self.cfg_optimizer["n_neighbors"]
        n_iters = self.cfg_optimizer["n_iters"]
        score_key = self.cfg_optimizer["score_key"]
        select_split = self.cfg_optimizer["select_split"]
        
        
        logger.info(f"generating {n_initial} initial candidates...")
        for idx in range(n_initial):
            op_result = self.init_op.generate_initial()
            candidate = op_result["candidate"]
            if op_result["skipped"]:
                li(f"skipped candidate id={candidate['id']} {idx+1}/{n_initial}")
            else:
                li(f"generated candidate id={candidate['id']} {idx+1}/{n_initial}")
            self.pel.predict_evaluate_log(candidate)
            
        parents = self.population.get_candidates() # takes the whole initial population
        for it in range(n_iters):
            li(f"iteration {it+1}")
            ld(f"parent ids={[p['id'] for p in parents]}")
            
            rank_idxs = rank_candidates(parents, split=select_split, score_key=score_key)
            assert top_k <= len(rank_idxs), (top_k, len(rank_idxs))
            
            selected_parents = [parents[idx] for idx in rank_idxs[:top_k]]
            parents = selected_parents[:]
            for parent_idx, parent in enumerate(selected_parents):
                if self.xval_permute:
                    parent = self.split_tracker.resample_splits_for_candidate(parent)
                ld(f'final split2indices: {parent["split2indices"]}')
                    
                li(f"selected parent {parent_idx+1}/{top_k} id={parent['id']}")
                op_results = self.mutate_op.mutate(parent, 
                                                  n_neighbors=n_neighbors)
                for neighbor_idx, op_result in enumerate(op_results):
                    neighbor = op_result["candidate"]
                    if op_result["skipped"]:
                        li(f"mutation skipped for candidate id={neighbor['id']} {neighbor_idx+1}/{n_neighbors}")
                    else:
                        li(f"mutated candidate id={neighbor['id']} {neighbor_idx+1}/{n_neighbors}")
                    self.pel.predict_evaluate_log(neighbor)
                    parents.append(neighbor)
                
        li(f'archive saved to: {self.archive_jsonl}')
        return self.population.get_candidates()

        