from collections import defaultdict
from pathlib import Path

import numpy as np
import wandb

from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl, process_to_jsonl

from ..optimizers.abstract_optimizer import AbstractObtimizer
from ..utils import *


class EvolutionaryAlgorithm(AbstractObtimizer):
    def __init__(self, cfg, exp_path, dataset_loader, predictors, rng: np.random.RandomState, resume:bool=False):
        li("loading EvolutionaryAlgorithm...")
        self.n_initial = cfg["n_initial"]
        self.n_iters = cfg["n_iters"]
        super().__init__(cfg, exp_path, dataset_loader, predictors, rng, resume)
        
        
    def _load_ops(self):
        super()._load_ops()
        self.select_op = get_class_instance_by_config(
            self.cfg_optimizer["ops"]["select_op"],
            exp_path=self.exp_path,
            predictors=self.predictors,
            rng=self.rng)
        
        self.reproduce_op = get_class_instance_by_config(
            self.cfg_optimizer["ops"]["reproduce_op"],
            exp_path=self.exp_path,
            mutate_op=self.mutate_op)
    
        self.reduce_op = get_class_instance_by_config(
            self.cfg_optimizer["ops"]["reduce_op"],
            exp_path=self.exp_path,
            predictors=self.predictors,
            rng=self.rng)
        
        
    def _create_initial_population(self):
        li(f"generating {self.n_initial} initial candidates...")
        for idx in range(self.n_initial):
            op_result = self.init_op.generate_initial()
            candidate = op_result["candidate"]
            if op_result["skipped"]:
                ld(f"skipped candidate id={candidate['id']} {idx+1}/{self.n_initial}")
            else:
                li(f"generated candidate id={candidate['id']} {idx+1}/{self.n_initial}")
            self.pel.predict_evaluate_log(candidate)
    

    def optimize(self):
        li("starting optimization...")
        
        self._create_initial_population()
            
        parents = self.population.get_candidates() # takes the whole initial population
        for it in range(self.n_iters):
            li(f"iteration {it+1}")
            ld(f"parent ids={[p['id'] for p in parents]}")

            li("selecting parents...")
            parents = self.select_op.select(parents, self.n_initial)
            
            offspring = []
            for i, parent in enumerate(parents, start=1):
                if self.xval_permute:
                    parent = self.split_tracker.resample_splits_for_candidate(parent)
                    
                li(f"reproducing parent {i}/{len(parents)}...")
                children = self.reproduce_op.reproduce([parent], n_neighbors=1)
                li(f"evaluating offspring for parent {i}/{len(parents)}...")
                
                for child in children:
                    self.pel.predict_evaluate_log(child)
                    offspring.append(child)
                    
            li("reducing population...")
            parents = self.reduce_op.reduce(parents, offspring)
            
                
        li(f'archive saved to: {self.archive_jsonl}')
        return self.population.get_candidates()

        