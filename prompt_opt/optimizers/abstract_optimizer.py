from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from loguru import logger
import numpy as np
import wandb

from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl, process_to_jsonl

from .population import Population
from ..agents.agent_chat import AgentChat
from ..optimizers.predict_evaluate import PredictEvaluateAndLogCandidate, rank_candidates
from ..optimizers.split_tracker import SplitTracker
from ..utils import *


class AbstractObtimizer:
    def __init__(self, cfg, exp_path, dataset_loader, predictors, rng: np.random.RandomState, resume:bool=False):
        self.exp_path = exp_path
        self.cfg_optimizer = cfg
        self.dataset_loader = dataset_loader
        self.predictors = predictors
        self.resume = resume
        self.archive_jsonl = Path(self.exp_path, "archive.jsonl")
        self.rng = rng
        
        # TODO: make a configurable class?
        self.prompt_format = self.cfg_optimizer.get("prompt_format", "md")
        assert self.prompt_format in ["md", "dseek"], self.prompt_format
        self.candidate2prompt = candidate2prompt_md if self.prompt_format == "md" else candidate2prompt_dseek
        
        self.xval_trn_and_dev = self.cfg_optimizer.get("xval_trn_and_dev", False)
        self.xval_permute = self.cfg_optimizer.get("xval_permute", False)
        
        if self.xval_permute and not self.xval_trn_and_dev:
            assert False, "xval_permute requires xval_trn_and_dev"
            
        self.split_tracker = SplitTracker(self.dataset_loader, self.xval_trn_and_dev, rng)
        
        self.population = Population(archive_jsonl=self.archive_jsonl, split_tracker=self.split_tracker, resume=resume)
        
        eval_splits = self.cfg_optimizer["eval_splits"]
        self.pel = PredictEvaluateAndLogCandidate(cfg=self.cfg_optimizer,
                                                  exp_path=self.exp_path,
                                                  population = self.population,
                                                  output_schema=self.dataset_loader.get_output_schema(),
                                                  predictors=self.predictors,
                                                  split_tracker=self.split_tracker,
                                                  splits=eval_splits,
                                                  candidate2prompt=self.candidate2prompt)
        self._load_ops()
        
        
    def _load_ops(self):
        self.init_op = get_class_instance_by_config(
            self.cfg_optimizer["ops"]["init_op"],
            exp_path=self.exp_path,
            population = self.population,
            output_schema=self.dataset_loader.get_output_schema(),
            split_tracker=self.split_tracker,
            predictors=self.predictors,
            rng=self.rng)
        
        self.mutate_op = get_class_instance_by_config(
            self.cfg_optimizer["ops"]["mutate_op"],
            exp_path=self.exp_path,
            population = self.population,
            output_schema=self.dataset_loader.get_output_schema(),
            predictors=self.predictors,
            split_tracker=self.split_tracker,
            predict_op=self.pel.predict_op,
            rng=self.rng)
        
        
    def optimize(self):
        raise NotImplementedError("NOT IMPLEMENTED in an abstract class...")
        

        