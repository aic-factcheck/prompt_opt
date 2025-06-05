from copy import deepcopy
from typing import Any
from loguru import logger
import numpy as np
from pathlib import Path
import time

from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl, process_to_jsonl

from ..agents.agent_chat import AgentChat
from ..optimizers.population import Population
from ..optimizers.split_tracker import SplitTracker
from ..utils import *


class TRRInitAllExamplesJSON:
    def __init__(self, cfg, exp_path, population: Population, output_schema, split_tracker: SplitTracker, predictors, rng: np.random.RandomState):
        logger.info("loading TRRInitAllExamplesJSON...")
        self.cfg_op = cfg
        self.population = population
        self.predictor = predictors[self.cfg_op["model"]]
        self.split_tracker = split_tracker
        trn_source = self.split_tracker.get_source("trn")
        self.trn_size = cfg.get("trn_size", len(trn_source))
        assert self.trn_size <= len(trn_source), (self.trn_size, len(trn_source))
        self.output_schema = output_schema
        self.rng = rng
        
        self.system_content = self.predictor.render_template(self.cfg_op.get("template_system_content"))
        self.template_init_all_examples = self.predictor.get_template(self.cfg_op["template_init_all_examples"])
        self.template_generate_prompt = self.predictor.get_template(self.cfg_op["template_generate_prompt"])
        

    def _generate_initial_helper(self):
        st = time.time()
        split2indices = self.split_tracker.sample_indices("trn", self.trn_size)
        trn_data = self.split_tracker.get_samples("trn", split2indices)
        assert 0 < self.trn_size <= len(trn_data), f"trn_size = {self.trn_size}, len(self.trn_data) = {len(trn_data)}"
        trn_data = deepcopy(trn_data)
        
        trn_data = self.rng.permutation(trn_data)[:self.trn_size]
        schema_str = jformat(self.output_schema)
        
        agent = AgentChat(self.predictor, self.system_content)
        
        batch = [{"query": e["query"], "answer": jformat(e["answer"])} for e in trn_data]
        examples_prompt = self.template_init_all_examples.render(dataset=batch, schema=schema_str)
        
        response = agent.query(prompt=examples_prompt, frequency_penalty=0.05, seed=self.rng.randint(int(1e10)))
        prompt_generate_prompt = self.template_generate_prompt.render(schema=schema_str)
        response = agent.query(prompt=prompt_generate_prompt, frequency_penalty=0.05, seed=self.rng.randint(int(1e10)))
        
        candidate = {"messages": agent.history(), "split2indices": split2indices, "duration": time.time()-st}
        return candidate
          
          
    def generate_initial(self):
        return self.population.add_candidate(lambda: self._generate_initial_helper())
    
    
class DSeekInitAllExamplesJSON:
    def __init__(self, cfg, exp_path, population: Population, output_schema, split_tracker: SplitTracker, predictors, rng: np.random.RandomState):
        logger.info("loading DSeekInitAllExamplesJSON...")
        self.cfg_op = cfg
        self.population = population
        self.predictor = predictors[self.cfg_op["model"]]
        self.split_tracker = split_tracker
        trn_source = self.split_tracker.get_source("trn")
        self.trn_size = cfg.get("trn_size", len(trn_source))
        assert self.trn_size <= len(trn_source), (self.trn_size, len(trn_source))
        self.output_schema = output_schema
        self.rng = rng
        
        self.system_content = self.predictor.render_template(self.cfg_op.get("template_system_content"))
        self.template_init_using_all_examples = self.predictor.get_template(self.cfg_op["template_init_using_all_examples"])
   

    def _generate_initial_helper(self):
        st = time.time()
        split2indices = self.split_tracker.sample_indices("trn", self.trn_size)
        trn_data = self.split_tracker.get_samples("trn", split2indices)
        assert 0 < self.trn_size <= len(trn_data), f"trn_size = {self.trn_size}, len(self.trn_data) = {len(trn_data)}"
        trn_data = deepcopy(trn_data)
        
        trn_data = self.rng.permutation(trn_data)[:self.trn_size]
        schema_str = jformat(self.output_schema)
        
        agent = AgentChat(self.predictor, self.system_content)
        
        batch = [{"query": e["query"], "answer": jformat(e["answer"])} for e in trn_data]
        examples_prompt = self.template_init_using_all_examples.render(dataset=batch, schema=schema_str)

        response = agent.query(prompt=examples_prompt, frequency_penalty=0.05, seed=self.rng.randint(int(1e10)))
        ld(pf(agent.history))
        
        candidate = {"messages": agent.history(), "split2indices": split2indices, "duration": time.time()-st}
        return candidate
          
          
    def generate_initial(self):
        return self.population.add_candidate(lambda: self._generate_initial_helper())
    
    
class InitExisting:
    def __init__(self, cfg, exp_path, population: Population, output_schema, split_tracker: SplitTracker, predictors, rng: np.random.RandomState):
        li("loading InitExisting...")
        self.cfg_op = cfg
        self.population = population
        self.split_tracker = split_tracker
        # trn_source = self.split_tracker.get_source("trn")
        # self.trn_size = cfg.get("trn_size", len(trn_source))
        self.rng = rng
        self.source_cfgs = []
        for scfg in self.cfg_op["sources"]:
            count = scfg.get("count", 1)
            self.source_cfgs += [scfg] * count
        ld(f"#initial candidates: {len(self.source_cfgs)}, #unique: {len(self.cfg_op['sources'])}")
        self.current_idx = 0
        
    def _extract_best_candidate(self, archive_dir, split, metric) -> Any:
        archive_path = Path(archive_dir, "archive.jsonl")
        archive = read_jsonl(archive_path)
        ld(f'extracting best from: "{str(archive_path)}" #candidates: {len(archive)}')
        best_score = 0
        best_candidate = None
        best_idx = -1
        for idx, candidate in enumerate(archive):
            if "split" not in candidate or split not in candidate["split"]:
                continue
            try:
                scores = [sample["eval"][metric]['score'] for sample in candidate["split"][split]]
            except:
                ld("  skipping incomplete candidate")
                continue
            # pprint(scores)
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_candidate = candidate
                best_idx = idx
        
        ld(f"  best score: {best_score} for candidate idx: {best_idx}")
        return best_candidate


    def _generate_initial_helper(self):
        st = time.time()
        cfg = self.source_cfgs[self.current_idx]
        assert cfg["type"] in ["archive"], cfg["type"]
        if cfg["type"] == "archive":
            # assert cfg["format"] in ["dseek", "md"], cfg["format"]
            # candidate2prompt = {"dseek": candidate2prompt_dseek, "md": candidate2prompt_md}
            # TODO messages are just copied for now, if mixing from multiple sources a conversion to different formats will be needed
            
            assert cfg["select"] in ["best"], cfg["select"]
            if cfg["select"] == "best":
                best_candidate = self._extract_best_candidate(cfg["source"], cfg["split"], cfg["metric"])
                messages = best_candidate["messages"]
                split2indices = best_candidate["split2indices"]
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
            # split2indices = self.split_tracker.sample_indices("trn", self.trn_size)
        
        candidate = {"messages": messages, "split2indices": split2indices, "duration": time.time()-st}
        return candidate
          
          
    def generate_initial(self):
        if not self.current_idx < len(self.source_cfgs):
            raise IndexError(f"getting an existing candidate {self.current_idx+1}")
        ret = self.population.add_candidate(lambda: self._generate_initial_helper())
        self.current_idx += 1
        return ret