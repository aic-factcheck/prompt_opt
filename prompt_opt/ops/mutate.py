from copy import deepcopy
from pathlib import Path

from loguru import logger
import numpy as np

from ..agents.agent_chat import AgentChat
from ..agents.agent_json import AgentJSONSteppedCoT
from ..optimizers.population import Population
from ..utils import *


class TRRMutateJSON:
    def __init__(self, cfg, population: Population, dataset_loader, predictors):
        logger.info("loading TRRMutateJSON...")
        self.cfg_op = cfg
        self.population = population
        self.predictor = predictors[self.cfg_op["model"]]
        self.output_schema = dataset_loader.get_output_schema()
        
        self.system_content = self.predictor.get_template(self.cfg_op["template_system_content"]).render()
        self.template_mutate_first = self.predictor.get_template(self.cfg_op["template_mutate_first"])
        self.template_mutate_next = self.predictor.get_template(self.cfg_op["template_mutate_next"])
        
        
    def _mutate_helper(self, candidate, n_neighbors: int, rng: np.random.RandomState):
        original_prompt = candidate2prompt_md(candidate)
        schema_str = jformat(self.output_schema)
        
        agent = AgentChat(self.predictor, self.system_content)
        
        neighbors = []
        first = True
        for neighbor_idx in range(n_neighbors):
            logger.info(f"generating neighbor {neighbor_idx+1}/{n_neighbors}")
            if first:
                prompt_mutate = self.template_mutate_first.render(prompt=original_prompt, schema=schema_str)
                first = False
            else:
                prompt_mutate = self.template_mutate_next.render()
                
            response = agent.query(prompt=prompt_mutate, temperature=0.3, frequency_penalty=0.05, seed=rng.randint(1e10))
            neighbor = {"messages": agent.history(), "parent_id": candidate["id"]}
            neighbors.append(neighbor)
            
        return neighbors
    
    def mutate(self, candidate, n_neighbors: int, rng: np.random.RandomState):
        return self.population.add_multiple_candidates(lambda: self._mutate_helper(candidate, n_neighbors, rng), n=n_neighbors)
    
    
class TRRImproveJSON:
    def __init__(self, cfg, population: Population, dataset_loader, predictors):
        logger.info("loading TRRImproveJSON...")
        self.cfg_op = cfg
        self.population = population
        self.predictor = predictors[self.cfg_op["model"]]
        self.output_schema = dataset_loader.get_output_schema()
        
        self.select_split = self.cfg_op["select_split"]
        self.score_key = self.cfg_op["score_key"]
        self.max_error_samples = self.cfg_op["max_error_samples"]
        
        self.system_content = self.predictor.get_template(self.cfg_op["template_system_content"]).render()
        self.template_improve_first_sample = self.predictor.get_template(self.cfg_op["template_improve_first_sample"])
        self.template_improve_next_sample = self.predictor.get_template(self.cfg_op["template_improve_next_sample"])
        self.template_suggest_changes_for_sample = self.predictor.get_template(self.cfg_op["template_suggest_changes_for_sample"])
        self.template_generate_improved_prompt = self.predictor.get_template(self.cfg_op["template_generate_improved_prompt"])
        
        
    def _mutate_helper(self, candidate, n_neighbors: int, rng: np.random.RandomState):
        original_prompt = candidate2prompt_md(candidate)
        schema_str = jformat(self.output_schema)
        
        agent = AgentChat(self.predictor, self.system_content)
        
        error_samples = [sample for sample in candidate[self.select_split] if sample["eval"][self.score_key]["score"] < 1.0]
        if len(error_samples) == 0:
            logger.warning("all samples correct, no way to improve...")
            return [deepcopy(candidate) for _ in range(n_neighbors)]
        
        neighbors = []
        for neighbor_idx in range(n_neighbors):
            logger.info(f"generating neighbor {neighbor_idx+1}/{n_neighbors}")
            sel_samples = rng.permutation(error_samples[:])[:self.max_error_samples]
            
            first = True
            for sample in sel_samples:
                query = sample["query"]
                # raw_pred = sample["messages"][-1]["content"]
                # score = sample["eval"][self.score_key]["score"]
                pred = sample["pred"]
                gold = sample["gold"]
                pred_str = jformat(pred)
                gold_str = jformat(gold)
                
                if first:
                    prompt_improve_sample = self.template_improve_first_sample.render(prompt=original_prompt, query=query, schema=schema_str)
                    first = False
                else:
                    prompt_improve_sample = self.template_improve_next_sample.render(query=query)
                    
                agent.add_user(prompt_improve_sample)
                agent.add_assistant(pred_str)
                
                prompt_suggest_changes_for_sample = self.template_suggest_changes_for_sample.render(gold=gold_str, pred=pred_str)
                response = agent.query(prompt=prompt_suggest_changes_for_sample, temperature=0.3, frequency_penalty=0.05, seed=rng.randint(1e10))
                
            prompt_generate_improved_prompt = self.template_generate_improved_prompt.render()
            response = agent.query(prompt=prompt_generate_improved_prompt, temperature=0.3, frequency_penalty=0.05, seed=rng.randint(1e10))
            
            neighbor = {"messages": agent.history(), "parent_id": candidate["id"]}
            neighbors.append(neighbor)
        
        return neighbors
    
    def mutate(self, candidate, n_neighbors: int, rng: np.random.RandomState):
        return self.population.add_multiple_candidates(lambda: self._mutate_helper(candidate, n_neighbors, rng), n=n_neighbors)