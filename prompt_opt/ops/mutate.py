from copy import deepcopy
from pathlib import Path
import time

from loguru import logger
import numpy as np
from prompt_opt.optimizers.split_tracker import SplitTracker

from ..agents.agent_chat import AgentChat
from ..agents.agent_json import AgentJSONSteppedCoT
from ..optimizers.population import Population
from ..utils import *


class TRRMutateJSON:
    def __init__(self, cfg, exp_path, population: Population, output_schema, predictors, split_tracker: SplitTracker, predict_op, rng: np.random.RandomState):
        logger.info("loading TRRMutateJSON...")
        self.cfg_op = cfg
        self.population = population
        self.predictor = predictors[self.cfg_op["model"]]
        self.output_schema = output_schema
        self.rng = rng
        
        self.system_content = self.predictor.render_template(self.cfg_op.get("template_system_content"))
        self.template_mutate_first = self.predictor.get_template(self.cfg_op["template_mutate_first"])
        self.template_mutate_next = self.predictor.get_template(self.cfg_op["template_mutate_next"])
        
        
    def _mutate_helper(self, candidate, n_neighbors: int):
        original_prompt = candidate2prompt_md(candidate)
        schema_str = jformat(self.output_schema)
        
        agent = AgentChat(self.predictor, self.system_content)
        
        neighbors = []
        first = True
        for neighbor_idx in range(n_neighbors):
            st = time.time()
            logger.info(f"generating neighbor {neighbor_idx+1}/{n_neighbors}")
            if first:
                prompt_mutate = self.template_mutate_first.render(prompt=original_prompt, schema=schema_str)
                first = False
            else:
                prompt_mutate = self.template_mutate_next.render()
                
            response = agent.query(prompt=prompt_mutate, frequency_penalty=0.05, seed=self.rng.randint(int(1e10)))
            neighbor = {"messages": agent.history(), "parent_id": candidate["id"], "duration": time.time()-st}
            neighbors.append(neighbor)
            
        return neighbors
    
    def mutate(self, candidate, n_neighbors: int):
        return self.population.add_multiple_candidates(lambda: self._mutate_helper(candidate, n_neighbors), n=n_neighbors)
    
    
class TRRImproveJSON:
    def __init__(self, cfg, exp_path, population: Population, output_schema, predictors, split_tracker: SplitTracker, predict_op, rng: np.random.RandomState):
        logger.info("loading TRRImproveJSON...")
        self.cfg_op = cfg
        self.population = population
        self.predictor = predictors[self.cfg_op["model"]]
        self.output_schema = output_schema
        self.rng = rng
        
        self.select_split = self.cfg_op["select_split"]
        self.score_key = self.cfg_op["score_key"]
        self.max_error_samples = self.cfg_op["max_error_samples"]
        
        self.system_content = self.predictor.get_template(self.cfg_op["template_system_content"]).render()
        self.template_improve_first_sample = self.predictor.get_template(self.cfg_op["template_improve_first_sample"])
        self.template_improve_next_sample = self.predictor.get_template(self.cfg_op["template_improve_next_sample"])
        self.template_suggest_changes_for_sample = self.predictor.get_template(self.cfg_op["template_suggest_changes_for_sample"])
        self.template_generate_improved_prompt = self.predictor.get_template(self.cfg_op["template_generate_improved_prompt"])
        
        
    def _mutate_helper(self, candidate, n_neighbors: int):
        original_prompt = candidate2prompt_md(candidate)
        schema_str = jformat(self.output_schema)
        
        error_samples = [sample for sample in candidate["split"][self.select_split] if sample["eval"][self.score_key]["score"] < 1.0]
        if len(error_samples) == 0:
            logger.warning("all samples correct, no way to improve...")
            return [deepcopy(candidate) for _ in range(n_neighbors)]
        
        neighbors = []
        for neighbor_idx in range(n_neighbors):
            st = time.time()
            logger.info(f"generating neighbor {neighbor_idx+1}/{n_neighbors}")
            sel_samples = self.rng.permutation(error_samples[:])[:self.max_error_samples]
            
            agent = AgentChat(self.predictor, self.system_content)
            
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
                response = agent.query(prompt=prompt_suggest_changes_for_sample, frequency_penalty=0.05, seed=self.rng.randint(int(1e10)))
                
            prompt_generate_improved_prompt = self.template_generate_improved_prompt.render()
            response = agent.query(prompt=prompt_generate_improved_prompt, frequency_penalty=0.05, seed=self.rng.randint(int(1e10)))
            
            neighbor = {"messages": agent.history(), "parent_id": candidate["id"], "split2indices": candidate["split2indices"], "duration": time.time()-st}
            neighbors.append(neighbor)
        
        return neighbors
    
    def mutate(self, candidate, n_neighbors: int):
        return self.population.add_multiple_candidates(lambda: self._mutate_helper(candidate, n_neighbors), n=n_neighbors)
    
    
class DSeekImproveJSON:
    # based on TRRImproveJSON
    # CHANGES: candidate2prompt_md -> candidate2prompt_dseek
    def __init__(self, cfg, exp_path, population: Population, output_schema, predictors, split_tracker: SplitTracker, predict_op, rng: np.random.RandomState):
        logger.info("loading DSeekImproveJSON...")
        self.cfg_op = cfg
        self.population = population
        self.predictor = predictors[self.cfg_op["model"]]
        self.output_schema = output_schema
        self.rng = rng
        
        self.select_split = self.cfg_op["select_split"]
        self.score_key = self.cfg_op["score_key"]
        self.max_error_samples = self.cfg_op["max_error_samples"]
        
        self.system_content = self.predictor.render_template(self.cfg_op.get("template_system_content"))
        self.template_improve_first_sample = self.predictor.get_template(self.cfg_op["template_improve_first_sample"])
        self.template_improve_next_sample = self.predictor.get_template(self.cfg_op["template_improve_next_sample"])
        self.template_suggest_changes_for_sample = self.predictor.get_template(self.cfg_op["template_suggest_changes_for_sample"])
        self.template_generate_improved_prompt = self.predictor.get_template(self.cfg_op["template_generate_improved_prompt"])
        
        
    def _mutate_helper(self, candidate, n_neighbors: int):
        original_prompt = candidate2prompt_dseek(candidate)
        schema_str = jformat(self.output_schema)
        
        error_samples = [sample for sample in candidate["split"][self.select_split] if sample["eval"][self.score_key]["score"] < 1.0]
        if len(error_samples) == 0:
            logger.warning("all samples correct, no way to improve...")
            return [deepcopy(candidate) for _ in range(n_neighbors)] # TODO mutate somehow
        
        neighbors = []
        for neighbor_idx in range(n_neighbors):
            st = time.time()
            logger.info(f"generating neighbor {neighbor_idx+1}/{n_neighbors}")
            sel_samples = self.rng.permutation(error_samples[:])[:self.max_error_samples]
            
            agent = AgentChat(self.predictor, self.system_content)
            
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
                response = agent.query(prompt=prompt_suggest_changes_for_sample, frequency_penalty=0.05, seed=self.rng.randint(int(1e10)))
                
            prompt_generate_improved_prompt = self.template_generate_improved_prompt.render()
            response = agent.query(prompt=prompt_generate_improved_prompt, frequency_penalty=0.05, seed=self.rng.randint(int(1e10)))
            
            neighbor = {"messages": agent.history(), "parent_id": candidate["id"], "split2indices": candidate["split2indices"], "duration": time.time()-st}
            neighbors.append(neighbor)
        
        return neighbors
    
    def mutate(self, candidate, n_neighbors: int):
        return self.population.add_multiple_candidates(lambda: self._mutate_helper(candidate, n_neighbors), n=n_neighbors)


class DSeekDirectImproveJSON:
    # based on DSeekImproveJSON
    # CHANGES: the error examples are not given as simulated outputs of the current model + they are given in a single "user" block
    def __init__(self, cfg, exp_path, population: Population, output_schema, predictors, split_tracker: SplitTracker, predict_op, rng: np.random.RandomState):
        li("loading DSeekDirectImproveJSON...")
        self.cfg_op = cfg
        self.population = population
        self.predictor = predictors[self.cfg_op["model"]]
        self.output_schema = output_schema
        self.rng = rng
        
        self.select_split = self.cfg_op["select_split"]
        self.score_key = self.cfg_op["score_key"]
        self.max_error_samples = self.cfg_op["max_error_samples"]
        
        self.system_content = self.predictor.render_template(self.cfg_op.get("template_system_content"))
        # self.template_give_error_samples = self.predictor.get_template(self.cfg_op["template_give_error_samples"])
        self.template_give_samples = self.predictor.get_template(self.cfg_op["template_give_samples"])
        
        
    def _mutate_helper(self, candidate, n_neighbors: int):
        original_prompt = candidate2prompt_dseek(candidate)
        schema_str = jformat(self.output_schema)
        
        scores = [sample["eval"][self.score_key]["score"] for sample in candidate["split"][self.select_split]]
        avg_score = np.mean(scores)
        
        error_samples = [sample for sample in candidate["split"][self.select_split] if sample["eval"][self.score_key]["score"] < 1.0]
        if len(error_samples) == 0:
            lw("all samples correct, no way to improve...")
            return [deepcopy(candidate) for _ in range(n_neighbors)] # TODO mutate somehow
        
        neighbors = []
        for neighbor_idx in range(n_neighbors):
            st = time.time()
            li(f"generating neighbor {neighbor_idx+1}/{n_neighbors}")
            sel_samples = self.rng.permutation(error_samples[:])[:self.max_error_samples]
            
            agent = AgentChat(self.predictor, self.system_content)
            
            dataset = []
            for sample in sel_samples:
                query = sample["query"]
                pred = sample["pred"]
                gold = sample["gold"]
                score = sample["eval"][self.score_key]["score"]
                pred_str = jformat(pred)
                gold_str = jformat(gold)
                dataset.append({"query": query, "predictions": pred_str, "gold": gold_str, "score": score})
            
            prompt_give_samples = self.template_give_samples.render(instructions=original_prompt, schema=schema_str, dataset=dataset, avg_score=avg_score)

            response = agent.query(prompt=prompt_give_samples, frequency_penalty=0.05, seed=self.rng.randint(int(1e10)))
            
            neighbor = {"messages": agent.history(), "parent_id": candidate["id"], "split2indices": candidate["split2indices"], "duration": time.time()-st}
            neighbors.append(neighbor)
        
        return neighbors
    
    
    def mutate(self, candidate, n_neighbors: int):
        return self.population.add_multiple_candidates(lambda: self._mutate_helper(candidate, n_neighbors), n=n_neighbors)


class MultiMutation:
    def __init__(self, cfg, exp_path, population: Population, output_schema, predictors, split_tracker: SplitTracker, predict_op, rng: np.random.RandomState):
        li("loading MultiMutation...")
        self.cfg_op = cfg
        self.weights = []
        self.ops = []
        for cfg_sub in cfg["ops"]:
            self.weights.append(cfg_sub["weight"])
            op = get_class_instance_by_config(
                cfg_sub["cfg"],
                exp_path=exp_path,
                population=population,
                output_schema=output_schema,
                predictors=predictors,
                split_tracker=split_tracker,
                predict_op=predict_op,
                rng=rng)
            self.ops.append(op)
            
        self.weights = np.array(self.weights)
        self.weights /= self.weights.sum()
        ld(f"weights = {self.weights}")
        self.rng = rng
        
    
    def mutate(self, candidate, n_neighbors: int):
        op_idx = self.rng.choice(len(self.weights), p=self.weights)
        op_impl = self.cfg_op["ops"][op_idx]["cfg"]["impl"]
        mutate_op = self.ops[op_idx]
        ld(f"selected mutation class: {op_impl} at idx: {op_idx}")
        return mutate_op.mutate(candidate, n_neighbors)
