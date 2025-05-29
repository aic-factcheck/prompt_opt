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


class DSeekImproveUnlabeledJSON:
    # based on DSeekImproveJSON
    def __init__(self, cfg, exp_path, population: Population, output_schema, predictors, split_tracker: SplitTracker, predict_op, rng: np.random.RandomState):
        li("loading DSeekImproveUnlabeledJSON...")
        self.cfg_op = cfg
        self.population = population
        self.output_schema = output_schema
        self.predictor = predictors[self.cfg_op["model"]]
        self.labeled_size = self.cfg_op["labeled_size"]
        self.split_tracker = split_tracker
        self.unl_samples = self.split_tracker.get_source(self.cfg_op["unlabeled_split"])
        self.predict_op = predict_op
        self.rng = rng
        
        self.unlabeled_split = self.cfg_op["unlabeled_split"]
        
        self.system_content = self.predictor.render_template(self.cfg_op.get("template_system_content"))
        self.template_predict_first = self.predictor.get_template(self.cfg_op["template_predict_first"])
        self.template_improve_instructions = self.predictor.get_template(self.cfg_op["template_improve_instructions"])
        
        
    def _target_predict(self, prompt, samples):
        idx = 0
        responses = []
        for sample in samples:
            response, messages = self.predict_op.predict(
                prompt=prompt, 
                query=sample["query"],
                output_schema=self.output_schema,
                examples=None)

            response.update(
                {
                    "query": sample["query"],
                    "messages": messages,
                }
            )
            responses.append(response)
            idx += 1
            li(f'done {idx}/{len(samples)}')
        return responses
            
        
    def _mutate_helper(self, candidate, n_neighbors: int):
        original_instructions = candidate2prompt_dseek(candidate)
        schema_str = jformat(self.output_schema)
        
        split2indices = self.split_tracker.sample_indices("trn", self.labeled_size)
        lbl_samples = self.split_tracker.get_samples("trn", split2indices)
        
        # get random unlabeled samples and make predictions for them
        unl_samples = self.rng.choice(self.unl_samples, n_neighbors, replace=False)
        responses = self._target_predict(original_instructions, unl_samples)
        
        neighbors = []
        for neighbor_idx, (sample, response) in enumerate(zip(unl_samples, responses)):
            st = time.time()
            li(f"generating neighbor {neighbor_idx+1}/{n_neighbors}")
            # ld("sample\n", sample)
            # ld("response\n", pf(response))
            
            query = sample["query"]
            
            agent = AgentChat(self.predictor, self.system_content)
            
            # first simulate that the optimizer model did the actual prediction for this unlabeled sample
            prompt_improve_sample = self.template_predict_first.render(instructions=original_instructions, 
                                                                       query=query, 
                                                                       schema=schema_str)
            agent.add_user(prompt_improve_sample)
            agent.add_assistant(response["messages"][-1]["content"])
            
            # now provide training data including gold predictions, so LLM can infer if "its" prediction is align with that
            # and generate new instructions
            lbl_dataset = []
            for sample in lbl_samples:
                query = sample["query"]
                gold = sample["answer"]
                gold_str = jformat(gold)
                lbl_dataset.append({"query": query, "gold": gold_str})
            prompt_improve_instructions = self.template_improve_instructions.render(dataset=lbl_dataset)
            response = agent.query(prompt=prompt_improve_instructions,frequency_penalty=0.05, seed=self.rng.randint(int(1e10)))
            
            neighbor = {"messages": agent.history(), "parent_id": candidate["id"], "split2indices": candidate["split2indices"], "duration": time.time()-st}
            neighbors.append(neighbor)
        
        return neighbors
    
    def mutate(self, candidate, n_neighbors: int):
        return self.population.add_multiple_candidates(lambda: self._mutate_helper(candidate, n_neighbors), n=n_neighbors)

