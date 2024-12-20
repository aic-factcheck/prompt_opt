from copy import deepcopy
from loguru import logger
import numpy as np
from pathlib import Path


from ..agents.agent_chat import AgentChat
from ..optimizers.population import Population
from ..utils import *


class TRRInitAllExamplesJSON:
    def __init__(self, cfg, population: Population, dataset_loader, predictors):
        logger.info("loading TRRInitAllExamplesJSON...")
        self.cfg_op = cfg
        self.population = population
        self.predictor = predictors[self.cfg_op["model"]]
        self.trn_data = dataset_loader.get_data()["trn"]
        self.output_schema = dataset_loader.get_output_schema()
        
        self.system_content = self.predictor.get_template(self.cfg_op["template_system_content"]).render()
        self.template_init_all_examples = self.predictor.get_template(self.cfg_op["template_init_all_examples"])
        self.template_generate_prompt = self.predictor.get_template(self.cfg_op["template_generate_prompt"])
        

    def _generate_initial_helper(self, rng):
        trn_data = deepcopy(self.trn_data)
        trn_data = rng.permutation(trn_data)
        schema_str = jformat(self.output_schema)
        
        agent = AgentChat(self.predictor, self.system_content)
        
        batch = [{"query": e["query"], "answer": jformat(e["answer"])} for e in trn_data]
        examples_prompt = self.template_init_all_examples.render(dataset=batch, schema=schema_str)
        
        response = agent.query(prompt=examples_prompt, temperature=0.3, frequency_penalty=0.05, seed=rng.randint(1e10))
        prompt_generate_prompt = self.template_generate_prompt.render(schema=schema_str)
        response = agent.query(prompt=prompt_generate_prompt, temperature=0.3, frequency_penalty=0.05, seed=rng.randint(1e10))
        
        return {"messages": agent.history()}
          
    def generate_initial(self, rng):
        return self.population.add_candidate(lambda: self._generate_initial_helper(rng))