from copy import deepcopy
from pathlib import Path

from loguru import logger
import numpy as np

from ..agents.agent_chat import AgentChat
from ..agents.agent_json import AgentJSONSteppedCoT
from ..utils import *


class COTPredictSteppedJSON:
    def __init__(self, cfg, dataset_loader, predictors):
        logger.info("loading COTPredictStepped...")
        self.cfg_op = cfg
        self.predictor = predictors[self.cfg_op["model"]]
        self.trn_data = dataset_loader.get_data()["trn"]
        self.output_schema = dataset_loader.get_output_schema()
        
        self.system_content = self.predictor.get_template(self.cfg_op["template_system_content"]).render()
        self.template_think = self.predictor.get_template(self.cfg_op["template_think"])
        self.template_result = self.predictor.get_template(self.cfg_op["template_result"])
        
        
    def predict(self, prompt, query, examples=None):
        assert examples is None, "Strictly zero-shot!"
        schema_str = jformat(self.output_schema)
        
        agent = AgentJSONSteppedCoT(AgentChat(self.predictor, self.system_content))
        prompt_think = self.template_think.render(prompt=prompt, query=query, schema=schema_str)
        prompt_result = self.template_result.render(schema=schema_str)
        
        try:
            response = agent.query(prompt_think, prompt_result, schema=schema_str, temperature=0.0, frequency_penalty=0.05, debug=False)
        except Exception as e:
            logger.warning(f"constrained JSON prediction failed, ignoring schema:\n{e}")
            response = agent.query(prompt_think, prompt_result, schema=None, temperature=0.0, frequency_penalty=0.05, debug=False)
            response['pred'] = decode_unicode_escapes(response['pred'])
        return response, agent.history()
    