from copy import deepcopy
from pathlib import Path

from loguru import logger
import numpy as np

from ..agents.agent_chat import AgentChat
from ..agents.agent_json import AgentJSONSteppedCoT
from ..metrics.object_aligner import ObjectAligner
from ..utils import *


class ScoreObjectAligner:
    def __init__(self, cfg, exp_path, predictors):
        logger.info("loading ScoreObjectAligner...")
        self.cfg_op = cfg
        self.metric_schema = self.cfg_op["schema"]
        logger.info('dataset metric schema:\n' + jformat(self.metric_schema))
        self.metric = ObjectAligner(id_="object_aligner_metric", schema=self.metric_schema)
        self.score_key = self.cfg_op["score_key"]
        logger.info(f'dataset metric score_key: {self.score_key}')
    
        
    def get_score_key(self) -> str:
        return self.score_key
        
        
    def score_sample(self, gold, pred):
        ld("ScoreObjectAligner:evaluating...")
        return self.metric.metric(gold, pred)
    
    
class ModelBasedMD:
    def __init__(self, cfg, exp_path, predictors):
        logger.info("loading ModelBased...")
        self.cfg_op = cfg
        self.model_name = self.cfg_op["model"]
        self.predictor = predictors[self.model_name]
        self.score_key = self.cfg_op["score_key"]
        
        self.system_content = self.predictor.render_template(self.cfg_op.get("template_system_content"))
        self.template_think = self.predictor.get_template(self.cfg_op["template_think"])
        self.template_result = self.predictor.get_template(self.cfg_op["template_result"])


    def get_score_key(self) -> str:
       return self.score_key
        
            
    def score_sample(self, gold, pred):
        ld("ModelBasedMD:evaluating...")
        gold_str = jformat(gold)
        pred_str = jformat(pred)
        # logger.debug(f"gold_str={gold_str}")
        # logger.debug(f"pred_str={pred_str}")
        
        agent = AgentChat(self.predictor, self.system_content)
        prompt_think = self.template_think.render(gold=gold_str, pred=pred_str)
        prompt_result = self.template_result.render()
        
        think_response = agent.query(prompt_think, temperature=0.0, frequency_penalty=0.05, debug=False)
        response = agent.query(prompt_result, temperature=0.0, frequency_penalty=0.05, debug=False)
        
        # reasoning_match = re.search(r"(?<=# Reasoning\n)(.*?)(?=# Score)", response, re.DOTALL)
        # reasoning = reasoning_match.group(0).strip() if reasoning_match else None
        reasoning_match = re.search(r"# Reasoning\s+(.*?)\s+# Score", response, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None

        if not reasoning:
            logger.warning(f"No reasoning:\n{response}")
            reasoning = "Reasoning is missing."
        
        score_match = re.search(r"# Score\s*(\d+)", response)
        score = score_match.group(1) if score_match else None
        if not score:
            logger.warning(f"No score:\n{response}")
            assert False
            
        try:
            score = int(score.strip())/100.0
        except Exception as e:
            logger.error(f"Failed to parse score {score} for the response:\n{response}")
            
        return {"reasoning": reasoning, "score": score, "messages": agent.history()}
    
    
class ModelBasedDSeek:
    # based on ModelBasedMD
    # CHANGES:
    #   - template_think & template_result => template_score
    
    def __init__(self, cfg, exp_path, predictors):
        logger.info("loading ModelBasedDSeek...")
        self.cfg_op = cfg
        self.model_name = self.cfg_op["model"]
        self.predictor = predictors[self.model_name]
        self.score_key = self.cfg_op["score_key"]
        
        self.system_content = self.predictor.render_template(self.cfg_op.get("template_system_content"))
        self.template_score = self.predictor.get_template(self.cfg_op["template_score"])


    def get_score_key(self) -> str:
        return self.score_key
        
            
    def score_sample(self, gold, pred):
        logger.debug("ModelBasedDSeek:evaluating...")
        gold_str = jformat(gold)
        pred_str = jformat(pred)
        # logger.debug(f"gold_str={gold_str}")
        # logger.debug(f"pred_str={pred_str}")
        
        agent = AgentChat(self.predictor, self.system_content)
        prompt_score = self.template_score.render(gold=gold_str, pred=pred_str)
        
        model_response = agent.query(prompt_score, temperature=0.0, frequency_penalty=0.05, debug=False)
        response = extract_response_dseek(model_response)["answer"]
        
        reasoning_match = re.search(r"# Reasoning\s+(.*?)\s+# Score", response, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None

        if not reasoning:
            logger.warning(f"Could not extract reasoning from:\n{response}")
            reasoning = model_response
        
        score_match = re.search(r"# Score\s*(\d+)", response)
        score = score_match.group(1) if score_match else None
        if not score:
            logger.warning(f"No score found, trying to match last 0-100 number:\n{response}")
            matches = re.findall(r'\b(?:100|\d{1,2})\b', response)
            if matches:
                score = matches[-1]
            else:
                assert False, "No 0-100 number found"
            
        try:
            score = int(score.strip())/100.0
            if score < 0:
                logger.warning(f"score < 0: {score}, clamping")
                score = 0.0
            elif score > 1:
                logger.warning(f"score > 1: {score}, clamping")
                score = 1.0
        except Exception as e:
            logger.error(f"Failed to parse score {score} for the response:\n{response}")
            
        return {"reasoning": reasoning, "score": score, "messages": agent.history()}