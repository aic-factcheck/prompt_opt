from copy import deepcopy
from pathlib import Path

from loguru import logger
import numpy as np

from ..agents.agent_chat import AgentChat
from ..agents.agent_json import AgentJSONCorrecting, AgentJSONCorrectingSteppedDSeek, AgentJSONSteppedCoT, AgentJSONCorrectingSteppedCoT, AgentJSONForReasoningModels
from ..utils import *


class COTPredictSteppedJSON:
    def __init__(self, cfg, exp_path, predictors):
        logger.info("loading COTPredictStepped...")
        self.cfg_op = cfg
        self.predictor = predictors[self.cfg_op["model"]]

        self.system_content = self.predictor.render_template(self.cfg_op.get("template_system_content"))
        self.template_think = self.predictor.get_template(self.cfg_op["template_think"])
        self.template_result = self.predictor.get_template(self.cfg_op["template_result"])
        self.template_correct = (
            self.predictor.get_template(self.cfg_op["template_correct"]) if "template_correct" in self.cfg_op else None
        )
        self.max_corrections = self.cfg_op.get("max_corrections")


    def predict(self, prompt, query, output_schema, examples=None):
        assert examples is None, "Strictly zero-shot!"
        schema_str = jformat(output_schema)

        try:
            prompt_think = self.template_think.render(prompt=prompt, query=query, schema=schema_str)
            prompt_result = self.template_result.render(schema=schema_str)
            if not self.template_correct:
                agent = AgentJSONSteppedCoT(AgentChat(self.predictor, self.system_content))
                response = agent.query(
                    prompt_think,
                    prompt_result,
                    schema=schema_str,
                    temperature=0.0,
                    frequency_penalty=0.05,
                    debug=False,
                )
            else:
                agent = AgentJSONCorrectingSteppedCoT(
                    AgentChat(self.predictor, self.system_content), max_corrections=self.max_corrections
                )
                prompt_correct = self.template_correct.render(schema=schema_str)
                response = agent.query(
                    think_prompt=prompt_think,
                    result_prompt=prompt_result,
                    correct_prompt=prompt_correct,
                    schema=schema_str,
                    temperature=0.0,
                    frequency_penalty=0.05,
                    debug=False,
                )

        except Exception as e:
            logger.warning(f"constrained JSON prediction failed, ignoring schema:\n{e}")
            response = agent.query(
                prompt_think, prompt_result, schema=None, temperature=0.0, frequency_penalty=0.05, debug=False
            )
            response["pred"] = decode_unicode_escapes(response["pred"])
        return response, agent.history()


class DSeekPredictSteppedJSON:
    # based on COTPredictSteppedJSON
    # CHANGES:
    #  - template_think & template_result => template_process, template_failsafe
    #  - template_correct is obligatory
    #  - simplified - we always expect a correct JSON output (forcing schema in failsafe step)

    def __init__(self, cfg, exp_path, predictors):
        logger.info("loading DSeekPredictSteppedJSON...")
        self.cfg_op = cfg
        self.predictor = predictors[self.cfg_op["model"]]

        self.system_content = self.predictor.render_template(self.cfg_op.get("template_system_content"))
        self.template_process = self.predictor.get_template(self.cfg_op["template_process"])
        self.template_correct = self.predictor.get_template(self.cfg_op["template_correct"])
        self.template_failsafe = self.predictor.get_template(self.cfg_op["template_failsafe"])
        self.max_corrections = self.cfg_op.get("max_corrections")

    def predict(self, prompt, query, output_schema, examples=None):
        assert examples is None, "Strictly zero-shot!"
        schema_str = jformat(output_schema)

        prompt_process = self.template_process.render(prompt=prompt, query=query, schema=schema_str)
        prompt_correct = self.template_correct.render(query=query, schema=schema_str)
        prompt_failsafe = self.template_failsafe.render(prompt=prompt, query=query, schema=schema_str)
        
        agent = AgentJSONCorrectingSteppedDSeek(
            AgentChat(self.predictor, self.system_content), max_corrections=self.max_corrections
        )
        response = agent.query(
            process_prompt=prompt_process,
            correct_prompt=prompt_correct,
            failsafe_prompt=prompt_failsafe,
            schema=schema_str,
            temperature=0.0,
            frequency_penalty=0.05,
            debug=False,
        )

        return response, agent.history()
    
    
class PredictCorrectedJSON:
    # similar to DSeekPredictSteppedJSON, but using AgentJSONCorrecting
    # aimed for GPT-OSS models which, at this time, do not support constrained JSON generation

    def __init__(self, cfg, exp_path, predictors):
        logger.info("loading PredictCorrectedJSON...")
        self.cfg_op = cfg
        self.predictor = predictors[self.cfg_op["model"]]

        self.system_content = self.predictor.render_template(self.cfg_op.get("template_system_content"))
        self.template_process = self.predictor.get_template(self.cfg_op["template_process"])
        self.template_correct = self.predictor.get_template(self.cfg_op["template_correct"])
        self.max_corrections = self.cfg_op.get("max_corrections")
        if "temperature" in self.cfg_op:
            self.sampling_opts = {"temperature": self.cfg_op["temperature"]}
        else:
            self.sampling_opts = {"temperature": 0.0, "frequency_penalty": 0.05}
            
        

    def predict(self, prompt, query, output_schema, examples=None):
        assert examples is None, "Strictly zero-shot!"
        schema_str = jformat(output_schema)

        prompt_process = self.template_process.render(prompt=prompt, query=query, schema=schema_str)
        prompt_correct = self.template_correct.render(query=query, schema=schema_str)
        
        agent = AgentJSONCorrecting(
            AgentChat(self.predictor, self.system_content), max_corrections=self.max_corrections
        )
        response = agent.query(
            process_prompt=prompt_process,
            correct_prompt=prompt_correct,
            schema=schema_str,
            debug=False,
            **self.sampling_opts
        )

        return response, agent.history()


class PredictReasoningJSON:
    # for reasoning models as supported by VLLM now: https://github.com/vllm-project/vllm/pull/12955
    # based on AgentJSONCorrectingSteppedDSeek
    # CHANGES: removed correcting steps, simplified

    def __init__(self, cfg, exp_path, predictors):
        logger.info("loading PredictReasoningJSON...")
        self.cfg_op = cfg
        self.predictor = predictors[self.cfg_op["model"]]

        self.system_content = self.predictor.render_template(self.cfg_op.get("template_system_content"))
        self.template_process = self.predictor.get_template(self.cfg_op["template_process"])

    def predict(self, prompt, query, output_schema, examples=None):
        assert examples is None, "Strictly zero-shot!"
        schema_str = jformat(output_schema)

        prompt_process = self.template_process.render(prompt=prompt, query=query, schema=schema_str)
        
        agent = AgentJSONForReasoningModels(
            AgentChat(self.predictor, self.system_content)
        )
        response = agent.query(
            process_prompt=prompt_process,
            schema=schema_str,
            temperature=0.0,
            frequency_penalty=0.05,
            debug=False,
        )

        return response, agent.history()