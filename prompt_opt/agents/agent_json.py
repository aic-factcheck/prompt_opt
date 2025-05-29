import json

from jinja2 import Template
from loguru import logger
import openai

from prompt_opt.agents.agent_chat import AgentChat
from prompt_opt.utils import extract_json_response_dseek, is_valid_json, pf, ld, lw, le


class AgentJSONDirect:
    def __init__(self, parent: AgentChat):
        self._parent = parent

    def history(self):
        return self._parent.history()

    def add_user(self, prompt: str):
        self._parent.add_user(prompt)

    def add_assistant(self, prompt: str):
        self._parent.add_assistant(prompt)

    def query(self, prompt: str, schema, **kwargs):
        model_response = self._parent.query(prompt, guided_json=schema, **kwargs)
        try:
            return json.loads(model_response)
        except Exception as e:
            return {"error": str(e)}


class AgentJSONSteppedCoT:
    def __init__(self, parent: AgentChat):
        self._parent = parent

    def history(self):
        return self._parent.history()

    def add_user(self, prompt: str):
        self._parent.add_user(prompt)

    def add_assistant(self, prompt: str):
        self._parent.add_assistant(prompt)

    def query(self, think_prompt: str, result_prompt: str, schema, **kwargs):
        think_response = self._parent.query(think_prompt, **kwargs)
        result = self._parent.query(result_prompt, guided_json=schema, **kwargs)
        return {"think": think_response, "pred": json.loads(result)}


class AgentJSONCorrectingSteppedCoT:
    def __init__(self, parent: AgentChat, max_corrections: int):
        # works as AgentJSONSteppedCoT but firstly tries to get the output JSON without output constraining (which is slow)
        # after max_corrections rounds it fallsback to the constraining
        self._parent = parent
        self.max_corrections = max_corrections

    def history(self):
        return self._parent.history()

    def add_user(self, prompt: str):
        self._parent.add_user(prompt)

    def add_assistant(self, prompt: str):
        self._parent.add_assistant(prompt)

    def query(self, think_prompt: str, result_prompt: str, correct_prompt: str, schema, **kwargs):
        think_response = self._parent.query(think_prompt, desc="AgentJSONCorrectingSteppedCoT:think", **kwargs)

        corrections = 0
        result = self._parent.query(result_prompt, desc="AgentJSONCorrectingSteppedCoT:result", **kwargs)
        while not is_valid_json(result, schema) and corrections < self.max_corrections:
            result = self._parent.query(correct_prompt, **kwargs)
            corrections += 1

        if not is_valid_json(result, schema):
            result = self._parent.query(result_prompt, guided_json=schema, desc="AgentJSONCorrectingSteppedCoT:result_constrained", **kwargs)
        return {"think": think_response, "pred": json.loads(result), "corrections": corrections}


class AgentJSONCorrectingSteppedDSeek:
    # based on AgentJSONCorrectingSteppedCoT
    # CHANGES: think_prompt & result_prompt => process_prompt

    def __init__(self, parent: AgentChat, max_corrections: int):
        # works as AgentJSONSteppedCoT but firstly tries to get the output JSON without output constraining (which is slow)
        # after max_corrections rounds it fallsback to the constraining
        self._parent = parent
        self.max_corrections = max_corrections

    def history(self):
        return self._parent.history()

    def add_user(self, prompt: str):
        self._parent.add_user(prompt)

    def add_assistant(self, prompt: str):
        self._parent.add_assistant(prompt)

    def query(self, process_prompt: str, correct_prompt: str, failsafe_prompt: str, schema, **kwargs):

        corrections = 0
        result = extract_json_response_dseek(
            self._parent.query(process_prompt, desc="AgentJSONCorrectingSteppedDSeek:process", **kwargs)
        )
        while not is_valid_json(result["answer"], schema) and corrections < self.max_corrections:
            corrections += 1
            result = extract_json_response_dseek(
                self._parent.query(
                    correct_prompt, desc=f"AgentJSONCorrectingSteppedDSeek:correct_{corrections}", **kwargs
                )
            )

        if not is_valid_json(result["answer"], schema):
            result = extract_json_response_dseek(
                self._parent.query(
                    failsafe_prompt, guided_json=schema, desc="AgentJSONCorrectingSteppedDSeek:failsafe", **kwargs
                )
            )
        try:
            pred = json.loads(result["answer"])
            return {"think": result["think"], "pred": pred, "corrections": corrections}
        except Exception as e:
            logger.debug(result)
            logger.debug(result["answer"])
            raise(e)


class AgentJSONForReasoningModels:
    # for reasoning models as supported by VLLM now: https://github.com/vllm-project/vllm/pull/12955
    # based on AgentJSONCorrectingSteppedDSeek
    # CHANGES: removed correcting steps, simplified

    def __init__(self, parent: AgentChat):
        self._parent = parent

    def history(self):
        return self._parent.history()

    def add_user(self, prompt: str):
        self._parent.add_user(prompt)

    def add_assistant(self, prompt: str):
        self._parent.add_assistant(prompt)

    def query(self, process_prompt: str, schema, **kwargs):
        try:
            result, think = self._parent.query(
                process_prompt, guided_json=schema, 
                desc="AgentJSONForReasoningModels:process", 
                reasoning=True, **kwargs)
        except openai.APITimeoutError:
            errmsg = "ERROR: JSON generation failed!"
            lw(f"JSON generation failed due to timeout: process_prompt=\n{process_prompt}")
            return {"think": errmsg, "pred": errmsg}
        
        try:
            if result is None:
                pred = "ERROR: JSON generation failed!"
                lw(f"JSON generation failed: process_prompt=\n{process_prompt}\nthink=\n{think}")
            else:
                try:
                    pred = json.loads(result)
                except Exception as e:
                    le("result\n", result)
                    le("think\n", think)
                    raise e
            return {"think": think, "pred": pred}
        except Exception as e:
            ld(think)
            ld(result)
            raise(e)