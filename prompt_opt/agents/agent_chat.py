from copy import deepcopy
from typing import Union

from jinja2 import Template

from ..models.llm_predictor import LLMPredictor


class AgentChat:
    def __init__(self, predictor: LLMPredictor, system_content: Union[str,Template]):
        self._predictor = predictor
        self._system_content = system_content
        self._history = [
            {"role": "system", "content": self._system_content},
        ]
        
        
    def history(self):
        return deepcopy(self._history)
    
    
    def add_user(self, prompt: str):
        assert self._history[-1]["role"] in ["system", "assistant"], self._history[-1]["role"]
        self._history.append({"role": "user", "content": prompt})
        
        
    def add_assistant(self, prompt: str):
        assert self._history[-1]["role"] == "user", self._history[-1]["role"]
        self._history.append({"role": "assistant", "content": prompt})
        
        
    def query(self, prompt: str, debug=False, **kwargs):
        # print("kwargs", kwargs)
        self.add_user(prompt)
        if debug:
            print("prompt =\n", prompt)
            print("kwargs =", kwargs)
        model_response = self._predictor.predict(messages=self._history, **kwargs)
        self.add_assistant(model_response)
        if debug:
            print("model_response")
            print(model_response)
            print("=" * 120)
        return model_response