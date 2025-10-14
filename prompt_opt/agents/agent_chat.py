from copy import deepcopy
from typing import Any, Optional, Union

from jinja2 import Template
from loguru import logger

from prompt_opt.models.llm_predictor import LLMPredictor
from prompt_opt.utils import pf, ld, lw


class AgentChat:
    def __init__(self, predictor: LLMPredictor, system_content: Optional[Union[str, Template]]):
        self._predictor = predictor
        self._system_content = system_content
        self._history = [{"role": predictor.get_system_role(), "content": self._system_content}] if self._system_content else []
        # self.prepend_thinking = True # DEFAULT
        self.prepend_thinking = False #TODO: this is fixed now, it should be configurable, but there are so many instantiations...


    def history(self) -> list[dict[str, Any]]:
        return deepcopy(self._history)


    def _clean_history(self) -> list[dict[str, Any]]:
        # returns history with just "role" and "content" attributes as OpenAI API does not support any metadata
        return [{"role": e["role"], "content": e["content"]} for e in self._history]


    def add_user(self, prompt: str, duration: float = 0.0, desc: str = "") -> None:
        # assert len(self._history) == 0 or self._history[-1]["role"] in ["system", "assistant"], self._history[-1]["role"]
        self._history.append({"role": "user", "content": prompt, "duration": duration, "desc": desc})


    def add_assistant(self, prompt: str, duration: float = 0.0, desc: str = "") -> None:
        # assert len(self._history) == 0 or self._history[-1]["role"] == "user", self._history[-1]["role"]
        self._history.append({"role": "assistant", "content": prompt, "duration": duration, "desc": desc})


    def query(self, prompt: str, desc: str = "", reasoning=False, debug=False, **kwargs):
        # debug = True
        # `reasoning = True` forces support for VLLM reasoning models which decomposes LLM answer into "thinking" and "answer" stages
        self.add_user(prompt)
        if debug:
            ld(f"reasoning=\n{reasoning}")
            ld(f"prompt=\n{prompt}")
            ld(f"kwargs=\n{pf(kwargs)}")
            
        model_response = self._predictor.predict(messages=self._clean_history(), **kwargs)
        if debug:
            ld(f"model_response=\n{pf(model_response)}")
        
        if self.prepend_thinking and hasattr(model_response, "reasoning_content") and model_response.reasoning_content:
            # if reasoning model such as DeepSeek is used, we still need to get back to original output of the model,
            # when reasoning is not enforced
            # it is important for constrained generation only anywat
            # TODO make this general for any reasoning models using (possibly) different ways to encode thinking phase
            full_content = f"<think>\n{model_response.reasoning_content}\n</think>\n\n{model_response.content}"
        else:
            full_content = model_response.content
        
        if not full_content:
            ld("messages:\n", self._clean_history())
            ld(model_response)
            raise ValueError("full_content should not be None")
            
        self.add_assistant(prompt=full_content, duration=model_response.duration, desc=desc)
        if debug:
            ld(f"content=\n{full_content}")
            
        if reasoning:
            assert hasattr(model_response, "reasoning_content")
            if debug:
                ld(f"reasoning_content=\n{model_response.reasoning_content}")
            return model_response.content, model_response.reasoning_content
        else:
            return model_response.content
