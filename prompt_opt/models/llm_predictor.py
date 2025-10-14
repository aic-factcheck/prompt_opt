from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Optional
from loguru import logger
import time

from copy import deepcopy
from jinja2 import Environment, FileSystemLoader, Template
import openai
import httpx

from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl, process_to_jsonl

from prompt_opt.utils import extract_json_string, is_valid_json, pf, ld, lw

fix_json_template = """I need your answer in JSON with following schema:
    {{ schema }}
    """
    
@dataclass
class LLMOutput:
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    duration: float = 0.0


class LLMPredictor:
    def get_system_role(self) -> str:
        raise NotImplementedError("")
    
    def predict(self, messages: list[dict[str, Any]], seed=0, guided_json=None, **kw_args) -> LLMOutput:
        raise NotImplementedError("")


class OpenAIPredictor(LLMPredictor):
    def __init__(self, 
                 model_name: str, 
                 template_dir: str = '.',
                 base_url = None,
                 sampling_params={},
                 ignore_temperature: bool = False,
                 log_jsonl=None): # o1 and o3 models have no temperature
        assert not log_jsonl, "logging not implemented yet!"
        self._model_name = model_name
        self._prompt_env = Environment(loader=FileSystemLoader(template_dir))
        self._base_url = base_url
        self.sampling_params = sampling_params
        self.ignore_temperature = ignore_temperature
        # openai.timeout = httpx.Timeout(connect=5.0, read=10*600.0, write=10*600.0, pool=10*600.0)
        logger.debug(f"openai.timeout={openai.timeout}")


    def get_system_role(self) -> str:
        return "developer"


    def get_template(self, name: Optional[str]) -> Optional[Template]:
        return self._prompt_env.get_template(name) if name else None
    
    
    def render_template(self, name: Optional[str], **kwargs) -> Optional[str]:
        return self._prompt_env.get_template(name).render(**kwargs) if name else None
    
    
    def predict(self, messages: list[dict[str, Any]], seed=0, guided_json=None, **kw_args) -> LLMOutput:
        st = time.time()
        client = openai.OpenAI(base_url=self._base_url)
        if guided_json:
            guided_json = json.loads(guided_json)
        extra_kws = deepcopy(self.sampling_params)
        extra_kws.update(kw_args)
        # if guided_json: # OpenAI models
        #     extra_kws["response_format"] = {
        #             "type": "json_schema",
        #             "json_schema": {
        #                 "name": "answer",
        #                 "strict": True,
        #                 "schema": guided_json,
        #             }
        #     }
            
        if self.ignore_temperature:
            if "temperature" in extra_kws:
                del extra_kws["temperature"]
            if "frequency_penalty" in extra_kws:
                del extra_kws["frequency_penalty"]
            
        if guided_json: # llama.cpp server
            extra_kws["response_format"] = {
                    "type": "json_object",
                    "schema": guided_json,
                    "strict": True
            }
            
        if guided_json: # OpenAI models
            extra_kws["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "json_response",
                        "schema": guided_json,
                        "strict": True
                    }
            }
        
        completion = client.chat.completions.create(
            model=self._model_name,
            messages=messages, # type: ignore
            seed=seed,
            **extra_kws
        ) # type: ignore

        model_response = completion.choices[0].message
        if guided_json:
            assert not model_response.refusal, model_response
        duration = time.time() - st
        return LLMOutput(content=model_response.content, duration=duration)
    

class VLLMPredictor(LLMPredictor):
    def __init__(self, 
                 model_name: str,
                 post_completion,
                 openai_base_url: str, 
                 api_key: str='EMPTY',
                 sampling_params={},
                 guided_decoding_backend: Optional[str]=None,
                 template_dir: str = '.',
                 log_jsonl=None):
        self._model_name = model_name
        self. post_completion = post_completion
        self._openai_base_url = openai_base_url
        self._api_key = api_key
        self.sampling_params = sampling_params
        self._guided_decoding_backend = guided_decoding_backend
        self._prompt_env = Environment(loader=FileSystemLoader(template_dir))
        openai.timeout = httpx.Timeout(connect=5.0, read=10*600.0, write=10*600.0, pool=10*600.0)
        self.log_jsonl = log_jsonl
        logger.debug(f"openai.timeout={openai.timeout}")


    def get_system_role(self) -> str:
        return "system"
    

    def get_template(self, name: Optional[str]) -> Optional[Template]:
        return self._prompt_env.get_template(name) if name else None
    
    
    def render_template(self, name: Optional[str], **kwargs) -> Optional[str]:
        return self._prompt_env.get_template(name).render(**kwargs) if name else None
    
    
    def predict(self, messages: list[dict[str, Any]], seed=0, guided_json=None, guided_regex=None, **kw_args):
        st = time.time()
        client = openai.OpenAI(base_url=self._openai_base_url, api_key=self._api_key)

        # ld("kw_args\n", pf(kw_args))
        # ld("self.sampling_params\n", pf(self.sampling_params))
        extra_kws = deepcopy(kw_args)
        # extra_kws.update(self.sampling_params)
        extra_kws["extra_body"] = deepcopy(self.sampling_params)
        if "temperature" in extra_kws and "temperature" in extra_kws["extra_body"]:
            # extra_kws["temperature"] = extra_kws["extra_body"]["temperature"]
            del extra_kws["extra_body"]["temperature"]
        
        # ld("extra_kws\n", pf(extra_kws))
        
        if guided_json:
            extra_kws["extra_body"].update({
                "guided_json": guided_json,
            })
            if self._guided_decoding_backend:
                extra_kws["extra_body"]["guided_decoding_backend"] = self._guided_decoding_backend
                
        elif guided_regex:
            extra_kws["extra_body"].update({
                "guided_regex": guided_regex,
                "guided_decoding_backend": self._guided_decoding_backend
            })
            if self._guided_decoding_backend:
                extra_kws["extra_body"]["guided_decoding_backend"] = self._guided_decoding_backend

        completion_kws = {
            "model": self._model_name,
            "messages": messages,
            "seed": seed,
            **extra_kws
        }
        
        # ld(pf(completion_kws))
        if self.log_jsonl: # in case the LLM call fails save at least exact params for the debugging
            last_completion_path = Path(self.log_jsonl).with_suffix('.last.json')
            write_json(last_completion_path, completion_kws)
            
        completion = client.chat.completions.create(**completion_kws)
        duration = time.time() - st
        
        llm_output = self.post_completion.postprocess(completion, duration_completion=duration)
        
        if self.log_jsonl:
            logrec = {
                "call": completion_kws,
                "raw_response": json.loads(completion.model_dump_json()),
                "response": asdict(llm_output)
            }
            write_jsonl(self.log_jsonl, [logrec], append=True)
        
        return llm_output
    
    
class OllamaPredictor(LLMPredictor):
    # based on VLLMPredictor
    def __init__(self, 
                 model_name: str,
                 post_completion,
                 openai_base_url: str, 
                 api_key: str='EMPTY',
                 sampling_params={},
                 guided_decoding="standard",
                 template_dir: str = '.',
                 log_jsonl=None):
        
        assert guided_decoding in {"standard", "chat"}
        self._model_name = model_name
        self. post_completion = post_completion
        self._openai_base_url = openai_base_url
        self._api_key = api_key
        self.sampling_params = sampling_params
        self.guided_decoding = guided_decoding
        self._prompt_env = Environment(loader=FileSystemLoader(template_dir))
        openai.timeout = httpx.Timeout(connect=5.0, read=10*600.0, write=10*600.0, pool=10*600.0)
        self.log_jsonl = log_jsonl
        logger.debug(f"openai.timeout={openai.timeout}")


    def get_system_role(self) -> str:
        return "system"
    

    def get_template(self, name: Optional[str]) -> Optional[Template]:
        return self._prompt_env.get_template(name) if name else None
    
    
    def render_template(self, name: Optional[str], **kwargs) -> Optional[str]:
        return self._prompt_env.get_template(name).render(**kwargs) if name else None
    
    
    def predict(self, messages: list[dict[str, Any]], seed=0, guided_json=None, guided_regex=None, **kw_args):
        assert not guided_regex, "Guided regex not yet implemented!"
        st = time.time()
        long_timeout_client = httpx.Client(timeout=7200)
        client = openai.OpenAI(base_url=self._openai_base_url, api_key=self._api_key, http_client=long_timeout_client)

        # ld("kw_args\n", pf(kw_args))
        # ld("self.sampling_params\n", pf(self.sampling_params))
        extra_kws = deepcopy(kw_args)
        # extra_kws.update(self.sampling_params)
        extra_kws["extra_body"] = deepcopy(self.sampling_params)
        if "temperature" in extra_kws and "temperature" in extra_kws["extra_body"]:
            # extra_kws["temperature"] = extra_kws["extra_body"]["temperature"]
            del extra_kws["extra_body"]["temperature"]
        
        # extra_kws["options"] = {"think": True} # not yet implemented in ollama
        
        # ld("extra_kws\n", pf(extra_kws))
        
        if guided_json:
            if self.guided_decoding == "standard":
                extra_kws["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "result", "schema": guided_json},
                    "strict": True
                }           

        completion_kws = {
            "model": self._model_name,
            "messages": messages,
            "seed": seed,
            **extra_kws
        }
        
        # ld(pf(completion_kws))
        if self.log_jsonl: # in case the LLM call fails save at least exact params for the debugging
            last_completion_path = Path(self.log_jsonl).with_suffix('.last.json')
            write_json(last_completion_path, completion_kws)
            
        completion = client.chat.completions.create(**completion_kws)
        duration = time.time() - st
        llm_output = self.post_completion.postprocess(completion, duration_completion=duration)
        
        if self.log_jsonl:
            logrec = {
                "call": completion_kws,
                "raw_response": json.loads(completion.model_dump_json()),
                "response": asdict(llm_output)
            }
            write_jsonl(self.log_jsonl, [logrec], append=True)
        
        return llm_output
    
    
class DebugPredictor(LLMPredictor):
    def __init__(self, 
                 model_name: str,
                 sampling_params={},
                 template_dir: str = '.',
                 log_jsonl=None):
        assert not log_jsonl, "logging not implemented yet!"
        self._model_name = model_name
        self.sampling_params = sampling_params
        self._prompt_env = Environment(loader=FileSystemLoader(template_dir))
        
        
    def get_system_role(self) -> str:
        return "system"
    

    def get_template(self, name: Optional[str]) -> Optional[Template]:
        return self._prompt_env.get_template(name) if name else None
    
    
    def render_template(self, name: Optional[str], **kwargs) -> Optional[str]:
        prompt = self._prompt_env.get_template(name).render(**kwargs) if name else None
        return prompt
    
    
    def predict(self, messages: list[dict[str, Any]], seed=0, guided_json=None, **kw_args):
        extra_kws = deepcopy(self.sampling_params)
        extra_kws.update(kw_args)
        
        ld(f"seed={seed}, guided_json={pf(guided_json)}")
        ld(f"extra_kws={pf(extra_kws)}")
        ld(f"messages=\n{pf(messages)}")
        
        assert False, "This is DebugPredictor: no actual prediction happens"