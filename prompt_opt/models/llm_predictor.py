from loguru import logger

from copy import deepcopy
from jinja2 import Environment, FileSystemLoader, Template
import openai
import httpx


class LLMPredictor:
    def __init__(self, 
                 model_name: str, 
                 openai_base_url: str, 
                 api_key: str='EMPTY', 
                 guided_decoding_backend: str='outlines',
                 template_dir: str = '.'):
        self._model_name = model_name
        self._openai_base_url = openai_base_url
        self._api_key = api_key
        self._guided_decoding_backend = guided_decoding_backend
        self._prompt_env = Environment(loader=FileSystemLoader(template_dir))
        # openai.timeout = httpx.Timeout(connect=5.0, read=10*600.0, write=10*600.0, pool=10*600.0)
        logger.debug(f"openai.timeout={openai.timeout}")


    def get_template(self, name: str) -> Template:
        return self._prompt_env.get_template(name)
    
    
    def predict(self, messages: str, temperature=0.3, seed=0, guided_json=None, guided_regex=None, **kw_args):
        client = openai.OpenAI(base_url=self._openai_base_url, api_key=self._api_key)

        extra_kws = deepcopy(kw_args)
        if guided_json:
            extra_kws["extra_body"] = {
                "guided_json": guided_json,
                "guided_decoding_backend": self._guided_decoding_backend
            }
        elif guided_regex:
            extra_kws["extra_body"] = {
                "guided_regex": guided_regex,
                "guided_decoding_backend": self._guided_decoding_backend
            }
            
        completion = client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=temperature,
            seed=seed,
            **extra_kws
        )

        model_response = completion.choices[0].message.content
        return model_response