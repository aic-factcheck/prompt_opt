from copy import deepcopy
import importlib
import json
from pprint import pformat
import re
import textwrap

from loguru import logger
from jsonschema import validate, ValidationError


def extract_json_string(s: str) -> str:
    t = s.strip()
    if t.startswith("```json") and t.endswith("```"):
        return t[7:-3].strip()
    return s


def extract_response_md(model_response: str):
    match = re.search(r"# Response\s*(.*)", model_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        print(f"WARNING: missing Response section tag, taking full output")
        return model_response.strip()
    

def candidate2prompt_md(candidate, messages_key="messages"):
    model_response = candidate[messages_key][-1]["content"]
    return extract_response_md(model_response)


def extract_response_dseek(model_response: str):
    try:
        think = ''.join(re.findall(r'<think>(.*?)</think>', model_response, flags=re.DOTALL))
        answer = re.sub(r'<think>.*?</think>', '', model_response, flags=re.DOTALL)
        return {"think": think.strip(), "answer": answer.strip()}
    except:
        return  {"think": "Thinking stage skipped or failed.", "answer": model_response}
    
    
def extract_json_response_dseek(model_response: str):
    # logger.debug(f"model_response=\n{model_response}")
    # logger.debug("----------------------------------------------------")
    response = extract_response_dseek(model_response)
    # logger.debug(f"BEFORE response=\n{response}")
    # logger.debug("----------------------------------------------------")
    response["answer"] = extract_json_string(response["answer"])
    # logger.debug(f"AFTER response.answer=\n{response['answer']}")
    return response


def candidate2prompt_dseek(candidate, messages_key="messages"):
    model_response = candidate[messages_key][-1]["content"]
    return extract_response_dseek(model_response)["answer"].strip()


def decode_unicode_escapes(o):
    # `o` JSON object
    # recursively fixes encoding in all escaped json strings
    
    if isinstance(o, str):
        try:
            if '\\u' in o:
                return o.encode('utf-8').decode('unicode_escape')
        except:
            pass
        return o
    elif isinstance(o, dict):
        new_dict = {}
        for k, v in o.items():
            new_dict[decode_unicode_escapes(k)] = decode_unicode_escapes(v)
        return new_dict
    elif isinstance(o, list):
        new_array = []
        for e in o:
            new_array.append(decode_unicode_escapes(e))
        return new_array
    else:
        return o
    

def get_class_instance(class_path: str, *args, **kwargs):
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(*args, **kwargs)


def get_class_instance_by_config(cfg, **kwargs):
    class_path = cfg["impl"]
    logger.debug(f"class_path: {class_path}, kwargs: {kwargs}")
    assert 'cfg' not in kwargs
    kwargs['cfg'] = deepcopy(cfg)
    return get_class_instance(class_path, **kwargs)


def jformat(o):
    return json.dumps(o, indent=2, ensure_ascii=False)
    
    
def pf(txt, width=120): # type: ignore
    if txt is not None:
        if not isinstance(txt, str):
            txt = str(txt)
        for par in txt.splitlines():
            print(textwrap.fill(par, width=width, replace_whitespace=False))


def print_messages(messages):
    for m in messages:
        print(m["role"].upper())
        pf(m["content"])
        print()
        
        
def is_valid_json(json_string: str, schema):
    if isinstance(schema, str):
        schema = json.loads(schema)
    try:
        data = json.loads(json_string)  # Convert string to JSON
        validate(instance=data, schema=schema)  # Validate against schema
        return True
    except (json.JSONDecodeError, ValidationError) as e:
        return False
    
# logging
def pf(obj):
    return pformat(obj, sort_dicts=False)

def li(*args, **kwargs):
    message = ' '.join(map(str, args))
    logger.opt(depth=1).info(message, **kwargs)
    
def ld(*args, **kwargs):
    message = ' '.join(map(str, args))
    logger.opt(depth=1).debug(message, **kwargs)

def lw(*args, **kwargs):
    message = ' '.join(map(str, args))
    logger.opt(depth=1).warning(message, **kwargs)
    
def le(*args, **kwargs):
    message = ' '.join(map(str, args))
    logger.opt(depth=1).error(message, **kwargs)