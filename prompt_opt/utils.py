from copy import deepcopy
import importlib
import json
import re
import textwrap


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
    assert 'cfg' not in kwargs
    kwargs['cfg'] = deepcopy(cfg)
    return get_class_instance(class_path, **kwargs)


def jformat(o):
    return json.dumps(o, indent=2, ensure_ascii=False)
    
    
def pf(txt, width=120):
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
        