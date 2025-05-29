from copy import deepcopy
from pathlib import Path
from pprint import pformat
import re

from loguru import logger
import numpy as np

from prompt_opt.pipeline.stage import PipelineStage
from prompt_opt.pipeline.utils import select
from prompt_opt.utils import get_class_instance_by_config, pf, ld, lw


class Identity(PipelineStage):
    def __init__(self, cfg):
        super().__init__(cfg)
        deps = cfg["deps"]
        assert len(deps) == 1, deps
        self.source_key = deps[0]   
        
    def execute(self, state, load_fn, save_fn):
        return save_fn(state["store"][self.source_key], None)
    

def format_obj(obj, fmt):
    fmt_vars = re.findall(r"\{(.*?)\}", fmt)
    fmt_dict = {}
    for fv in fmt_vars:
        if '|' in fv:
            fv_alternatives = fv.split('|')
            found = False
            fv_sel = None
            for fv_sel in fv_alternatives:
                if fv_sel in obj and obj[fv_sel] != "":
                    found = True
                    break
            if not found:
                fv_sel = fv_alternatives[0]
            fmt = fmt.replace(f'{{{fv}}}', f'{{{fv_sel}}}')  
            fv = fv_sel
        fmt_dict[fv] = obj[fv]
    return fmt.format_map(fmt_dict)
    
    
class Id2Text(PipelineStage):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.target = cfg["target"]
        self.transform = cfg["transform"]
        
    def execute(self, state, load_fn, save_fn):

        def sample_fn(sample, context):
            sample = deepcopy(sample)
            # ld(f"sample={sample}")
            for cfg in self.transform:
                for inp in select(sample, cfg["source"]):
                    id2data = {e["id"]: e for e in inp}
                    # ld(pformat(id2data, sort_dicts=False))
                    fmt = cfg["format"]
                    
                    for lst in select(sample, cfg["target"]):
                        # each `lst` is list of ids selected from the sample, we will replace them in place
                        # which won't work for primitive types!
                        for i in range(len(lst)):
                            # if id not found, do not replace with text, keep it
                            if lst[i] in id2data:
                                e = id2data[lst[i]]
                                lst[i] = format_obj(e, fmt)
                            
            # ld(f"sample2={pf(sample)}")
            return sample, None

        input_ = state["store"][self.target]
        
        return self._execute_batch(sample_fn, input_, load_fn, save_fn)
