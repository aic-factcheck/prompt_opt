from copy import deepcopy
from pathlib import Path
from pprint import pformat
import re

import jmespath
import numpy as np

from prompt_opt.pipeline.stage import PipelineStage
from prompt_opt.pipeline.utils import select
from prompt_opt.utils import get_class_instance_by_config, pf, ld, lw


class Evaluate(PipelineStage):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.gold = cfg["gold"]
        self.pred = cfg["pred"]
        self.match_keys = cfg["match_keys"]
        self.jmespath = cfg["jmespath"]
        self.score_op_cfgs = cfg["score_ops"]
        
        
    def execute(self, state, load_fn, save_fn):

        def context_init_fn():
            exp_path = Path(self.cfg["out_dir"], "evaluate")
            exp_path.mkdir(parents=True, exist_ok=True)
            
            score_ops = {}
            for score_op_cfg in self.score_op_cfgs:
                score_op = get_class_instance_by_config(
                    score_op_cfg,
                    exp_path=exp_path,
                    predictors=state["predictors"])
                score_ops[score_op.get_score_key()] = score_op
            return {"score_ops": score_ops}
        
        
        def sample_fn(sample, context):
            sample = deepcopy(sample)
            pred = sample["pred"]
            gold = sample["gold"]
            # ld(f"pred=\n{pf(pred)}")
            # ld(f"gold=\n{pf(gold)}")
            scores = sample.get("scores", {})
            for score_key, score_op in context["score_ops"].items():
                ld(f"score_op={score_op}")
                score = score_op.score_sample(gold, pred)
                # ld(f"scoring=\n{pf(scoring)}")
                scores[score_key] = score

            return scores, None
        
        first_match_key = self.match_keys[0] # align based on the first key, check match for all match keys later
        # ld(f"self.gold={self.gold}")
        # ld(f"self.pred={self.pred}")
        gold_samples = state["store"][self.gold]
        mkey2gold = {e[first_match_key]: e for e in gold_samples}
        pred_samples = state["store"][self.pred]
        # ld(f"gold_samples[0]\n{pf(gold_samples[0])}")
        # ld(f"pred_samples[0]\n{pf(pred_samples[0])}")
        input_ = []
        for i, psample in enumerate(pred_samples):
            gsample = mkey2gold[psample[first_match_key]]
            for mkey in self.match_keys:
                pval = psample.get(mkey, f"MISSING in pred: {mkey}") 
                gval = gsample.get(mkey, f"MISSING in gold: {mkey}") 
                assert pval == gval, (pval, gval)
            pselect = jmespath.search(self.jmespath, psample)
            gselect = jmespath.search(self.jmespath, gsample)
            input_.append({"pred": pselect, "gold": gselect})
            if i == 0:
                ld(f"gold example\n{pf(gselect)}")
                ld(f"pred example\n{pf(pselect)}")
        return self._execute_batch(sample_fn, input_, load_fn, save_fn, context_init_fn=context_init_fn)
