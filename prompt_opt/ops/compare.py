from pathlib import Path
from typing import Any

from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl, process_to_jsonl

from ..agents.agent_chat import AgentChat
from ..optimizers.predict_evaluate import get_candidate_score_source
from ..utils import *


class DebugCompare:
    def __init__(self, cfg, exp_path, predictors, rng):
        logger.info("loading DebugCompare...")
        self.cfg_op = cfg
        self.log_file = Path(exp_path, cfg["log"])

        self.ops = []
        for cfg_sub in cfg["ops"]:
            op = get_class_instance_by_config(
                cfg_sub,
                exp_path=exp_path,
                predictors=predictors,
                rng=rng)
            self.ops.append(op)


    def better_candidate(self, candidate1, candidate2):
        results = []
        dbg = []
        for op in self.ops:
            op_name = type(op).__name__
            ld(f"comparing {op_name}")
            candidate = op.better_candidate(candidate1, candidate2)
            candidate_idx = 0 if candidate == candidate1 else 1
            dbg.append({"op_name": op_name, "candidate_idx": candidate_idx})
            results.append(candidate)
        write_jsonl(self.log_file, [dbg], append=True)
        return results[0]


class CompareScore:
    def __init__(self, cfg, exp_path, predictors, rng):
        logger.info("loading CompareScore...")
        self.cfg_op = cfg
        self.select_split = self.cfg_op["select_split"]
        self.score_key = self.cfg_op["score_key"]

    def better_candidate(self, candidate1, candidate2):
        score1 = get_candidate_score_source(candidate1, self.select_split, self.score_key)
        score2 = get_candidate_score_source(candidate2, self.select_split, self.score_key)
        if score1 >= score2:
            return candidate1
        else:
            return candidate2


class DSeekCompareJSON:
    def __init__(self, cfg, exp_path, predictors, rng) -> None:
        logger.info("loading CompareScore...")
        self.cfg_op = cfg
        self.predictor = predictors[cfg["model"]]
        self.rng = rng
        
        self.select_split = self.cfg_op["select_split"]

        self.system_content = self.predictor.render_template(self.cfg_op.get("template_system_content"))
        self.template_compare = self.predictor.get_template(self.cfg_op["template_compare"])

        self.cache_file: Path = Path(exp_path, "dseek_compare_json.jsonl")
        if self.cache_file.is_file():
            self.cache = {tuple(e["key"]): e["value" ] for e in read_jsonl(self.cache_file)}
        else:
            self.cache = {}
            
    def _save_cache(self, k, v, meta):
        self.cache[k] = v
        e = {"key": k, "value": v, "meta": meta}
        write_jsonl(self.cache_file, [e], append=True)


    def better_candidate(self, candidate1, candidate2):
        cache_key = (candidate1["id"], candidate2["id"])
        if cache_key[1] < cache_key[0]:
            cache_key = (cache_key[1], cache_key[0])
        
        if cache_key in self.cache:
            ld(f"cache hit for: {cache_key}")
            cid =  self.cache[cache_key]
            if cid == candidate1["id"]:
                return candidate1
            elif cid == candidate2["id"]:
                return candidate2
            else:
                raise ValueError(f"unknown cid: {cid}")
        
        agent = AgentChat(self.predictor, self.system_content)
        
        # select random prediction
        # it is not that easy, because candidates have shuffled target splits
        # we must first retrieve source split indices
        src_indices = []
        for tgt_split, tgt_cfg in candidate1["split2indices"].items():
            if tgt_cfg["source"] == self.select_split:
                src_indices += tgt_cfg["indices"]
        ld("src_indices", src_indices)
        
        # select random index in source split
        sel_idx = self.rng.choice(len(src_indices)) 
        ld("sel_idx", sel_idx)
        
        def prediction_for_source_idx(candidate, idx) -> Any:
            for tgt_split, tgt_cfg in candidate["split2indices"].items():
                if tgt_cfg["source"] == self.select_split:
                    if idx in tgt_cfg["indices"]:
                        tgt_idx = tgt_cfg["indices"].index(idx)
                        return candidate["split"][tgt_split][tgt_idx]
                
        pred1 = prediction_for_source_idx(candidate1, sel_idx)
        pred2 = prediction_for_source_idx(candidate2, sel_idx)
        # check that we have indeed matching source split predictions
        assert pred1["gold"] == pred2["gold"], (pred1["gold"], pred2["gold"])
        gold_str = jformat(pred1["gold"])
        pred1_str = jformat(pred1["pred"])
        pred2_str = jformat(pred2["pred"])
        # ld("pred1_str", pred1_str)
        
        prompt_compare = self.template_compare.render(
            query=pred1["query"], gold=gold_str, prediction1=pred1_str, prediction2=pred2_str
        )

        response = agent.query(
            prompt=prompt_compare,
            frequency_penalty=0.05,
            seed=self.rng.randint(int(1e10)),
        )

        # ld("response\n:", response)
        meta = {"history": agent.history(), "response": response, "gold": gold_str, "pred1": pred1_str, "pred2": pred2_str}
        if response:
            # find the last occurence of 1/2 label in the response
            matches = re.findall(r'\b[12]\b', str(response))
            if matches:
               meta["label"] = int(matches[-1])
               if meta["label"] == 1:
                   ld(f"candidate1 id={candidate1['id']} wins")
                   self._save_cache(cache_key, candidate1["id"], meta)
                   return candidate1
               else:
                   ld(f"candidate2 id={candidate2['id']} wins")
                   self._save_cache(cache_key, candidate2["id"], meta)
                   return candidate2
            else:
                meta["error"] = "no match"
                lw(f"No match in compare. Response: {response}")
        else:
            meta["error"] = "no response"
            lw(f"No response!")
            
        ld(f"candidate1 id={candidate1['id']} selected (draw or error)")
        self._save_cache(cache_key, candidate1["id"], meta)
        return candidate1
        

