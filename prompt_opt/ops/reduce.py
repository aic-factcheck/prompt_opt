from functools import cmp_to_key, total_ordering
import heapq
from pathlib import Path
import numpy as np

from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl, process_to_jsonl


from prompt_opt.optimizers.predict_evaluate import get_candidate_score

from ..utils import *


class ReduceBest:
    def __init__(self, cfg, exp_path, predictors, rng):
        logger.info("loading ReduceBest...")
        self.cfg_op = cfg
        self.compare_op = get_class_instance_by_config(
            self.cfg_op["compare_op"],
            exp_path=exp_path,
            predictors=predictors,
            rng=rng
        )
        self.cache_file: Path = Path(exp_path, cfg["cache"])
        self.cache = read_jsonl(self.cache_file) if self.cache_file.is_file() else []
        self.rng = rng
        self.count = 0


    def reduce(self, parents, children):
        self.count += 1

        candidates = parents + children

        if self.count <= len(self.cache):
            selected_ids = self.cache[self.count-1]
            assert len(selected_ids) == len(parents), (len(selected_ids), len(parents))
            ld(f"{self.count} cached selection: {selected_ids}")
            id2candidate = {c["id"]: c for c in candidates}
            candidates = [id2candidate[id_] for id_ in selected_ids]
        else:
            log_compare = {"counter": 0, "pairs": []}

            def compare(a, b):
                aid = a["id"]
                bid = b["id"]
                
                best = self.compare_op.better_candidate(a, b)
                if best == a and best != b:
                    res = 1 # a > b
                elif best == b and best != a:
                    res = -1 # a < b
                else:
                    res = 0
                    
                log_compare["counter"] = log_compare["counter"]  + 1
                log_compare["pairs"].append((aid, bid))
                
                return res
                
            @total_ordering
            class Comparable:
                # wrap candidate so we can use heapq
                def __init__(self, obj):
                    self.obj = obj

                def __lt__(self, other) -> bool:
                    return compare(self.obj, other.obj) < 0

                def __eq__(self, other) -> bool:
                    return compare(self.obj, other.obj) == 0
                
            
            k = len(parents)
            heap = [] # NOTE: this is min-heap, so we are looking for k maxima
            for candidate in candidates:
                wrapped = Comparable(candidate)
                if len(heap) < k:
                    heapq.heappush(heap, wrapped)
                else:
                    if wrapped > heap[0]:
                        heapq.heapreplace(heap, wrapped)
            ld(f"#comparisons: {log_compare['counter']}, #unique: {len(set(log_compare['pairs']))}")
            
            candidates = [e.obj for e in heap]
            selected_ids = [c['id'] for c in candidates]
            ld(f"{self.count} selection: {selected_ids}")
            write_jsonl(self.cache_file, [selected_ids], append=True)
        
        return candidates
