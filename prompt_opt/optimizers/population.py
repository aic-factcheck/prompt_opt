from loguru import logger
from pathlib import Path
from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl, process_to_jsonl
from prompt_opt.utils import ld, li, lw, le

from ..optimizers.split_tracker import SplitTracker

class Population:
    def __init__(self, archive_jsonl, split_tracker: SplitTracker, resume: bool):
        self._archive_jsonl = archive_jsonl
        self.split_tracker = split_tracker
        self._resume = resume
        
        if self._resume:
            done_population = read_jsonl(self._archive_jsonl)
            write_jsonl(str(self._archive_jsonl) + "~", done_population) # immediatelly backup
            self._resume_dict = {c["id"]: c for c in done_population}
        else:
            self._resume_dict = {}
        self._population = {}
        self._id = 0
        
        
    def is_resume(self):
        return self._resume
    
    
    def save(self):
        write_jsonl(self._archive_jsonl, self.get_candidates())
        # ld("SAVE")
                
        
    def _post_hook(self, candidate):
        # run after candidate ID is added
        # TODO: remove after checking that resume works
        # self.split_tracker.register_candidate_indices(candidate["id"], candidate["split2indices"])
        pass
        
        
    def add_candidate(self, candidate_fn, save=True):
        self._id += 1
        ld(f"self._id={self._id}")
        ld(f"self._resume_dict={self._resume_dict.keys()}")
        
        if self._id in self._resume_dict:
            ld("skipping...")
            candidate = self._resume_dict[self._id]
            self._post_hook(candidate)
            self._population[candidate["id"]] = candidate
            return {"candidate": candidate, "skipped": True}
        else:
            candidate = candidate_fn()
            candidate["id"] = self._id
            self._post_hook(candidate)
            self._population[candidate["id"]] = candidate
            if save:
                self.save()
            return {"candidate": candidate, "skipped": False}
        
        
    def add_multiple_candidates(self, candidates_fn, n: int, save=True):
        # all candidates added at once
        assert n > 0
        ids = list(range(self._id+1, self._id+n+1))
        
        ld(f"ids={ids}")
        ld(f"self._resume_dict={self._resume_dict.keys()}")

        if ids[0] in self._resume_dict:
            ld(f"skipping {len(ids)}...")
            # if the first id is already done, all remaining must be as well, so they must be in the _resume_dict
            ids_check = [id_ in self._resume_dict for id_ in ids]
            assert all(ids_check), ids_check
            
            res = []
            for id_ in ids:
                self._id += 1
                candidate = self._resume_dict[id_]
                self._post_hook(candidate)
                self._population[candidate["id"]] = candidate
                res.append({"candidate": candidate, "skipped": True})
            assert len(res) == n
            return res
        else:
            candidates = candidates_fn()
            assert len(candidates) == n, f"expected {n} candidates, got {len(candidates)}!"
            res = []
            assert len(candidates) == n, (len(candidates), n)
            for candidate in candidates:
                self._id += 1
                candidate["id"] = self._id
                self._post_hook(candidate)
                self._population[candidate["id"]] = candidate
                res.append({"candidate": candidate, "skipped": False})
            assert len(res) == n
            if save:
                self.save()
            return res
        
        
    def get_candidates(self):
        return list(self._population.values())
    
    
    def get_candidate_by_id(self, id_):
        return self._population[id_]
