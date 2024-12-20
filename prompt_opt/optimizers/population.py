from loguru import logger
from pathlib import Path
from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl, process_to_jsonl

class Population:
    def __init__(self, archive_jsonl, resume):
        self._archive_jsonl = archive_jsonl
        self._resume = resume
        
        if self._resume:
            done_population = read_jsonl(self._archive_jsonl)
            self._resume_dict = {c["id"]: c for c in done_population}
        else:
            self._resume_dict = {}
        self._population = []
        self._id = 0
        
        
    def is_resume(self):
        return self._resume
    
    
    def save(self):
        write_jsonl(self._archive_jsonl, self._population)
        # logger.debug("SAVE")
    
        
    def add_candidate(self, candidate_fn, save=True):
        self._id += 1
        logger.debug(f"self._id={self._id}")
        logger.debug(f"self._resume_dict={self._resume_dict.keys()}")
        
        if self._id in self._resume_dict:
            logger.debug("skipping...")
            candidate = self._resume_dict[self._id]
            self._population.append(candidate)
            return {"candidate": candidate, "skipped": True}
        else:
            candidate = candidate_fn()
            candidate["id"] = self._id
            self._population.append(candidate)
            if save:
                self.save()
            return {"candidate": candidate, "skipped": False}
        
        
    def add_multiple_candidates(self, candidates_fn, n: int, save=True):
        # all candidates added at once
        assert n > 0
        ids = [id_ for id_ in range(self._id+1, self._id+n+1)]
        
        logger.debug(f"ids={ids}")
        logger.debug(f"self._resume_dict={self._resume_dict.keys()}")

        if ids[0] in self._resume_dict:
            logger.debug(f"skipping {len(ids)}...")
            # if the first id is already done, all remaining must be as well, so they must be in the _resume_dict
            res = []
            for id_ in ids:
                self._id += 1
                candidate = self._resume_dict[id_]
                self._population.append(candidate)
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
                self._population.append(candidate)
                res.append({"candidate": candidate, "skipped": False})
            assert len(res) == n
            if save:
                self.save()
            return res
        
        
    def get_candidates(self):
        return self._population[:]
        