from pathlib import Path

from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl, process_to_jsonl

from ..utils import *


class Tournament:
    def __init__(self, cfg, exp_path, predictors, rng):
        logger.info("loading Tournament...")
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

    def select(self, population, n_select):
        self.count += 1
        if self.count <= len(self.cache):
            selected_ids = self.cache[self.count-1]
            ld(f"{self.count} cached selection: {selected_ids}")
        else:
            selected_ids = []
            for i in range(n_select):
                pi1, pi2 = self.rng.choice(len(population), 2, replace=False)
                parent1 = population[pi1]
                parent2 = population[pi2]
                parent = self.compare_op.better_candidate(parent1, parent2)
                second = parent1 if parent != parent1 else parent2
                selected_ids.append(parent["id"])
                ld(f"chosen {parent['id']} >= {second['id']}")
            ld(f"{self.count} selection: {selected_ids}")
            write_jsonl(self.cache_file, [selected_ids], append=True)
        
        parents = []
        id2candidate = {c["id"]: c for c in population}
        for sel_id in selected_ids:
            parent = id2candidate[sel_id]
            parents.append(deepcopy(parent))
        return parents
