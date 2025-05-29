from copy import deepcopy
from pathlib import Path
from pprint import pformat

from deepmerge import Merger
import numpy as np

from prompt_opt.pipeline.stage import PipelineStage
from prompt_opt.pipeline.utils import collect_key_values, descend, descend_and_set, select
from prompt_opt.utils import get_class_instance_by_config, pf, ld, lw

class Append(PipelineStage):
    # TODO: maybe use just for ID generation, then Merge! See IdSequence below....
    def __init__(self, cfg):
        super().__init__(cfg)
        self.target = cfg["target"]
        self.append_source = cfg["append_source"]
        self.append_path = cfg["append_path"]
        self.append_key = cfg["append_key"]
        
        self.generate_ids = cfg.get("generate_ids", True)
        self.rng = np.random.RandomState(cfg["seed"])
        
    
    def execute(self, state, load_fn, save_fn):
        
        def sample_fn(sample, context, rng):
            out, app = sample["out"], sample["app"]
            # ld(f"out={pf(out)}")
            # ld(f"app={pf(app)}")
            
            if self.generate_ids:
                ex_ids = collect_key_values(out, "id")
                all_ids = list(set(range(10000)) - ex_ids)
                assert len(all_ids) >= len(app), "not enough ids, all taken!"
                ids = rng.choice(all_ids, size=len(app), replace=False)
                app = [{"id": int(id_), **e} for id_, e in zip(ids, app)]
                
            if self.append_key:
                out[self.append_key] = app
            else:
                out.update(app)
                        
            return out, None
        
        
        store = state["store"]
        output = deepcopy(store[self.target])
        append_samples = [descend(src, self.append_path) for src in store[self.append_source]]
        
        input_ = []
        for o, a in zip(output, append_samples):
            input_.append({"out": o, "app": a})
        
        return self._execute_batch(sample_fn, input_, load_fn, save_fn, rng=self.rng)


class Merge(PipelineStage):
    # merges by aligning source to target
    def __init__(self, cfg):
        super().__init__(cfg)
        self.target = cfg["target"]
        self.source = cfg["source"]
        self.target_path = cfg.get("target_path")
        self.source_path = cfg.get("source_path")
        self.source_rename = cfg.get("source_rename", {})
        self.rev_source_rename = {v: k for k, v in self.source_rename.items()}
        
    
    def execute(self, state, load_fn, save_fn):
        
        def common_keys(tgt_lst, src_lst, path):
            assert "." not in path, f'NOT YET implemented: {path}'
            # common keys or alignment
            # get only keys with non-iterable values for simplicity
            tkeys = [k for k, v in tgt_lst[0].items() if not isinstance(v, list) and not isinstance(v, dict)]
            skeys = set([self.source_rename.get(k, k) for k, v in src_lst[0].items() if not isinstance(v, list) and not isinstance(v, dict)])
            common = set(tkeys).intersection(skeys)
            assert len(common) > 0
            return [k for k in tkeys if k in common]
            
            
        def merge_lst(tgt_lst, src_lst, path):
            # 1) align src with tgt based on common key values, 2) merge all source items to the target
            if len(tgt_lst) == 0 or len(src_lst) == 0:
                return tgt_lst
            ckeys = common_keys(tgt_lst, src_lst, path)
            srckeys2src = {}
            for src in src_lst:
                comp_key = tuple([src[self.rev_source_rename.get(ck, ck)] for ck in ckeys]) # composite key
                srckeys2src[comp_key] = src
            merged_lst = []
            for tgt in tgt_lst:
                comp_key = tuple([tgt[ck] for ck in ckeys])
                if comp_key in srckeys2src:
                    src = srckeys2src[comp_key]
                    merged = merge(tgt, src, path)
                    merged_lst.append(merged)
                else:
                    lw("missing record for composite key:", comp_key)
            return merged_lst
        
        
        def merge_dct(tgt_dct, src_dct, path):
            tgt_dct = deepcopy(tgt_dct)
            # ld("tgt_dct", pf(tgt_dct))
            # ld("src_dct", pf(src_dct))
            for sk, sv in src_dct.items():
                sk = self.source_rename.get(sk, sk)
                if sk in tgt_dct:
                    tgt_dct[sk] = merge(tgt_dct[sk], sv, sk if path =="" else path + "." + sk)
                else:
                    tgt_dct[sk] = sv
            return tgt_dct
            
            
        def merge(tgt, src, path):
            assert type(tgt) == type(src)
            if isinstance(tgt, list):
                return merge_lst(tgt, src, path)
            elif isinstance(tgt, dict):
                return merge_dct(tgt, src, path)
            else:
                assert tgt == src, (tgt, src)
                return deepcopy(tgt)
                

        def sample_fn(sample, context):
            target, source = sample["target"], sample["source"]
            tgt = descend(target, self.target_path)
            src = descend(source, self.source_path)
            # logger.debug(f"tgt={pf(tgt)}")
            # logger.debug(f"src={pf(src)}")
            nsample = merge(tgt, src, "")
            descend_and_set(target, self.target_path, nsample)
            return target, None
        
        
        store = state["store"]
        assert len(store[self.target]) == len(store[self.source])
        samples = []
        for t, s in zip(store[self.target], store[self.source]):
            samples.append({"target": t, "source": s})
        
        return self._execute_batch(sample_fn, samples, load_fn, save_fn)


class IdSequence(PipelineStage):
    def __init__(self, cfg):
        super().__init__(cfg)
        deps = cfg["deps"]
        assert len(deps) == 1, deps
        self.source_key = deps[0]
        self.id_path = cfg["id_path"]
        self.id_key = cfg["id_key"]
    
    
    def execute(self, state, load_fn, save_fn):
        
        def sample_fn(sample, context):
            # ld(f"sample={pf(sample)}")
            for subevents in select(sample, self.id_path):
                for idx in range(len(subevents)):
                    se = {self.id_key: idx+1, **subevents[idx]}
                    subevents[idx] = se
            return sample, None

            
        store = state["store"]
        input_ = store[self.source_key]
        
        return self._execute_batch(sample_fn, input_, load_fn, save_fn)


class DeepMerge(PipelineStage):
    # merges by aligning source to target
    def __init__(self, cfg):
        super().__init__(cfg)
        self.target = cfg["target"]
        self.sources = cfg["deps"]
        self.sources.remove(self.target)
        
    
    def execute(self, state, load_fn, save_fn):

        def sample_fn(sample, context):
            target, sources = sample["target"], sample["sources"]
            merger = Merger(
                # pass in a list of tuple, with the
                # strategies you are looking to apply
                # to each type.
                [
                    (list, ["override"]),
                    (dict, ["merge"]),
                    (set, ["union"])
                ],
                # next, choose the fallback strategies,
                # applied to all other types:
                ["override"],
                # finally, choose the strategies in
                # the case where the types conflict:
                ["override"]
            )

            for source in sources:
                merger.merge(target, source)
            return target, None
        
        
        store = state["store"]
        samples = []
        for i in range(len(store[self.target])):
            target = store[self.target][i]
            sources = []
            for s in self.sources:
                sources.append(store[s][i])
            samples.append({"target": target, "sources": sources})
        ld(f"target example\n{pf(samples[0]['target'])}")
        ld(f"source example\n{pf(samples[0]['sources'][0])}")
        
        return self._execute_batch(sample_fn, samples, load_fn, save_fn)