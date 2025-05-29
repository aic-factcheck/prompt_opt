from pprint import pformat

from loguru import logger
import xmltodict

from prompt_opt.pipeline.stage import PipelineStage

from prompt_opt.utils import pf, ld, lw


class NERQuery(PipelineStage):
    def __init__(self, cfg):
        super().__init__(cfg)
        deps = cfg["deps"]
        assert len(deps) == 1, deps
        self.source_key = deps[0]

    def execute(self, state, load_fn, save_fn):
        def sample_fn(sample, context):
            return {"query": sample["text"]}, None

        input_ = state["store"][self.source_key]
        return self._execute_batch(sample_fn, input_, load_fn, save_fn)


class EventQuery(PipelineStage):
    def __init__(self, cfg):
        super().__init__(cfg)
        deps = cfg["deps"]
        assert len(deps) == 1, deps
        self.source_key = deps[0]

        valid_keys = ["date", "text", "people", "locations", "organizations", "events"]
        self.sources = cfg["sources"] # order is important
        for s in self.sources:
            assert s in valid_keys, s
        self.source_cfg = cfg["source_cfg"]

        self.answer_keys = cfg.get("answer_keys", []) # answer is optional; it is a dictionary with these keys
        for s in self.answer_keys:
            assert s in valid_keys, s

        self.copy = cfg.get("copy", [])
        self.answer_select = cfg.get("answer_select", [])
        self.answer_remove = cfg.get("answer_remove", [])

    def execute(self, state, load_fn, save_fn):
        data = state["store"][self.source_key]
        # logger.debug(pformat(data, sort_dicts=False))

        def collect_answer(sample, path=""):
            # ld("sample", sample)
            if isinstance(sample, dict):
                res = {}
                for k, v in sample.items():
                    cpath = k if path == "" else path + "." + k
                    # ld(cpath, self.answer_remove)
                    if cpath in self.answer_remove:
                        continue
                    if len(self.answer_select) == 0 or any([asel.startswith(cpath) for asel in self.answer_select]):
                        res[k] = collect_answer(v, cpath)
            elif isinstance(sample, list):
                res = []
                for v in sample:
                    res.append(collect_answer(v, path))
            else:
                res = sample
            # ld("res", res)
            return res


        def add_sources(sample, cfg):
            rets = {}
            for key, kcfg in cfg.items():
                if "source" in kcfg:
                    source = kcfg["source"]
                    rets[key] = sample[source]
                else:
                    items = kcfg["items"]
                    element = kcfg["element"]
                    ret = []
                    assert isinstance(sample[key], list), type(sample[key])
                    for sval in sample[key]:
                        r = {}
                        for src, tgt in items.items():
                            if isinstance(tgt, dict):
                                routput = {}
                                routput = add_sources(sval, {src: tgt})
                                r[src] = routput[src]
                            else:
                                r[tgt] = sval[src]
                        ret.append(r)
                    rets[key] = {element: ret}
            # ld(pf(rets))
            return rets
                

        def sample_fn(sample, context):
            output = add_sources(sample, self.source_cfg)

            query = xmltodict.unparse({"root": output}, pretty=True, indent="|")
            query = '\n'.join(line[1:].replace('|', ' ') if line.startswith('|') else line for line in query.splitlines()[2:-1])

            nsample = {k: sample[k] for k in self.copy}
            nsample["query"] = query

            answer = {k: sample[k] for k in self.answer_keys}
            
            if len(self.answer_select) > 0 or len(answer) > 0:
                answer = collect_answer(answer)    
                
            if len(answer) > 0:    
                nsample["answer"] = answer    
            return nsample, None            

        input_ = []
        for i in range(len(data)):
            sample = {}
            for src in self.copy + self.sources + self.answer_keys:
                sample[src] = data[i][src]
            input_.append(sample)

        # logger.debug(pformat(input_, sort_dicts=False))

        return self._execute_batch(sample_fn, input_, load_fn, save_fn)
