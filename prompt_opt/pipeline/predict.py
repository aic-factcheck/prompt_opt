from loguru import logger
from pathlib import Path

from aic_nlp_utils.files import create_parent_dir

from prompt_opt.pipeline.stage import PipelineStage
from prompt_opt.utils import get_class_instance_by_config


class Predict(PipelineStage):
    def __init__(self, cfg):
        super().__init__(cfg)
        deps = cfg["deps"]
        assert len(deps) == 1, deps
        self.source_key = deps[0]
        
        if "prompt_path" in cfg:
            assert "prompt_text" not in cfg
            self.prompt = Path(cfg["prompt_path"]).read_text()
        elif "prompt_text" in cfg:
            assert "prompt_path" not in cfg
            self.prompt = cfg["prompt_text"]
            
        self.output_schema = cfg["output_schema"]
        save_prompt = cfg.get("save_prompt", False)
        if save_prompt:
            prompt_file = Path(cfg["out_dir"], "prompts", f"{cfg['id']}.txt")
            create_parent_dir(prompt_file)
            prompt_file.write_text(self.prompt)
    
    
    def execute(self, state, load_fn, save_fn):
        def context_init_fn():
            predict_op = get_class_instance_by_config(
                self.cfg["predict_op"],
                exp_path=self.cfg["out_dir"],
                predictors=state["predictors"])
            return {"predict_op": predict_op}
        
        def sample_fn(sample, context):
            predict_op = context["predict_op"]
            response, messages = predict_op.predict(
                prompt=self.prompt, 
                query=sample["query"],
                output_schema=self.output_schema,
                examples=None)
            output = response["pred"]
            del response["pred"]
            meta = {**response, "messages": messages}
            return output, meta
        
        input_ = state["store"][self.source_key]
        return self._execute_batch(sample_fn, input_, load_fn, save_fn, context_init_fn=context_init_fn)
