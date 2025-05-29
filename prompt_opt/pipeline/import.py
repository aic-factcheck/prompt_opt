from loguru import logger
from pathlib import Path

from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl, process_to_jsonl
from prompt_opt.pipeline.stage import PipelineStage


class LoadSamples(PipelineStage):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.file = Path(self.cfg["file"])
        assert self.file.is_file(), self.file
        self.first = self.cfg.get("first")

    def execute(self, state, load_fn, save_fn):
        output_old, _ = load_fn()
        if output_old:
            if self.first:
                if len(output_old) >= self.first:
                    return output_old[: self.first], None
            else:
                return output_old, None

        output = read_jsonl(self.file)
        if self.first:
            output = output[: self.first]
        logger.info(f"loaded {len(output)} samples")
        return save_fn(output, None)
