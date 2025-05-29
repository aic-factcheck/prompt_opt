from loguru import logger
from pathlib import Path

from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl, process_to_jsonl
from prompt_opt.pipeline.stage import PipelineStage


class Stop(PipelineStage):
    def __init__(self, cfg):
        super().__init__(cfg)

    def execute(self, state, load_fn, save_fn):
        logger.debug("STOPPING the Pipeline")
        assert False
