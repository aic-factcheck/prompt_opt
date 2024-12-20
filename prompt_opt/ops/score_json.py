from copy import deepcopy
from pathlib import Path

from loguru import logger
import numpy as np

from ..metrics.object_aligner import ObjectAligner
from ..utils import *


class ScoreObjectAligner:
    def __init__(self, cfg):
        logger.info("loading ScoreObjectAligner...")
        self.cfg_op = cfg
        self.metric_schema = self.cfg_op["schema"]
        logger.info('dataset metric schema:\n' + jformat(self.metric_schema))
        self.metric = ObjectAligner(id_="object_aligner_metric", schema=self.metric_schema)
        self.score_key = self.cfg_op["score_key"]
        logger.info(f'dataset metric score_key: {self.score_key}')
    
        
    def get_score_key(self):
        return self.score_key
        
        
    def score_sample(self, gold, pred):
        return self.metric.metric(gold, pred)
    