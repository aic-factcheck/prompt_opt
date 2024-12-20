from loguru import logger

from aic_nlp_utils.json import read_jsonl
from ..utils import *


class DatasetLoaderAlignScoreV1:
    def __init__(self, cfg):
        logger.info("loading DatasetLoaderAlignScoreV1...")
        
        data_dir = "data/labeled_datasets/alignscore_bench_val_json_query_named_tags.jsonl"
        logger.info("data_dir: " + data_dir)
        data = read_jsonl(data_dir)
        data = {"trn": data[:6], "dev": data[6:26], "tst": data[26:46], "tst2": data[46:]}
        split_txt = ', '.join([f"{k}({len(v)})" for k, v in data.items()])
        logger.info(f'dataset loaded: {split_txt}')
        
        if "sizes" in cfg:
            for split in data.keys():
                if split in cfg["sizes"]:
                    split_size = cfg["sizes"][split]
                    assert split_size <= len(data[split]), (split_size, len(data[split]))
                    data[split] = data[split][:split_size]
            split_txt = ', '.join([f"{k}({len(v)})" for k, v in data.items()])
            logger.info(f'dataset subsampled to: {split_txt}')

        self.data = data
        self.output_schema = {
            "type": "object",
            "properties": {
                "class": {
                "type": "integer",
                "enum": [0, 1]
                }
            },
            "required": ["class"]
        }
        logger.info('dataset output schema:\n' + jformat(self.output_schema))

        
    def get_data(self):
        return self.data
    
    
    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderAlignScoreMaskedV1:
    def __init__(self, cfg):
        logger.info("loading DatasetLoaderAlignScoreMaskedV1...")
        
        data_dir = "data/labeled_datasets/alignscore_bench_val_json_query_masked_tags.jsonl"
        logger.info("data_dir: " + data_dir)
        data = read_jsonl(data_dir)
        data = {"trn": data[:6], "dev": data[6:26], "tst": data[26:46], "tst2": data[46:]}
        split_txt = ', '.join([f"{k}({len(v)})" for k, v in data.items()])
        logger.info(f'dataset loaded: {split_txt}')
        
        if "sizes" in cfg:
            for split in data.keys():
                if split in cfg["sizes"]:
                    split_size = cfg["sizes"][split]
                    assert split_size <= len(data[split]), (split_size, len(data[split]))
                    data[split] = data[split][:split_size]
            split_txt = ', '.join([f"{k}({len(v)})" for k, v in data.items()])
            logger.info(f'dataset subsampled to: {split_txt}')

        self.data = data
        self.output_schema = {
            "type": "object",
            "properties": {
                "class": {
                "type": "integer",
                "enum": [0, 1]
                }
            },
            "required": ["class"]
        }
        logger.info('dataset output schema:\n' + jformat(self.output_schema))

        
    def get_data(self):
        return self.data
    
    
    def get_output_schema(self):
        return self.output_schema