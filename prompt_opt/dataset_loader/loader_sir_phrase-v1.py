from loguru import logger

from aic_nlp_utils.json import read_jsonl
from ..utils import *


class DatasetLoaderSiRPhraseV1:
    def __init__(self, cfg):
        logger.info("loading DatasetLoaderSiRPhraseV1...")
        # logger.info("cfg:\n" + jformat(cfg))
        
        data_dir = "data/labeled_datasets/sir1.0_triple_manual_phrases.jsonl"
        logger.info("data_dir: " + data_dir)
        data = read_jsonl(data_dir)
        
        merge_trn_and_dev = cfg.get("merge_trn_and_dev", False)
        if merge_trn_and_dev:
            data = {"trn": data[:15], "tst": data[15:25], "tst2": data[25:]}
        else:
            data = {"trn": data[:5], "dev": data[5:15], "tst": data[15:25], "tst2": data[25:]}
            
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
            "type": "array",
            "items": {"type": "string"},
        }
        logger.info('dataset output schema:\n' + jformat(self.output_schema))

        
    def get_data(self):
        return self.data
    
    
    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderSiRPhraseEncapsulatedV1:
    def __init__(self, cfg):
        logger.info("loading DatasetLoaderSiRPhraseEncapsulatedV1...")
        # logger.info("cfg:\n" + jformat(cfg))
        
        data_dir = "data/labeled_datasets/sir1.0_triple_manual_phrases_enc.jsonl"
        logger.info("data_dir: " + data_dir)
        data = read_jsonl(data_dir)
        data = {"trn": data[:5], "dev": data[5:15], "tst": data[15:25], "tst2": data[25:]}
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
                "array": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                }
            },
            "required": ["array"],
            "additionalProperties": False
        }
        logger.info('dataset output schema:\n' + jformat(self.output_schema))

        
    def get_data(self):
        return self.data
    
    
    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderSiRPhraseOccurencesV1:
    def __init__(self, cfg):
        logger.info("loading DatasetLoaderSiRPhraseOccurencesV1...")
        # logger.info("cfg:\n" + jformat(cfg))
        
        data_dir = "data/labeled_datasets/sir1.0_triple_manual_phrases_occurences.jsonl"
        logger.info("data_dir: " + data_dir)
        data = read_jsonl(data_dir)
        
        merge_trn_and_dev = cfg.get("merge_trn_and_dev", False)
        if merge_trn_and_dev:
            data = {"trn": data[:15], "tst": data[15:25], "tst2": data[25:]}
        else:
            data = {"trn": data[:5], "dev": data[5:15], "tst": data[15:25], "tst2": data[25:]}
            
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
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "span": {
                        "type": "string"
                    },
                    "occurence": {
                        "type": "integer"
                    }
                },
                "required": ["span", "occurence"],
                "additionalProperties": False
            }
        }
        logger.info('dataset output schema:\n' + jformat(self.output_schema))

        
    def get_data(self):
        return self.data
    
    
    def get_output_schema(self):
        return self.output_schema