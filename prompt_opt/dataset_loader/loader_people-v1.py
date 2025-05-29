from loguru import logger

from aic_nlp_utils.json import read_json, read_jsonl
from ..utils import *


class DatasetLoaderPeopleRolesV1:
    def __init__(self, cfg):
        logger.info("loading DatasetLoaderPeopleRolesV1...")
        
        data_dir = "data/labeled_datasets/people_roles.jsonl"
        logger.info("data_dir: " + data_dir)
        data = read_jsonl(data_dir)
        
        merge_trn_and_dev = cfg.get("merge_trn_and_dev", False)
        if merge_trn_and_dev:
            data = {"trn": data[:12], "tst": data[12:20]}
        else:
            data = {"trn": data[:4], "dev": data[4:12], "tst": data[12:20]}
            
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
        self.output_schema = read_json("data/schemas/schema_people.json")

        logger.info('dataset output schema:\n' + jformat(self.output_schema))

        
    def get_data(self):
        return self.data
    
    
    def get_output_schema(self):
        return self.output_schema
    