from pathlib import Path
from jsonschema import validate
from loguru import logger

from aic_nlp_utils.json import read_json, read_jsonl
from ..utils import *

class DatasetLoaderJSONOut:
    def __init__(self, cfg):
        data_path = Path(cfg["data_path"])
        logger.info("data_path: " + str(data_path))
        
        assert data_path.suffix in [".json", ".jsonl"], data_path.suffix
        if data_path.suffix == ".jsonl":
            data = read_jsonl(data_path)
        elif data_path.suffix == ".json":
            data = read_json(data_path)["examples"]
        
        if "unlabeled_data_path" in cfg:
            unlabeled_data_path = Path(cfg["unlabeled_data_path"])
            assert unlabeled_data_path.suffix in [".jsonl"], unlabeled_data_path.suffix
            logger.info("unlabeled_data_path: " + str(unlabeled_data_path))
            unlabeled_data = read_jsonl(unlabeled_data_path)
        else:
            unlabeled_data = None
        
        self.output_schema = read_json(cfg["schema_path"])
        ld("dataset output schema:\n" + jformat(self.output_schema))

        for sample in data:
            validate(sample["answer"], self.output_schema)
            
        trn_size = cfg["trn_size"]
        tst_size = cfg.get("tst_size", len(data)-trn_size)

        data = {"trn": data[:trn_size], "tst": data[trn_size:trn_size+tst_size]}
        if unlabeled_data:
            data["unl"] = unlabeled_data

        split_txt = ", ".join([f"{k}({len(v)})" for k, v in data.items()])
        li(f"dataset loaded: {split_txt}")

        if "sizes" in cfg:
            for split in data.keys():
                if split in cfg["sizes"]:
                    split_size = cfg["sizes"][split]
                    assert split_size <= len(data[split]), (split_size, len(data[split]))
                    data[split] = data[split][:split_size]
            split_txt = ", ".join([f"{k}({len(v)})" for k, v in data.items()])
            li(f"dataset subsampled to: {split_txt}")

        self.data = data
        
            
    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema
