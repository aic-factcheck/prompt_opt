from jsonschema import validate
from loguru import logger

from aic_nlp_utils.json import read_json, read_jsonl
from ..utils import *


class DatasetLoaderEventTemporalValidityV1:
    def __init__(self, cfg):
        logger.info("loading DatasetLoaderEventTemporalValidityV1...")

        data_dir = "data/labeled_datasets/event_temp_val_V1.jsonl"
        
        logger.info("data_dir: " + data_dir)
        data = read_jsonl(data_dir)
        self.output_schema = read_json("data/schemas/schema_events2temp.json")

        for sample in data:
            validate(sample["answer"], self.output_schema)

        merge_trn_and_dev = cfg.get("merge_trn_and_dev", False)
        if merge_trn_and_dev:
            data = {"trn": data[:16], "tst": data[16:30]}
        else:
            data = {"trn": data[:8], "dev": data[8:16], "tst": data[16:30]}

        split_txt = ", ".join([f"{k}({len(v)})" for k, v in data.items()])
        logger.info(f"dataset loaded: {split_txt}")

        if "sizes" in cfg:
            for split in data.keys():
                if split in cfg["sizes"]:
                    split_size = cfg["sizes"][split]
                    assert split_size <= len(data[split]), (split_size, len(data[split]))
                    data[split] = data[split][:split_size]
            split_txt = ", ".join([f"{k}({len(v)})" for k, v in data.items()])
            logger.info(f"dataset subsampled to: {split_txt}")

        self.data = data
        logger.info("dataset output schema:\n" + jformat(self.output_schema))

    def get_data(self):
        return self.data

    def get_output_schema(self):
        return self.output_schema
