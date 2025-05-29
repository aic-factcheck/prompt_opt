from jsonschema import validate
from loguru import logger

from aic_nlp_utils.json import read_jsonl
from ..utils import *


class DatasetLoaderEventsV3:
    def __init__(self, cfg):
        logger.info("loading DatasetLoaderEventsV3...")

        data_dir = "data/labeled_datasets/events_V3.jsonl"
        logger.info("data_dir: " + data_dir)
        data = read_jsonl(data_dir)
        self.output_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event": {
                        "type": "string",
                    },
                    "people": {"type": "array", "items": {"type": "integer"}},
                    "locations": {"type": "array", "items": {"type": "integer"}},
                    "orgs": {"type": "array", "items": {"type": "integer"}},
                    "attributions": {"type": "array", "items": {"type": "integer"}},
                    "future": {"type": "boolean"},
                },
                "required": ["event", "people", "locations", "orgs", "attributions"],
            },
        }
        for sample in data:
            validate(json.loads(sample["answer"]), self.output_schema)

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


class DatasetLoaderEventsV3Partial:
    def __init__(self, cfg):
        self.target = cfg["target"]
        self.target_element = cfg["target_element"]
        
        logger.info(f"loading DatasetLoaderEventsV3Partial for {self.target}/{self.target_element}...")

        data_dir = f"data/labeled_datasets/events_V3_{self.target}.jsonl"
        logger.info("data_dir: " + data_dir)
        data = read_jsonl(data_dir)

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
        self.output_schema = {
            "type": "object",
            "properties": {
                self.target: {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "event_id": {"type": "integer"},
                            self.target_element: {"type": "array", "items": {"type": "integer"}},
                        },
                    },
                },
            },
            "required": [self.target],
        }
        
        logger.info("dataset output schema:\n" + jformat(self.output_schema))

    def get_data(self):
        return self.data

    def get_output_schema(self):
        return self.output_schema
