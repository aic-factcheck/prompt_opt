from pathlib import Path
from jsonschema import validate
from loguru import logger

from aic_nlp_utils.json import read_json, read_jsonl
from ..utils import *

class DatasetLoaderCRO:
    def __init__(self, cfg):
        data_path = self._get_data_path()
        logger.info("data_path: " + data_path)
        data = read_jsonl(data_path)
        
        unlabeled_data_path = self._get_unlabeled_data_path()
        logger.info("unlabeled_data_path: " + unlabeled_data_path)
        unlabeled_data = read_jsonl(unlabeled_data_path)
        
        assert len(data) == 30, f"strange: data len != 30: {len(data)}"
        self.output_schema = read_json(self._get_schema_path())
        ld("dataset output schema:\n" + jformat(self.output_schema))

        for sample in data:
            validate(sample["answer"], self.output_schema)

        merge_trn_and_dev = cfg.get("merge_trn_and_dev", False)
        if merge_trn_and_dev:
            data = {"trn": data[:16], "tst": data[16:30], "unl": unlabeled_data}
        else:
            data = {"trn": data[:8], "dev": data[8:16], "tst": data[16:30], "unl": unlabeled_data}

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
        
            
    def _get_data_path(self) -> str:
        raise NotImplementedError("NOT IMPLEMENTED, use child class.")
        
        
    def _get_unlabeled_data_path(self):
        return "data/labeled_datasets/cro/partial/V2/cro_unlabeled_V2.jsonl"
        
        
    def _get_schema_path(self) -> str:
        raise NotImplementedError("NOT IMPLEMENTED, use child class.")
        
        
    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema


class DatasetLoaderPeopleV2(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V2/people_V1.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_people.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderOrgsV2(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V2/orgs_V1.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_orgs.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderLocsV2(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V2/locs_V1.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_locs.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderNERSV2(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V2/ners_V1.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_ners.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderEventsV2(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V2/events_V2.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_events.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderEventsOnlyV2(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V2/events_only_V2.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_events_only.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderEventsCompleteV2(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V2/events_complete_V2.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_events_complete.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderSubeventsV2(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V2/subevents_V2.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_subevents.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema


class DatasetLoaderEvents2PeopleV2(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V2/events2people_V2.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_events2people.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderEvents2OrgsV2(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V2/events2orgs_V2.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_events2orgs.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderEvents2LocsV2(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V2/events2locs_V2.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_events2locs.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderEvents2AttributionsV2(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V2/events2attributions_V2.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_events2attributions.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema

class DatasetLoaderEvents2NERsV2(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V2/events2ners_V2.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_events2ners.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderEvents2TempV2(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V2/events2temp_V2.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_events2temp.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderCROFullV2(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V2/cro_full_train_V2.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_cro_full_answer.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema