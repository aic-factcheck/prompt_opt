from jsonschema import validate
from loguru import logger

from aic_nlp_utils.json import read_json, read_jsonl
from ..utils import *

class DatasetLoaderCRO:
    def __init__(self, cfg):
        data_path = self._get_data_path()
        logger.info("data_path: " + data_path)
        data = read_jsonl(data_path)
        assert len(data) == 30, f"strange: data len != 30: {len(data)}"
        self.output_schema = read_json(self._get_schema_path())
        ld("dataset output schema:\n" + jformat(self.output_schema))

        for sample in data:
            validate(sample["answer"], self.output_schema)

        merge_trn_and_dev = cfg.get("merge_trn_and_dev", False)
        if merge_trn_and_dev:
            data = {"trn": data[:16], "tst": data[16:30]}
        else:
            data = {"trn": data[:8], "dev": data[8:16], "tst": data[16:30]}

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
        
            
    def _get_data_path(self):
        assert False, "NOT IMPLEMENTED, use child class."
        
        
    def _get_schema_path(self):
        assert False, "NOT IMPLEMENTED, use child class."
        
        
    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema


class DatasetLoaderPeopleV1(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V1/people_V1.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_people.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderOrgsV1(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V1/orgs_V1.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_orgs.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderLocsV1(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V1/locs_V1.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_locs.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderEventsV1(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V1/events_V1.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_events.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderEventsOnlyV1(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V1/events_only_V1.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_events_only.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderSubeventsV1(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V1/subevents_V1.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_subevents.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema


class DatasetLoaderEvents2PeopleV1(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V1/events2people_V1.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_events2people.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderEvents2OrgsV1(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V1/events2orgs_V1.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_events2orgs.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderEvents2LocsV1(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V1/events2locs_V1.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_events2locs.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderEvents2AttributionsV1(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V1/events2attributions_V1.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_events2attributions.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderEvents2TempV1(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V1/events2temp_V1.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_events2temp.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema
    
    
class DatasetLoaderCROFullV1(DatasetLoaderCRO):
    def __init__(self, cfg):
        super().__init__(cfg)


    def _get_data_path(self):
        return "data/labeled_datasets/cro/partial/V1/cro_full_train_V1.jsonl"
        
        
    def _get_schema_path(self):
        return "data/schemas/schema_cro_full.json"
        

    def get_data(self):
        return self.data


    def get_output_schema(self):
        return self.output_schema