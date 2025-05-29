import numpy as np
from pathlib import Path

from aic_nlp_utils.json import read_json

from prompt_opt.pipeline.utils import source_configurator


def get_out_dir(cfg):
    return str(Path(cfg["root"], cfg["pipeline_name"]).absolute())


def config():
    cpu_model_cfg = {
        "type": "openai",
        "base_url": "http://a01:8881/v1",
        "short": "dseek-lamma8B",
        "name": "dseek-lamma8B",
        "template_dir": "data/templates/agents",
    }

    cfg = {
        "root": "data/labeled_datasets/cro/partial",
        "pipeline_name": "V1",
        "pipeline_note": "CRO datasets V1",
        "seed": 1234,

        "pipeline": [
            # Load source articles
            {
                "id": "load_samples_raw",
                "impl": "prompt_opt.pipeline.import.LoadSamples",
                "file": "data/labeled_datasets/cro/full/cro_full_V1.jsonl",
                "persistent": False,
            },
            # ALL: for editing in the Labeler
            {
                "id": "query_all",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["load_samples_raw"],
                "output_format": "jsonl",
                "output_name": "cro_all_V1",
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "people", "locations", "organizations"],
                "source_cfg": source_configurator(["date", "text", "people", "locations", "organizations"]),
                "answer_keys": ["date", "text", "people", "locations", "organizations", "events"],
            },
            # FULL: for full training
            {
                "id": "query_train",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["load_samples_raw"],
                "output_format": "jsonl",
                "output_name": "cro_full_train_V1",
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text"],
                "source_cfg": source_configurator(["date", "text"]),
                "answer_keys": ["date", "text", "people", "locations", "organizations", "events"],
                "answer_remove": ["events.id", "events.subevents.subid"],
            },
            # PEOPLE
            {
                "id": "query_people",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["load_samples_raw"],
                "output_format": "jsonl",
                "output_name": "people_V1",
                "copy": ["id", "cro_idx"],
                "sources": ["text"],
                "source_cfg": source_configurator(["text"]),
                "answer_keys": ["people"],
                "answer_select": ["people.name", "people.roles"],
            },
            # ORGS
            {
                "id": "query_orgs",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["load_samples_raw"],
                "output_format": "jsonl",
                "output_name": "orgs_V1",
                "copy": ["id", "cro_idx"],
                "sources": ["text"],
                "source_cfg": source_configurator(["text"]),
                "answer_keys": ["organizations"],
                "answer_select": ["organizations.name", "organizations.abbreviation", "organizations.type"],
            },
            # LOCS
            {
                "id": "query_locs",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["load_samples_raw"],
                "output_format": "jsonl",
                "output_name": "locs_V1",
                "copy": ["id", "cro_idx"],
                "sources": ["text"],
                "source_cfg": source_configurator(["text"]),
                "answer_keys": ["locations"],
                "answer_select": ["locations.name", "locations.abbreviation", "locations.type"],
            },
            # EVENTS Only
            {
                "id": "query_event_only",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["load_samples_raw"],
                "output_format": "jsonl",
                "output_name": "events_only_V1",
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "people", "locations", "organizations"],
                "source_cfg": source_configurator(["date", "text", "people", "locations", "organizations"]),
                "answer_keys": ["events"],
                # "answer_select": ["events.event", "events.future"],
                "answer_select": ["events.event"],
            },
            # SUBEVENTS
            {
                "id": "query_subevent",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["load_samples_raw"],
                "output_format": "jsonl",
                "output_name": "subevents_V1",
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "people", "locations", "organizations", "events"],
                "source_cfg": source_configurator(["date", "text", "people", "locations", "organizations", "events_only"]),
                "answer_keys": ["events"],
                "answer_select": ["events.id", "events.subevents.event"],
            },
            # EVENTS & SUBEVENTS
            {
                "id": "query_event",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["load_samples_raw"],
                "output_format": "jsonl",
                "output_name": "events_V1",
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "people", "locations", "organizations"],
                "source_cfg": source_configurator(["date", "text", "people", "locations", "organizations"]),
                "answer_keys": ["events"],
                "answer_select": ["events.event", "events.subevents.event"],
            },
            # EVENTS2PEOPLE
            {
                "id": "query_events2people",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["load_samples_raw"],
                "output_format": "jsonl",
                "output_name": "events2people_V1",
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "people", "events"],
                "source_cfg": source_configurator(["date", "text", "people", "events"]),
                "answer_keys": ["events"],
                "answer_select": ["events.id", "events.people"],
            },
            # EVENTS2ORGS
            {
                "id": "query_events2orgs",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["load_samples_raw"],
                "output_format": "jsonl",
                "output_name": "events2orgs_V1",
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "organizations", "events"],
                "source_cfg": source_configurator(["date", "text", "organizations", "events"]),
                "answer_keys": ["events"],
                "answer_select": ["events.id", "events.orgs"],
            },
            # EVENTS2LOCS
            {
                "id": "query_events2locs",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["load_samples_raw"],
                "output_format": "jsonl",
                "output_name": "events2locs_V1",
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "locations", "events"],
                "source_cfg": source_configurator(["date", "text", "locations", "events"]),
                "answer_keys": ["events"],
                "answer_select": ["events.id", "events.locations"],
            },
            # EVENTS2ATTRIBUTIONS
            {
                "id": "query_events2attributions",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["load_samples_raw"],
                "output_format": "jsonl",
                "output_name": "events2attributions_V1",
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "people", "organizations", "events"],
                "source_cfg": source_configurator(["date", "text", "people", "organizations", "events"]),
                "answer_keys": ["events"],
                "answer_select": ["events.id", "events.attributions"],
            },
            # EVENTS2TEMP
            {
                "id": "query_events2temp",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["load_samples_raw"],
                "output_format": "jsonl",
                "output_name": "events2temp_V1",
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "events"],
                "source_cfg": source_configurator(["date", "text", "events"]),
                "answer_keys": ["events"],
                "answer_select": ["events.id", "events.time_start", "events.time_end", 
                                  "events.time_reported",
                                  "events.subevents.subid",
                                  "events.subevents.time_start", "events.subevents.time_end"],
            },
            
            # {"id": "STOP", "deps": [], "impl": "prompt_opt.pipeline.debug.Stop"},
        ],
    }
    cfg["out_dir"] = get_out_dir(cfg)
    return cfg
