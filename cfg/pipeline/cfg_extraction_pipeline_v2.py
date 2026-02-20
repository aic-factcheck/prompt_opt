import numpy as np
from pathlib import Path

from aic_nlp_utils.json import read_json

from prompt_opt.models.model_configs import get_openai_gpt_5_2
from prompt_opt.pipeline.utils import get_best_prompt, source_configurator


def get_out_dir(cfg):
    return str(Path(cfg["root"], cfg["pipeline_name"]).absolute())


def openai_predict_op(model_name):
    return {
        "impl": "prompt_opt.ops.predict.PredictReasoningJSON",
        "model": model_name,
        "template_process": "dseek/dseek_predict_01_process_json_schema_v2.txt.jinja",
    }


def config():
    pipeline_name, src_file = "extraction_pipeline_V2_croV2-gpt52", "data/CRO/cro/full/cro_full_V2.jsonl"

    cfg = {
        "root": "data/extraction_pipeline",
        "pipeline_name": pipeline_name,
        "pipeline_note": "extraction pipeline V2, 3-prompt (NERs, events, events2all)",
        "seed": 1234,
        "models": {
            "model1": get_openai_gpt_5_2(),
        },
        "pipeline": [
            # Load source articles
            {
                "id": "load_samples_raw",
                "impl": "prompt_opt.pipeline.import.LoadSamples",
                "file": src_file,
                "first": 10,
            },
            # Query for NER prediction (text only)
            {
                "id": "query_NER",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["load_samples_raw"],
                "copy": ["id", "cro_idx"],
                "sources": ["text"],
                "source_cfg": source_configurator(["text"]),
            },
            # Prompt 1: Predict all NERs (people, locations, organizations)
            {
                "id": "predict_ners",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_NER"],
                "predict_op": openai_predict_op("model1"),
                "prompt_text": get_best_prompt("EXP/CRO_ners_V2", "oa", split="tst", prompt_type="dseek"),
                "save_prompt": True,
                "output_schema": read_json("data/CRO/schemas/schema_ners.json"),
            },
            # Append PEOPLE from NER prediction
            {
                "id": "append_people",
                "impl": "prompt_opt.pipeline.append.Append",
                "deps": ["load_samples_raw", "predict_ners"],
                "target": "load_samples_raw",
                "append_source": "predict_ners",
                "append_path": "people",
                "append_key": "people",
            },
            # Append LOCATIONS from NER prediction
            {
                "id": "append_locs",
                "impl": "prompt_opt.pipeline.append.Append",
                "deps": ["append_people", "predict_ners"],
                "target": "append_people",
                "append_source": "predict_ners",
                "append_path": "locations",
                "append_key": "locations",
            },
            # Append ORGANIZATIONS from NER prediction
            {
                "id": "append_orgs",
                "impl": "prompt_opt.pipeline.append.Append",
                "deps": ["append_locs", "predict_ners"],
                "target": "append_locs",
                "append_source": "predict_ners",
                "append_path": "organizations",
                "append_key": "organizations",
            },
            # Query for event prediction (text + NERs)
            {
                "id": "query_events",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["append_orgs"],
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "people", "locations", "organizations"],
                "source_cfg": source_configurator(["date", "text", "people", "locations", "organizations"]),
            },
            # Prompt 2: Predict events + subevents (text only)
            {
                "id": "predict_events",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_events"],
                "predict_op": openai_predict_op("model1"),
                "prompt_text": get_best_prompt("EXP/CRO_events_V2", "mbj", split="tst", prompt_type="dseek"),
                "save_prompt": True,
                "output_schema": read_json("data/CRO/schemas/schema_events.json"),
            },
            # Add sequential IDs to events
            {
                "id": "add_ids_events",
                "impl": "prompt_opt.pipeline.append.IdSequence",
                "deps": ["predict_events"],
                "id_path": "events",
                "id_key": "id",
            },
            # Add sequential IDs to subevents
            {
                "id": "add_ids_subevents",
                "impl": "prompt_opt.pipeline.append.IdSequence",
                "deps": ["add_ids_events"],
                "id_path": "events[].subevents",
                "id_key": "subid",
            },
            # Append events to accumulated data
            {
                "id": "append_events",
                "impl": "prompt_opt.pipeline.append.Append",
                "deps": ["append_orgs", "add_ids_subevents"],
                "target": "append_orgs",
                "append_source": "add_ids_subevents",
                "append_path": "events",
                "append_key": "events",
            },
            # Query for events2all prediction (text + NERs + events)
            {
                "id": "query_events2all",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["append_events"],
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "people", "organizations", "locations", "events"],
                "source_cfg": source_configurator(["date", "text", "people", "organizations", "locations", "events"]),
            },
            # Prompt 3: Predict temporal + NER linking + future
            {
                "id": "predict_events2all",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_events2all"],
                "predict_op": openai_predict_op("model1"),
                "prompt_text": get_best_prompt("EXP/CRO_events2all_V2", "oa", split="tst", prompt_type="dseek"),
                "save_prompt": True,
                "output_schema": read_json("data/CRO/schemas/schema_events2all.json"),
            },
            # Merge events2all predictions into events
            {
                "id": "merge_events2all",
                "impl": "prompt_opt.pipeline.append.Merge",
                "deps": ["append_events", "predict_events2all"],
                "target": "append_events",
                "source": "predict_events2all",
                "target_path": "events",
                "source_path": "events",
            },
            # Output as JSONL
            {
                "id": "dataset",
                "impl": "prompt_opt.pipeline.transform.Identity",
                "deps": ["merge_events2all"],
                "output_format": "jsonl"
            },
            # Transform IDs to text for readability/debugging
            {
                "id": "transform_id2text",
                "impl": "prompt_opt.pipeline.transform.Id2Text",
                "deps": ["merge_events2all"],
                "target": "merge_events2all",
                "transform": [
                    {"source": "people", "target": "events[].people", "format": "{name}(P{id})"},
                    {
                        "source": "locations",
                        "target": "events[].locations",
                        "format": "{abbreviation|name}/{type}(L{id})",
                    },
                    {
                        "source": "organizations",
                        "target": "events[].orgs",
                        "format": "{abbreviation|name}/{type}(O{id})",
                    },
                    {"source": "people", "target": "events[].attributions", "format": "{name}(P{id})"},
                    {
                        "source": "organizations",
                        "target": "events[].attributions",
                        "format": "{abbreviation|name}/{type}(O{id})",
                    },
                ],
            },
            {
                "id": "query_id2text",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["transform_id2text"],
                "output_format": "jsonl",
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "people", "locations", "organizations"],
                "source_cfg": source_configurator(["date", "text", "people", "locations", "organizations"]),
                "answer_keys": ["events"]
            },
        ],
    }
    cfg["out_dir"] = get_out_dir(cfg)
    return cfg
