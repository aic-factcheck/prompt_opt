import numpy as np
from pathlib import Path

from aic_nlp_utils.json import read_json

from prompt_opt.models.model_configs import get_dseek_llama8b, get_dseek_llama70b, get_qwq32b, get_debug_model
from prompt_opt.pipeline.utils import get_best_prompt, source_configurator


def get_out_dir(cfg):
    return str(Path(cfg["root"], cfg["pipeline_name"]).absolute())


def dseek_predict_op_basic(model_name, max_corrections=3):
    return {
        "impl": "prompt_opt.ops.predict.DSeekPredictSteppedJSON",
        "model": model_name,
        "template_process": "dseek/dseek_predict_01_process_json_schema_v1.txt.jinja",
        "template_correct": "dseek/dseek_predict_02_correct_json_schema_v1.txt.jinja",
        "template_failsafe": "dseek/dseek_predict_03_failsafe_json_schema_v1.txt.jinja",
        "max_corrections": max_corrections,
    }


def dseek_predict_op_reasoning(model_name):
    return {
        "impl": "prompt_opt.ops.predict.PredictReasoningJSON",
        "model": model_name,
        "template_process": "dseek/dseek_predict_01_process_json_schema_v1.txt.jinja",
    }


def dseek_predict_op(model_name):
    # return dseek_predict_op_basic(model_name, max_corrections=1)
    return dseek_predict_op_reasoning(model_name)


def config():
    cpu_model_cfg = {
        "type": "openai",
        "base_url": "http://a01:8881/v1",
        "short": "dseek-lamma8B",
        "name": "dseek-lamma8B",
        "template_dir": "data/templates/agents",
    }
    
    npars = "npars5"
    # npars = "npars10"
    # npars = "npars-min5"

    cfg = {
        "root": "data/extraction_pipeline",
        # "pipeline_name": "extraction_pipeline_V1_qwq",
        # "pipeline_name": "extraction_pipeline_V1_ds-llama",
        "pipeline_name": f"extraction_pipeline_events_only_V1_{npars}-ds-llama",
        "pipeline_note": "extraction pipeline V1",
        "seed": 1234,
        "models": {
            "model1": get_dseek_llama70b(reasoning=True),
            # "model1": get_qwq32b(reasoning=True),
            # "model1": get_debug_model(),
            # "model1": cpu_model_cfg,
        },
        "pipeline": [
            # Load source articles
            {
                "id": "load_samples_raw",
                "impl": "prompt_opt.pipeline.import.LoadSamples",
                "file": f"data/extraction_pipeline/cro_data_{npars}.jsonl",
                "first": 10,
            },
            # Query NER
            {
                "id": "query_NER",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["load_samples_raw"],
                "copy": ["id", "cro_idx"],
                "sources": ["text"],
                "source_cfg": source_configurator(["text"]),
            },
            # Predict PEOPLE
            {
                "id": "predict_people",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_NER"],
                "predict_op": dseek_predict_op("model1"),
                "prompt_text": get_best_prompt("EXP/people_V1b", "oa", split="tst", prompt_type="dseek"),
                "save_prompt": True,
                "output_schema": read_json("data/schemas/schema_people.json"),
            },
            # Predict LOCATIONS
            {
                "id": "predict_locs",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_NER"],
                "predict_op": dseek_predict_op("model1"),
                "prompt_text": get_best_prompt("EXP/locs_V1", "oa", split="tst", prompt_type="dseek"),
                "save_prompt": True,
                "output_schema": read_json("data/schemas/schema_locs.json"),
            },
            # Predict ORGANIZATIONS
            {
                "id": "predict_orgs",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_NER"],
                "predict_op": dseek_predict_op("model1"),
                "prompt_text": get_best_prompt("EXP/orgs_V1", "oa", split="tst", prompt_type="dseek"),
                "save_prompt": True,
                "output_schema": read_json("data/schemas/schema_orgs.json"),
            },
            # Append PEOPLE
            {
                "id": "append_people",
                "impl": "prompt_opt.pipeline.append.Append",
                "deps": ["load_samples_raw", "predict_people"],
                "target": "load_samples_raw",
                "append_source": "predict_people",
                "append_path": "people",
                "append_key": "people",
            },
            # Append LOCATIONS
            {
                "id": "append_locs",
                "impl": "prompt_opt.pipeline.append.Append",
                "deps": ["append_people", "predict_locs"],
                "target": "append_people",
                "append_source": "predict_locs",
                "append_path": "locations",
                "append_key": "locations",
            },
            # Append ORGANIZATIONS
            {
                "id": "append_orgs",
                "impl": "prompt_opt.pipeline.append.Append",
                "deps": ["append_locs", "predict_orgs"],
                "target": "append_locs",
                "append_source": "predict_orgs",
                "append_path": "organizations",
                "append_key": "organizations",
            },
            # Query EVENTS ONLY
            {
                "id": "query_events_only",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["append_orgs"],
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "people", "locations", "organizations"],
                "source_cfg": source_configurator(["date", "text", "people", "locations", "organizations"]),
            },
            # Predict EVENTS ONLY
            {
                "id": "predict_events_only",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_events_only"],
                "predict_op": dseek_predict_op("model1"),
                "prompt_text": get_best_prompt("EXP/events_only_V1b", "mbj", split="tst", prompt_type="dseek"),
                "save_prompt": True,
                "output_schema": read_json("data/schemas/schema_events_only.json"),
            },
            # Append EVENTS ONLY
            {
                "id": "append_events",
                "impl": "prompt_opt.pipeline.append.Append",
                "deps": ["append_orgs", "predict_events_only"],
                "target": "append_orgs",
                "append_source": "predict_events_only",
                "append_path": "events",
                "append_key": "events",
            },
            # Query SUBEVENTS
            {
                "id": "query_subevents",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["append_events"],
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "people", "locations", "organizations", "events"],
                "source_cfg": source_configurator(
                    ["date", "text", "people", "locations", "organizations", "events_only"]
                ),
            },
            # Predict SUBEVENTS
            {
                "id": "predict_subevents",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_subevents"],
                "predict_op": dseek_predict_op("model1"),
                "prompt_text": get_best_prompt("EXP/subevents_V1b", "mbj", split="tst", prompt_type="dseek"),
                "save_prompt": True,
                "output_schema": read_json("data/schemas/schema_subevents.json"),
            },
            # Add IDs to SUBEVENTS
            {
                "id": "add_ids_subevents",
                "impl": "prompt_opt.pipeline.append.IdSequence",
                "deps": ["predict_subevents"],
                "id_path": "events[].subevents",
                "id_key": "subid",
            },
            # Merge SUBEVENTS
            {
                "id": "merge_subevents",
                "impl": "prompt_opt.pipeline.append.Merge",
                "deps": ["append_events", "add_ids_subevents"],
                "target": "append_events",
                "source": "add_ids_subevents",
                "target_path": "events",
                "source_path": "events",
            },
            # Query EVENTS2PEOPLE
            {
                "id": "query_events2people",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["merge_subevents"],
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "people", "events"],
                "source_cfg": source_configurator(["date", "text", "people", "events"]),
            },
            # Predict EVENTS2PEOPLE
            {
                "id": "predict_events2people",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_events2people"],
                "predict_op": dseek_predict_op("model1"),
                "prompt_text": get_best_prompt("EXP/events2people_V1", "oa", split="tst", prompt_type="dseek"),
                "save_prompt": True,
                "output_schema": read_json("data/schemas/schema_events2people.json"),
            },
            # Query EVENTS2ORGS
            {
                "id": "query_events2orgs",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["merge_subevents"],
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "organizations", "events"],
                "source_cfg": source_configurator(["date", "text", "organizations", "events"])
            },
            # Predict EVENTS2ORGS
            {
                "id": "predict_events2orgs",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_events2orgs"],
                "predict_op": dseek_predict_op("model1"),
                "prompt_text": get_best_prompt("EXP/events2orgs_V1", "oa", split="tst", prompt_type="dseek"),
                "save_prompt": True,
                "output_schema": read_json("data/schemas/schema_events2orgs.json"),
            },
            # Query EVENTS2LOCS
            {
                "id": "query_events2locs",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["merge_subevents"],
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "locations", "events"],
                "source_cfg": source_configurator(["date", "text", "locations", "events"])
            },
            # Predict EVENTS2LOCS
            {
                "id": "predict_events2locs",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_events2locs"],
                "predict_op": dseek_predict_op("model1"),
                "prompt_text": get_best_prompt("EXP/events2locs_V1", "oa", split="tst", prompt_type="dseek"),
                "save_prompt": True,
                "output_schema": read_json("data/schemas/schema_events2locs.json"),
            },
            # Query EVENTS2ATTRIBUTIONS
            {
                "id": "query_events2attributions",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["merge_subevents"],
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "people", "organizations", "events"],
                "source_cfg": source_configurator(["date", "text", "people", "organizations", "events"])
            },
            # Predict EVENTS2ATTRIBUTIONS
            {
                "id": "predict_events2attributions",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_events2attributions"],
                "predict_op": dseek_predict_op("model1"),
                "prompt_text": get_best_prompt("EXP/events2attributions_V1", "oa", split="tst", prompt_type="dseek"),
                "save_prompt": True,
                "output_schema": read_json("data/schemas/schema_events2attributions.json"),
            },
            # Query EVENTS2TEMP
            {
                "id": "query_events2temp",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["merge_subevents"],
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "events"],
                "source_cfg": source_configurator(["date", "text", "events"])
            },
            # Predict EVENTS2TEMP
            {
                "id": "predict_events2temp",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_events2temp"],
                "predict_op": dseek_predict_op("model1"),
                "prompt_text": get_best_prompt("EXP/events2temp_V1", "oa", split="tst", prompt_type="dseek"),
                "save_prompt": True,
                "output_schema": read_json("data/schemas/schema_events2temp.json"),
            },
            # Merge EVENTS2PEOPLE
            {
                "id": "merge_events2people",
                "impl": "prompt_opt.pipeline.append.Merge",
                "deps": ["merge_subevents", "predict_events2people"],
                "target": "merge_subevents",
                "source": "predict_events2people",
                "target_path": "events",
                "source_path": "events",
            },
            # Merge EVENTS2LOCS
            {
                "id": "merge_events2locs",
                "impl": "prompt_opt.pipeline.append.Merge",
                "deps": ["merge_events2people", "predict_events2locs"],
                "target": "merge_events2people",
                "source": "predict_events2locs",
                "target_path": "events",
                "source_path": "events",
            },
            # Merge EVENTS2ORGS
            {
                "id": "merge_events2orgs",
                "impl": "prompt_opt.pipeline.append.Merge",
                "deps": ["merge_events2locs", "predict_events2orgs"],
                "target": "merge_events2locs",
                "source": "predict_events2orgs",
                "target_path": "events",
                "source_path": "events",
            },
            # Merge EVENTS2ATTRIBUTIONS
            {
                "id": "merge_events2attributions",
                "impl": "prompt_opt.pipeline.append.Merge",
                "deps": ["merge_events2orgs", "predict_events2attributions"],
                "target": "merge_events2orgs",
                "source": "predict_events2attributions",
                "target_path": "events",
                "source_path": "events",
            },
            # Merge EVENTS2TEMP
            {
                "id": "merge_events2temp",
                "impl": "prompt_opt.pipeline.append.Merge",
                "deps": ["merge_events2attributions", "predict_events2temp"],
                "target": "merge_events2attributions",
                "source": "predict_events2temp",
                "target_path": "events",
                "source_path": "events",
            },
            # Convert for the Labeler
            {
                "id": "dataset_events_only",
                "impl": "prompt_opt.pipeline.transform.Identity",
                "deps": ["merge_events2temp"],
                "output_format": "jsonl"
            },
            # Transform ids to texts for readability/debugging
            {
                "id": "transform_id2text",
                "impl": "prompt_opt.pipeline.transform.Id2Text",
                "deps": ["merge_events2temp"],
                "target": "merge_events2temp",
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
                "answer_keys":  ["events"]
            },
            # {"id": "STOP", "deps": [], "impl": "prompt_opt.pipeline.debug.Stop"},
        ],
    }
    cfg["out_dir"] = get_out_dir(cfg)
    return cfg
