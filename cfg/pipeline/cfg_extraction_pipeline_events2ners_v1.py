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
    
    pipeline_name, src_file = f"extraction_pipeline_events2ners_V1_croV2-ds-llama", "data/labeled_datasets/cro/full/cro_full_V2.jsonl"
    # pipeline_name, src_file = f"extraction_pipeline_events2ners_V1_npars5-ds-llama", f"data/extraction_pipeline/cro_data_npars5.jsonl"
    # pipeline_name, src_file = f"extraction_pipeline_events2ners_V1_npars10-ds-llama", f"data/extraction_pipeline/cro_data_npars10.jsonl"
    # pipeline_name, src_file = f"extraction_pipeline_events2ners_V1_npars-min5-ds-llama", f"data/extraction_pipeline/cro_data_npars-min5.jsonl"

    cfg = {
        "root": "data/extraction_pipeline",
        # "pipeline_name": "extraction_pipeline_V1_qwq",
        # "pipeline_name": "extraction_pipeline_V1_ds-llama",
        "pipeline_name": pipeline_name,
        "pipeline_note": "extraction pipeline V1-V2, events2ners (combined references to NERs)",
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
                "file": src_file,
                # "first": 10,
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
                "prompt_text": get_best_prompt("EXP/orgs_V1b", "oa", split="tst", prompt_type="dseek"),
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
            # Query EVENTS
            {
                "id": "query_events",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["append_orgs"],
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "people", "locations", "organizations"],
                "source_cfg": source_configurator(["date", "text", "people", "locations", "organizations"]),
            },
            # Predict EVENTS
            {
                "id": "predict_events",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_events"],
                "predict_op": dseek_predict_op("model1"),
                "prompt_text": get_best_prompt("EXP/events_V2", "mbj", split="tst", prompt_type="dseek"),
                "save_prompt": True,
                "output_schema": read_json("data/schemas/schema_events.json"),
            },
            # Add IDs to EVENTS
            {
                "id": "add_ids_events",
                "impl": "prompt_opt.pipeline.append.IdSequence",
                "deps": ["predict_events"],
                "id_path": "events",
                "id_key": "id",
            },
            # Add IDs to SUBEVENTS
            {
                "id": "add_ids_subevents",
                "impl": "prompt_opt.pipeline.append.IdSequence",
                "deps": ["add_ids_events"],
                "id_path": "events[].subevents",
                "id_key": "subid",
            },
            # Append EVENTS
            {
                "id": "append_events",
                "impl": "prompt_opt.pipeline.append.Append",
                "deps": ["append_orgs", "add_ids_subevents"],
                "target": "append_orgs",
                "append_source": "add_ids_subevents",
                "append_path": "events",
                "append_key": "events",
            },
            # Query EVENTS2NERS
            {
                "id": "query_events2ners",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["append_events"],
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "people", "organizations", "locations", "events"],
                "source_cfg": source_configurator(["date", "text", "people", "organizations", "locations", "events"]),
            },
            # Predict EVENTS2NERS
            {
                "id": "predict_events2ners",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_events2ners"],
                "predict_op": dseek_predict_op("model1"),
                "prompt_text": get_best_prompt("EXP/events2ners_V2", "oa", split="tst", prompt_type="dseek"),
                "save_prompt": True,
                "output_schema": read_json("data/schemas/schema_events2ners.json"),
            },
            # Query EVENTS2TEMP
            {
                "id": "query_events2temp",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["append_events"],
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
                "prompt_text": get_best_prompt("EXP/events2temp_V2", "oa", split="tst", prompt_type="dseek"),
                "save_prompt": True,
                "output_schema": read_json("data/schemas/schema_events2temp.json"),
            },
            # Merge EVENTS2NERS
            {
                "id": "merge_events2ners",
                "impl": "prompt_opt.pipeline.append.Merge",
                "deps": ["append_events", "predict_events2ners"],
                "target": "append_events",
                "source": "predict_events2ners",
                "target_path": "events",
                "source_path": "events",
            },
            # Merge EVENTS2TEMP
            {
                "id": "merge_events2temp",
                "impl": "prompt_opt.pipeline.append.Merge",
                "deps": ["merge_events2ners", "predict_events2temp"],
                "target": "merge_events2ners",
                "source": "predict_events2temp",
                "target_path": "events",
                "source_path": "events",
            },
            # Convert for the Labeler
            {
                "id": "dataset",
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
