import numpy as np
from pathlib import Path

from aic_nlp_utils.json import read_json

from prompt_opt.models.model_configs import get_dseek_llama8b, get_dseek_llama70b, get_qwq32b, get_debug_model


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
        "template_process": "dseek/dseek_predict_01_process_json_schema_v1.txt.jinja"
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

    cfg = {
        "root": "data/extraction_pipeline",
        # "pipeline_name": "extraction_pipeline_V1_qwq",
        "pipeline_name": "extraction_pipeline_V1_ds-llama",
        "pipeline_note": "extraction pipeline V1",
        "seed": 1234,
        "models": {
            # "model1": get_dseek_llama70b(reasoning=True),
            # "model1": get_qwq32b(reasoning=True),
            "model1": get_debug_model(),
            # "model1": cpu_model_cfg,
        },
        "pipeline": [
            # Load source articles
            {
                "id": "load_samples_raw",
                "impl": "prompt_opt.pipeline.import.LoadSamples",
                "file": "data/extraction_pipeline/cro_data_npars5.jsonl",
                "first": 10,
            },
            # Prepare queries for NER
            {"id": "query_NER", "impl": "prompt_opt.pipeline.query.NERQuery", "deps": ["load_samples_raw"]},
            # Extract and append People
            {
                "id": "predict_people",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_NER"],
                "predict_op": dseek_predict_op("model1"),
                "prompt": "data/pipeline/prompts/prompt_peopleV2-V7-mbj.md",
                "output_schema": read_json("data/schemas/schema_people.json"),
            },
            {
                "id": "append_people",
                "impl": "prompt_opt.pipeline.append.Append",
                "deps": ["load_samples_raw", "predict_people"],
                "target": "load_samples_raw",
                "append_source": "predict_people",
                "append_path": "answer",
                "append_key": "people",
            },
            # Extract and append Locations
            {
                "id": "predict_locs",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_NER"],
                "predict_op": dseek_predict_op("model1"),
                "prompt": "data/pipeline/prompts/prompt_locs-V7-mbj.md",
                "output_schema": read_json("data/schemas/schema_locs.json"),
            },
            {
                "id": "append_locs",
                "impl": "prompt_opt.pipeline.append.Append",
                "deps": ["append_people", "predict_locs"],
                "target": "append_people",
                "append_source": "predict_locs",
                "append_path": "answer",
                "append_key": "locations",
            },
            # Extract and append Organizations
            {
                "id": "predict_orgs",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_NER"],
                "predict_op": dseek_predict_op("model1"),
                "prompt": "data/pipeline/prompts/prompt_orgs-V7-mbj.md",
                "output_schema": read_json("data/schemas/schema_orgs.json"),
            },
            {
                "id": "append_orgs",
                "impl": "prompt_opt.pipeline.append.Append",
                "deps": ["append_locs", "predict_orgs"],
                "target": "append_locs",
                "append_source": "predict_orgs",
                "append_path": "answer",
                "append_key": "organizations",
            },
            # Prepare, extract and append Events (inc. subevents)
            {
                "id": "query_event",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["append_orgs"],
                "sources": ["date", "text", "people", "locations", "organizations"],
            },
            {
                "id": "predict_events",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_event"],
                "predict_op": dseek_predict_op("model1",),
                "prompt": "data/pipeline/prompts/prompt_eventsV4events-mbj.md",
                "output_schema": read_json("data/schemas/schema_events.json"),
            },
            {
                "id": "append_subevent_ids",
                "impl": "prompt_opt.pipeline.append.IdSequence",
                "deps": ["predict_events"],
                "id_path": "answer[].subevents",
                "id_key": "subid"
            },
            {
                "id": "append_events",
                "impl": "prompt_opt.pipeline.append.Append",
                "deps": ["append_orgs", "append_subevent_ids"],
                "target": "append_orgs",
                "append_source": "append_subevent_ids",
                "append_path": "answer",
                "append_key": "events",
            },
            # Prepare, extract, append People for each event
            {
                "id": "query_events2people",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["append_events"],
                "sources": ["date", "text", "people", "events"],
            },
            {
                "id": "predict_events2people",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_events2people"],
                "predict_op": dseek_predict_op("model1"),
                "prompt": "data/pipeline/prompts/prompt_eventsV3people-oa.md",
                "output_schema": read_json("data/schemas/schema_events2people.json"),
            },
            {
                "id": "merge_events2people",
                "impl": "prompt_opt.pipeline.append.Merge",
                "deps": ["append_events", "predict_events2people"],
                "target": "append_events",
                "source": "predict_events2people",
                "target_path": "events",
                "source_path": "answer.people",
                "source_rename": {"event_id": "id"},
            },
            
            # Prepare, extract, append Locations for each event
            {
                "id": "query_events2locs",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["append_events"],
                "sources": ["date", "text", "locations", "events"],
            },
            {
                "id": "predict_events2locs",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_events2locs"],
                "predict_op": dseek_predict_op("model1"),
                "prompt": "data/pipeline/prompts/prompt_eventsV3locs-oa.md",
                "output_schema": read_json("data/schemas/schema_events2locs.json"),
            },
            {
                "id": "merge_events2locs",
                "impl": "prompt_opt.pipeline.append.Merge",
                "deps": ["merge_events2people", "predict_events2locs"],
                "target": "merge_events2people",
                "source": "predict_events2locs",
                "target_path": "events",
                "source_path": "answer.locations",
                "source_rename": {"event_id": "id"},
            },
            # Prepare, extract, append Organizations for each event
            {
                "id": "query_events2orgs",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["append_events"],
                "sources": ["date", "text", "organizations", "events"],
            },
            {
                "id": "predict_events2orgs",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_events2orgs"],
                "predict_op": dseek_predict_op("model1"),
                "prompt": "data/pipeline/prompts/prompt_eventsV3orgs-oa.md",
                "output_schema": read_json("data/schemas/schema_events2orgs.json"),
            },
            {
                "id": "merge_events2orgs",
                "impl": "prompt_opt.pipeline.append.Merge",
                "deps": ["merge_events2locs", "predict_events2orgs"],
                "target": "merge_events2locs",
                "source": "predict_events2orgs",
                "target_path": "events",
                "source_path": "answer.orgs",
                "source_rename": {"event_id": "id"},
            },
            # Prepare, extract, append Attributions (People and/or Orgs reporting the event) for each event
            {
                "id": "query_events2attributions",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["append_events"],
                "sources": ["date", "text", "people", "organizations", "events"],
            },
            {
                "id": "predict_events2attributions",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_events2attributions"],
                "predict_op": dseek_predict_op("model1"),
                "prompt": "data/pipeline/prompts/prompt_eventsV3attributions-oa.md",
                "output_schema": read_json("data/schemas/schema_events2attributions.json"),
            },
            {
                "id": "merge_events2attributions",
                "impl": "prompt_opt.pipeline.append.Merge",
                "deps": ["merge_events2orgs", "predict_events2attributions"],
                "target": "merge_events2orgs",
                "source": "predict_events2attributions",
                "target_path": "events",
                "source_path": "answer.attributions",
                "source_rename": {"event_id": "id"},
            },
            # Prepare, extract, append Temporal Validity
            {
                "id": "query_events2temp",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["merge_events2attributions"],
                "sources": ["date", "text", "events"],
            },
            {
                "id": "predict_events2temp",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_events2temp"],
                "predict_op": dseek_predict_op("model1"),
                "prompt": "data/pipeline/prompts/prompt_eventsV4temp-oa.md",
                "output_schema": read_json("data/schemas/schema_events2temp.json"),
            },
            {
                "id": "merge_events2temp",
                "impl": "prompt_opt.pipeline.append.Merge",
                "deps": ["merge_events2attributions", "predict_events2temp"],
                "target": "merge_events2attributions",
                "source": "predict_events2temp",
                "target_path": "events",
                "source_path": "answer"
            },
            # Convert ids to texts for readability/debugging
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
                "sources": ["date", "text", "people", "locations", "organizations"],
                "answer_keys":  ["events"]
            },
            # {"id": "STOP", "deps": [], "impl": "prompt_opt.pipeline.debug.Stop"},
        ],
    }
    cfg["out_dir"] = get_out_dir(cfg)
    return cfg
