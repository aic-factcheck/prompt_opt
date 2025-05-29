import numpy as np
from pathlib import Path

from aic_nlp_utils.json import read_json

from prompt_opt.models.model_configs import get_dseek_llama8b, get_dseek_llama70b, get_qwq32b, get_debug_model, get_qwen3_14b
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

# eval_model = "dseek_llama70b"
eval_model = "qwen3_14b"

pipeline_name, src_file = (
    f"extraction_pipeline_events_complete_V1_croV2-ds-llama_eval-{eval_model}",
    # f"extraction_pipeline_events_complete_V1_croV2-qwen3_14b-{eval_model}",
    "data/labeled_datasets/cro/full/cro_full_V2.jsonl"
)

# pipeline_name, src_file = f"extraction_pipeline_events_complete_V1_npars5-ds-llama", f"data/extraction_pipeline/cro_data_npars5.jsonl"
# pipeline_name, src_file = f"extraction_pipeline_events_complete_V1_npars10-ds-llama", f"data/extraction_pipeline/cro_data_npars10.jsonl"
# pipeline_name, src_file = f"extraction_pipeline_events_complete_V1_npars-min5-ds-llama", f"data/extraction_pipeline/cro_data_npars-min5.jsonl"


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
    
    prompt_people, model_people = get_best_prompt("EXP/people_V1b", "oa", split="tst", prompt_type="dseek"), dseek_predict_op("dseek_llama70b")
    prompt_locs, model_locs = get_best_prompt("EXP/locs_V1", "oa", split="tst", prompt_type="dseek"), dseek_predict_op("dseek_llama70b")
    prompt_orgs, model_orgs = get_best_prompt("EXP/orgs_V1", "oa", split="tst", prompt_type="dseek"), dseek_predict_op("dseek_llama70b")
    
    # prompt_events_complete, model_events_complete = get_best_prompt("EXP/events_complete_V2IU-U", "mbj", split="tst", prompt_type="dseek"), dseek_predict_op("dseek_llama70b")
    prompt_events_complete, model_events_complete = get_best_prompt("EXP/events_complete_V2_EAd", "mbj", split="tst", prompt_type="dseek"), dseek_predict_op("dseek_llama70b") # ds-llama
    # prompt_events_complete, model_events_complete = get_best_prompt("EXP/events_complete_V2_EAe/seed_5834908", "mbj", split="tst", prompt_type="dseek"), dseek_predict_op("qwen3_14b") # qwen3_14b


    prompt_events2people, model_events2people = get_best_prompt("EXP/events2people_V1", "oa", split="tst", prompt_type="dseek"), dseek_predict_op("dseek_llama70b")
    prompt_events2orgs, model_events2orgs = get_best_prompt("EXP/events2orgs_V1", "oa", split="tst", prompt_type="dseek"), dseek_predict_op("dseek_llama70b")
    prompt_events2locs, model_events2locs = get_best_prompt("EXP/events2locs_V1", "oa", split="tst", prompt_type="dseek"), dseek_predict_op("dseek_llama70b")
    prompt_events2attributions, model_events2attributions = get_best_prompt("EXP/events2attributions_V2_mes1", "oa", split="tst", prompt_type="dseek"), dseek_predict_op("dseek_llama70b")

    def pipeline_init():
        return [
            # Load source articles
            {
                "id": "load_samples_raw",
                "impl": "prompt_opt.pipeline.import.LoadSamples",
                "file": src_file,
                # "first": 1,
            },
        ]
        
    def pipeline_ners():
        return [
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
                "predict_op": model_people,
                "prompt_text": prompt_people,
                "save_prompt": True,
                "output_schema": read_json("data/schemas/schema_people.json"),
            },
            # Predict LOCATIONS
            {
                "id": "predict_locs",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_NER"],
                "predict_op": model_locs,
                "prompt_text": prompt_locs,
                "save_prompt": True,
                "output_schema": read_json("data/schemas/schema_locs.json"),
            },
            # Predict ORGANIZATIONS
            {
                "id": "predict_orgs",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_NER"],
                "predict_op": model_orgs,
                "prompt_text": prompt_orgs,
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
            # Convert for the Labeler
            {
                "id": "ners",
                "impl": "prompt_opt.pipeline.transform.Identity",
                "deps": ["append_orgs"],
                "persistent": False,
                "output_format": "jsonl",
            },
        ]
        
        
    def pipeline_events(parent: str):
        return [
            # Query EVENTS COMPLETE
            {
                "id": "query_event_complete",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": [parent],
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text"],
                "source_cfg": source_configurator(["date", "text"]),
            },
            # Predict EVENTS COMPLETE
            {
                "id": "predict_events_complete",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_event_complete"],
                "predict_op": model_events_complete,
                "prompt_text": prompt_events_complete,
                "save_prompt": True,
                "output_schema": read_json("data/schemas/schema_events_complete.json"),
            },
            # Add IDs to SUBEVENTS
            {
                "id": "add_ids_subevents",
                "impl": "prompt_opt.pipeline.append.IdSequence",
                "deps": ["predict_events_complete"],
                "id_path": "events[].subevents",
                "id_key": "subid",
            },
            # Append EVENTS COMPLETE
            {
                "id": "append_events",
                "impl": "prompt_opt.pipeline.append.Append",
                "deps": [parent, "add_ids_subevents"],
                "target": parent,
                "append_source": "add_ids_subevents",
                "append_path": "events",
                "append_key": "events",
            },
            # Query EVENTS2PEOPLE
            {
                "id": "query_events2people",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["append_events"],
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "people", "events"],
                "source_cfg": source_configurator(["date", "text", "people", "events"]),
            },
            # Predict EVENTS2PEOPLE
            {
                "id": "predict_events2people",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_events2people"],
                "predict_op": model_events2people,
                "prompt_text": prompt_events2people,
                "save_prompt": True,
                "output_schema": read_json("data/schemas/schema_events2people.json"),
            },
            # Query EVENTS2ORGS
            {
                "id": "query_events2orgs",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["append_events"],
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "organizations", "events"],
                "source_cfg": source_configurator(["date", "text", "organizations", "events"]),
            },
            # Predict EVENTS2ORGS
            {
                "id": "predict_events2orgs",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_events2orgs"],
                "predict_op": model_events2orgs,
                "prompt_text": prompt_events2orgs,
                "save_prompt": True,
                "output_schema": read_json("data/schemas/schema_events2orgs.json"),
            },
            # Query EVENTS2LOCS
            {
                "id": "query_events2locs",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["append_events"],
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "locations", "events"],
                "source_cfg": source_configurator(["date", "text", "locations", "events"]),
            },
            # Predict EVENTS2LOCS
            {
                "id": "predict_events2locs",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_events2locs"],
                "predict_op": model_events2locs,
                "prompt_text": prompt_events2locs,
                "save_prompt": True,
                "output_schema": read_json("data/schemas/schema_events2locs.json"),
            },
            # Query EVENTS2ATTRIBUTIONS
            {
                "id": "query_events2attributions",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["append_events"],
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "people", "organizations", "events"],
                "source_cfg": source_configurator(["date", "text", "people", "organizations", "events"]),
            },
            # Predict EVENTS2ATTRIBUTIONS
            {
                "id": "predict_events2attributions",
                "impl": "prompt_opt.pipeline.predict.Predict",
                "deps": ["query_events2attributions"],
                "predict_op": model_events2attributions,
                "prompt_text": prompt_events2attributions,
                "save_prompt": True,
                "output_schema": read_json("data/schemas/schema_events2attributions.json"),
            },
            # Merge EVENTS2PEOPLE
            {
                "id": "merge_events2people",
                "impl": "prompt_opt.pipeline.append.Merge",
                "deps": ["append_events", "predict_events2people"],
                "target": "append_events",
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
            # Convert for the Labeler
            {
                "id": "dataset_events_complete",
                "impl": "prompt_opt.pipeline.transform.Identity",
                "deps": ["merge_events2attributions"],
                "output_format": "jsonl",
            },
            # Transform ids to texts for readability/debugging
            {
                "id": "transform_id2text",
                "impl": "prompt_opt.pipeline.transform.Id2Text",
                "deps": ["merge_events2attributions"],
                "target": "merge_events2attributions",
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
        ]
        
    def evaluate_steps(model):
        return [
            
        ]
    

    cfg = {
        "root": "data/extraction_pipeline",
        # "pipeline_name": "extraction_pipeline_V1_qwq",
        # "pipeline_name": "extraction_pipeline_V1_ds-llama",
        "pipeline_name": pipeline_name,
        # "pipeline_name": pipeline_name + "_2",
        "pipeline_note": "extraction pipeline V1, evaluating Qwen3 14b model in context, evaluated using ds llama",
        "seed": 1234,
        "models": {
            # "dseek_llama70b": get_dseek_llama70b(reasoning=True),
            "qwen3_14b": get_qwen3_14b(reasoning=True),
            # "model1": get_qwq32b(reasoning=True),
            # "model1": get_debug_model(),
            # "model1": cpu_model_cfg,
        },
        "pipeline": pipeline_init() + pipeline_ners() + pipeline_events("ners") + [

            {
                "id": "query_id2text",
                "impl": "prompt_opt.pipeline.query.EventQuery",
                "deps": ["transform_id2text"],
                "output_format": "jsonl",
                "copy": ["id", "cro_idx"],
                "sources": ["date", "text", "people", "locations", "organizations"],
                "source_cfg": source_configurator(["date", "text", "people", "locations", "organizations"]),
                "answer_keys": ["events"],
            },
            # Evaluate: PEOPLE
            {
                "id": "evaluate_people",
                "impl": "prompt_opt.pipeline.evaluate.Evaluate",
                "deps": ["load_samples_raw", "merge_events2attributions"],
                "output_format": "jsonl",
                "gold": "load_samples_raw",
                "pred": "merge_events2attributions",
                "match_keys": ["cro_idx"],
                "jmespath": "{people: people[].{name: name, roles: roles}}",
                "score_ops": [
                    {
                        "impl": "prompt_opt.ops.score_json.ScoreObjectAligner",
                        "score_key": "oa_people",
                        "schema": read_json("data/oa/schema_people_V2.json"),
                    },
                    {
                        "impl": "prompt_opt.ops.score_json.ModelBasedDSeek",
                        "score_key": "mbj_people",
                        "model": eval_model,
                        "template_score": "metrics/dseek/dseek_model_based_metric_01_for_json.txt.jinja",
                    }
                ],
            },
            # Evaluate: LOCATIONS
            {
                "id": "evaluate_locs",
                "impl": "prompt_opt.pipeline.evaluate.Evaluate",
                "deps": ["load_samples_raw", "merge_events2attributions"],
                "output_format": "jsonl",
                "gold": "load_samples_raw",
                "pred": "merge_events2attributions",
                "match_keys": ["cro_idx"],
                "jmespath": "{locations: locations[].{name: name, abbreviation: abbreviation, type: type}}",
                "select": ["locations"],
                "score_ops": [
                    {
                        "impl": "prompt_opt.ops.score_json.ScoreObjectAligner",
                        "score_key": "oa_locs",
                        "schema": read_json("data/oa/schema_locs_V2.json"),
                    },
                    {
                        "impl": "prompt_opt.ops.score_json.ModelBasedDSeek",
                        "score_key": "mbj_locs",
                        "model": eval_model,
                        "template_score": "metrics/dseek/dseek_model_based_metric_01_for_json.txt.jinja",
                    }
                ],
            },
            # Evaluate: ORGANIZATIONS
            {
                "id": "evaluate_orgs",
                "impl": "prompt_opt.pipeline.evaluate.Evaluate",
                "deps": ["load_samples_raw", "merge_events2attributions"],
                "output_format": "jsonl",
                "gold": "load_samples_raw",
                "pred": "merge_events2attributions",
                "match_keys": ["cro_idx"],
                "jmespath": "{organizations: organizations[].{name: name, abbreviation: abbreviation, type: type}}",
                "score_ops": [
                    {
                        "impl": "prompt_opt.ops.score_json.ScoreObjectAligner",
                        "score_key": "oa_orgs",
                        "schema": read_json("data/oa/schema_orgs_V2.json"),
                    },
                    {
                        "impl": "prompt_opt.ops.score_json.ModelBasedDSeek",
                        "score_key": "mbj_orgs",
                        "model": eval_model,
                        "template_score": "metrics/dseek/dseek_model_based_metric_01_for_json.txt.jinja",
                    }
                ],
            },
            # Evaluate: EVENT descriptions only 
            {
                "id": "evaluate_event_desc",
                "impl": "prompt_opt.pipeline.evaluate.Evaluate",
                "deps": ["load_samples_raw", "merge_events2attributions"],
                "output_format": "jsonl",
                "gold": "load_samples_raw",
                "pred": "merge_events2attributions",
                "match_keys": ["cro_idx"],
                "jmespath": "events[].event",
                "score_ops": [
                    {
                        "impl": "prompt_opt.ops.score_json.ModelBasedDSeek",
                        "score_key": "mbj_evaluate_event_desc",
                        "model": eval_model,
                        "template_score": "metrics/dseek/dseek_model_based_metric_01_for_json.txt.jinja",
                    }
                ],
            },
            # Evaluate: EVENT+SUBEVENT descriptions only 
            {
                "id": "evaluate_event-subevent_desc",
                "impl": "prompt_opt.pipeline.evaluate.Evaluate",
                "deps": ["load_samples_raw", "merge_events2attributions"],
                "output_format": "jsonl",
                "gold": "load_samples_raw",
                "pred": "merge_events2attributions",
                "match_keys": ["cro_idx"],
                "jmespath": "events[].{event: event, subevents: subevents[].event}",
                "score_ops": [
                    {
                        "impl": "prompt_opt.ops.score_json.ModelBasedDSeek",
                        "score_key": "mbj_event-subevent_desc",
                        "model": eval_model,
                        "template_score": "metrics/dseek/dseek_model_based_metric_01_for_json.txt.jinja",
                    }
                ],
            },
            # Evaluate: EVENT+SUBEVENT descriptions + time span
            {
                "id": "evaluate_event-subevent_desc+span",
                "impl": "prompt_opt.pipeline.evaluate.Evaluate",
                "deps": ["load_samples_raw", "merge_events2attributions"],
                "output_format": "jsonl",
                "gold": "load_samples_raw",
                "pred": "merge_events2attributions",
                "match_keys": ["cro_idx"],
                "jmespath": "events[].{event: event, time_start: time_start, time_end: time_end, subevents: subevents[].event}",
                "score_ops": [
                    {
                        "impl": "prompt_opt.ops.score_json.ModelBasedDSeek",
                        "score_key": "mbj_event-subevent_desc+span",
                        "model": eval_model,
                        "template_score": "metrics/dseek/dseek_model_based_metric_01_for_json.txt.jinja",
                    }
                ],
            },
            # Evaluate: EVENT+SUBEVENT descriptions + time rep
            {
                "id": "evaluate_event-subevent_desc+rep",
                "impl": "prompt_opt.pipeline.evaluate.Evaluate",
                "deps": ["load_samples_raw", "merge_events2attributions"],
                "output_format": "jsonl",
                "gold": "load_samples_raw",
                "pred": "merge_events2attributions",
                "match_keys": ["cro_idx"],
                "jmespath": "events[].{event: event, time_reported: time_reported, subevents: subevents[].event}",

                "score_ops": [
                    {
                        "impl": "prompt_opt.ops.score_json.ModelBasedDSeek",
                        "score_key": "mbj_event-subevent_desc+rep",
                        "model": eval_model,
                        "template_score": "metrics/dseek/dseek_model_based_metric_01_for_json.txt.jinja",
                    }
                ],
            },
            # Evaluate: EVENT+SUBEVENT descriptions + time all
            {
                "id": "evaluate_event-subevent_desc+time",
                "impl": "prompt_opt.pipeline.evaluate.Evaluate",
                "deps": ["load_samples_raw", "merge_events2attributions"],
                "output_format": "jsonl",
                "gold": "load_samples_raw",
                "pred": "merge_events2attributions",
                "match_keys": ["cro_idx"],
                "jmespath": "events[].{event: event, time_start: time_start, time_end: time_end, time_reported: time_reported, subevents: subevents[].{event: event, time_start: time_start, time_end: time_end}}",
                "score_ops": [
                    {
                        "impl": "prompt_opt.ops.score_json.ModelBasedDSeek",
                        "score_key": "mbj_event-subevent_desc+time",
                        "model": "dseek_llama70b",
                        "template_score": "metrics/dseek/dseek_model_based_metric_01_for_json.txt.jinja",
                    }
                ],
            },
            # Evaluate: Merge
            {
                "id": "merge_evaluate",
                "impl": "prompt_opt.pipeline.append.DeepMerge",
                "deps": ["evaluate_people", "evaluate_locs", "evaluate_orgs", 
                         "evaluate_event-subevent_desc", "evaluate_event-subevent_desc+span", 
                         "evaluate_event-subevent_desc+rep", "evaluate_event-subevent_desc+time"],
                "output_format": "jsonl",
                "target": "evaluate_people"
            },
            # {"id": "STOP", "deps": [], "impl": "prompt_opt.pipeline.debug.Stop"},
        ],
    }
    cfg["out_dir"] = get_out_dir(cfg)
    return cfg
