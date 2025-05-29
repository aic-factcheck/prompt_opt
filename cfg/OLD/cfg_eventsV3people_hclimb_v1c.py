import numpy as np
from pathlib import Path

from prompt_opt.models.model_configs import get_dseek_llama8b, get_dseek_llama70b


def get_exp_dir(cfg):
    return str(Path(cfg["root"], cfg["experiment_name"], f'seed_{cfg["seed"]}').absolute())


def config():
    oa_metric_schema = {
        "type": "object",
        "properties": {
            "people": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "event_id": {"type": "integer", "score": "exact"},
                        "people": {
                            "type": "array",
                            "items": {"type": "integer", "score": "exact"},
                            "order": "align",
                        },
                    },
                    "keyImportance": 0.0,
                    "required": ["event_id", "people"],
                },
                "order": "align",
            },
        },
        "keyImportance": 0.0,
        "required": ["people"],
    }

    cfg = {
        "root": "EXP",
        "experiment_name": "eventsV3people-V1d",
        "experiment_note": """v1d: initial for Deepseek models.""",
        "seed": np.random.randint(10000000),
        "models": {
            # "optimizer": get_dseek_llama8b(),
            "optimizer": get_dseek_llama70b(),
        },
        "dataset_loader": {
            "impl": "prompt_opt.dataset_loader.loader_events-v3.DatasetLoaderEventsV3Partial",
            "target": "people",
            "target_element": "people",
            "merge_trn_and_dev": True,
            # "sizes": {
            #     "trn": 2,
            #     "dev": 2,
            #     "tst": 3
            # }
        },
        "optimizer": {
            "impl": "prompt_opt.optimizers.hill_climber.HillClimber",
            "n_initial": 10,
            # "n_initial": 2,
            "top_k": 2,
            "n_neighbors": 5,
            # "n_neighbors": 2,
            "n_iters": 9,
            # "n_iters": 2,
            "score_key": "oa",
            "xval_trn_and_dev": True,
            "xval_permute": True,
            "select_split": "dev",
            "eval_splits": ["trn", "dev", "tst"],
            "prompt_format": "dseek",
            "ops": {
                "init_op": {
                    "impl": "prompt_opt.ops.init.DSeekInitAllExamplesJSON",
                    "model": "optimizer",
                    "trn_size": 6,
                    "template_init_using_all_examples": "dseek/dseek_init_01_using_all_examples_for_json_output_simple_v1.txt.jinja",
                },
                "mutate_op": {
                    "impl": "prompt_opt.ops.mutate.DSeekImproveJSON",
                    "model": "optimizer",
                    "select_split": "trn",
                    "score_key": "oa",
                    "max_error_samples": 3,
                    "template_improve_first_sample": "dseek/dseek_improve_01_first_sample_v1.txt.jinja",
                    "template_improve_next_sample": "dseek/dseek_improve_02_next_sample_v1.txt.jinja",
                    "template_suggest_changes_for_sample": "dseek/dseek_improve_03_suggest_changes_for_sample_v1.txt.jinja",
                    "template_generate_improved_prompt": "dseek/dseek_improve_04_generate_prompt_v1.txt.jinja",
                },
                "predict_op": {
                    "impl": "prompt_opt.ops.predict.DSeekPredictSteppedJSON",
                    "model": "optimizer",
                    "template_process": "dseek/dseek_predict_01_process_json_schema_v1.txt.jinja",
                    "template_correct": "dseek/dseek_predict_02_correct_json_schema_v1.txt.jinja",
                    "template_failsafe": "dseek/dseek_predict_03_failsafe_json_schema_v1.txt.jinja",
                    "max_corrections": 3,
                },
                "score_ops": [
                    {
                        "impl": "prompt_opt.ops.score_json.ScoreObjectAligner",
                        "score_key": "oa",
                        "schema": oa_metric_schema
                    },
                    # {
                    #     "impl": "prompt_opt.ops.score_json.ModelBasedDSeek",
                    #     "score_key": "mbj",
                    #     "model": "optimizer",
                    #     "template_score": "metrics/dseek/dseek_model_based_metric_01_for_json.txt.jinja",
                    # },
                ],
            },
        },
    }
    cfg["exp_dir"] = get_exp_dir(cfg)
    return cfg
