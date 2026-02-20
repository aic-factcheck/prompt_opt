import numpy as np
from pathlib import Path

from aic_nlp_utils.json import read_json, read_jsonl

from prompt_opt.models.model_configs import *


def get_exp_dir(cfg):
    return str(Path(cfg["root"], cfg["experiment_name"], f'seed_{cfg["seed"]}').absolute())

def config():
    experiment_name = "CRO_events_V2"
    mutate_max_error_samples = 3
    
    cfg = {
        "root": "EXP",
        "experiment_name": f"{experiment_name}",
        "experiment_note": f"""{experiment_name}: basic; initial, now running on new (propriety) models""",
        "seed": np.random.randint(10000000),
        "models": {
            "optimizer": get_openai_gpt_5_2(),
            "target": get_openai_gpt_5_2(),
        },
        "dataset_loader": {
            "impl": "prompt_opt.dataset_loader.loader_cro-v2.DatasetLoaderEventsV2",
            "merge_trn_and_dev": True,
        },
        "optimizer": {
            "impl": "prompt_opt.optimizers.ea.EvolutionaryAlgorithm",
            "n_initial": 40,
            # "n_iters": 99,
            "n_iters": 49,
            "xval_trn_and_dev": True,
            "xval_permute": True,
            "eval_splits": ["trn", "dev", "tst"],
            "prompt_format": "dseek",
            "predict_jobs": 0,
            "score_jobs": 0,
            "ops": {
                "init_op": {
                    "impl": "prompt_opt.ops.init.DSeekInitAllExamplesJSON",
                    "model": "optimizer",
                    # "trn_size": 4,
                    "trn_size": 6,
                    # MOVE ELSEWHERE
                    "template_init_using_all_examples": "dseek/dseek_init_01_using_all_examples_for_json_output_simple_v2.txt.jinja",
                },
                "mutate_op": {
                    "impl": "prompt_opt.ops.mutate.DSeekImproveJSON",
                    "model": "optimizer",
                    "select_split": "trn",
                    "score_key": "mbj",
                    "max_error_samples": mutate_max_error_samples,
                    "template_improve_first_sample": "dseek/dseek_improve_01_first_sample_v2.txt.jinja",
                    "template_improve_next_sample": "dseek/dseek_improve_02_next_sample_v2.txt.jinja",
                    "template_suggest_changes_for_sample": "dseek/dseek_improve_03_suggest_changes_for_sample_v2.txt.jinja",
                    "template_generate_improved_prompt": (
                        "dseek/dseek_improve_04_generate_prompt_v2.txt.jinja"
                        if mutate_max_error_samples > 1
                        else "dseek/dseek_improve_04_generate_prompt_single_example_v2.txt.jinja"
                    ),
                },
                "select_op": {
                    "impl": "prompt_opt.ops.select.Tournament",
                    "cache": "select_cache.jsonl",
                    "compare_op": {
                        "impl": "prompt_opt.ops.compare.DebugCompare",
                        "log": "compare_log_select.jsonl",
                        "ops": [
                            # {
                            #     "impl": "prompt_opt.ops.compare.DSeekCompareJSON",
                            #     "select_split": "trn",
                            #     "model": "optimizer",
                            #     "template_compare": "dseekdir/dseekdir_compare_01_single_example_v1.txt.jinja",
                            # },
                            {
                                "impl": "prompt_opt.ops.compare.CompareScore",
                                "select_split": "trn",
                                "score_key": "mbj"
                            }
                        ],
                    },
                },
                "reproduce_op": {
                    "impl": "prompt_opt.ops.reproduce.ReproduceMutateOnly",
                },
                "reduce_op": {
                    "impl": "prompt_opt.ops.reduce.ReduceBest",
                    "cache": "reduce_cache.jsonl",
                    "compare_op": {
                        "impl": "prompt_opt.ops.compare.DebugCompare",
                        "log": "compare_log_reduce.jsonl",
                        "ops": [
                            # {
                            #     "impl": "prompt_opt.ops.compare.DSeekCompareJSON",
                            #     "select_split": "trn",
                            #     "model": "optimizer",
                            #     "template_compare": "dseekdir/dseekdir_compare_01_single_example_v1.txt.jinja",
                            # },
                            {
                                "impl": "prompt_opt.ops.compare.CompareScore",
                                "select_split": "trn",
                                "score_key": "mbj"
                            }
                        ],
                    },
                },
                "predict_op": {
                    "impl": "prompt_opt.ops.predict.PredictReasoningJSON",
                    "model": "target",
                    # "model": "optimizer",
                    "template_process": "dseek/dseek_predict_01_process_json_schema_v2.txt.jinja",
                },
                "score_ops": [
                    {
                        "impl": "prompt_opt.ops.score_json.ModelBasedDSeek",
                        "score_key": "mbj",
                        "model": "optimizer",
                        "template_score": "metrics/dseek/dseek_model_based_metric_02_for_json.txt.jinja",
                    },
                ],
            },
        },
    }
    cfg["exp_dir"] = get_exp_dir(cfg)
    return cfg
