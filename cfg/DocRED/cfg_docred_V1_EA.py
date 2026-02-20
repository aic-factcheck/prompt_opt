from ctypes import ArgumentError
import numpy as np
from pathlib import Path

from aic_nlp_utils.json import read_json, read_jsonl

from prompt_opt.models.model_configs import *


def get_exp_dir(cfg):
    return str(Path(cfg["root"], cfg["experiment_name"], f'seed_{cfg["seed"]}').absolute())


def dataset_loader_factory(dataset, trn_scale=1):
    return {
            "impl": "prompt_opt.dataset_loader.loader_common.DatasetLoaderJSONOut",
            "data_path": f"data/DocRED/{dataset}.json",
            "schema_path": f"data/DocRED/schemas/schema_{dataset}.json",
            "trn_size": 16*trn_scale,
            "tst_size": 24
        }
    
        
def score_oa_factory(dataset):
    schema = f"data/DocRED/oa/schema_{dataset}.json"
    return {
        "impl": "prompt_opt.ops.score_json.ScoreObjectAligner",
        "score_key": "oa",
        "schema": read_json(schema)
    }
    

def config():
    dataset = "docred_relations_hints"
    mutate_max_error_samples = 3
    # mutate_max_error_samples = 1
    # trn_scale = 1
    trn_scale = 4
    cfg = {
        "root": "EXP",
        "experiment_name": f"{dataset}_V1_EA_ts4",
        "experiment_note": f"""{dataset}_V1_EA_ts4: initial, DSeekDirectImproveJSON, trn_scale=4""",
        "seed": np.random.randint(10000000),
        "models": {
            # "optimizer": get_gptoss_120b(),
            # "optimizer": get_qwen3_32b(),
            # "optimizer": get_qwen3_next_80b_A3b_thinking(),
            "optimizer": get_qwen3_next_80b_A3b_instruct(),
        },
        "dataset_loader": dataset_loader_factory(dataset, trn_scale=trn_scale),
        "optimizer": {
            "impl": "prompt_opt.optimizers.ea.EvolutionaryAlgorithm",
            "n_initial": 20,
            # "n_initial": 100000, # init only
            "n_iters": 9,
            "xval_trn_and_dev": True,
            "xval_permute": True,
            "eval_splits": ["trn", "dev", "tst"],
            "prompt_format": "dseek",
            "ops": {
                "init_op": {
                    "impl": "prompt_opt.ops.init.DSeekInitAllExamplesJSON",
                    "model": "optimizer",
                    "trn_size": 6 * trn_scale,
                    "template_init_using_all_examples": "dseek/dseek_init_01_using_all_examples_for_json_output_simple_v2.txt.jinja",
                },
                "mutate_op": {
                    "impl": "prompt_opt.ops.mutate.DSeekImproveJSON",
                    "model": "optimizer",
                    "select_split": "trn",
                    "score_key": "oa",
                    # "score_key": "mbj",
                    "max_error_samples": mutate_max_error_samples,
                    # MOVE ELSEWHERE
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
                            {
                                "impl": "prompt_opt.ops.compare.CompareScore",
                                "select_split": "trn",
                                "score_key": "oa"
                                # "score_key": "mbj"
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
                                "score_key": "oa"
                                # "score_key": "mbj"
                            }
                        ],
                    }
                },
                "predict_op": {
                    "impl": "prompt_opt.ops.predict.PredictReasoningJSON",
                    "model": "optimizer",
                    "template_process": "dseek/dseek_predict_01_process_json_schema_v2.txt.jinja",
                },
                "score_ops": [
                    score_oa_factory(dataset),
                    # {
                    #     "impl": "prompt_opt.ops.score_json.ModelBasedDSeek",
                    #     "score_key": "mbj",
                    #     # "model": "scorer",
                    #     "model": "optimizer",
                    #     "template_score": "metrics/dseek/dseek_model_based_metric_02_for_json.txt.jinja",
                    # },
                ],
            },
        },
    }
    cfg["exp_dir"] = get_exp_dir(cfg)
    return cfg
