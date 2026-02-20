import numpy as np
from pathlib import Path

from aic_nlp_utils.json import read_json, read_jsonl

from prompt_opt.models.model_configs import *


def get_exp_dir(cfg):
    return str(Path(cfg["root"], cfg["experiment_name"], f'seed_{cfg["seed"]}').absolute())

def config():
    dataset_name, experiment_name, dataset_version = "cite_basic", "Cbsc_V3", "v1" 
    mutate_max_error_samples = 3
    # mutate_max_error_samples = 1
    
    # oa_schema_path = f"data/cite/{dataset_version}/oa/schema_{dataset_name}.json"
    oa_schema_path = f"data/cite/{dataset_version}/oa/schema_{dataset_name}_threshold9.json"
    
    cfg = {
        "root": "EXP",
        "experiment_name": f"{experiment_name}",
        "experiment_note": f"""{experiment_name}: basic; too few data samples: no tst, less iterations, Jaro threshold 0.9 for matching strings""",
        "seed": np.random.randint(10000000),
        "models": {
            # "optimizer": get_openai_gpt_5(),
            "optimizer": get_openai_gpt_5_2(),
            # "optimizer": get_gptoss_20b(),
            # "optimizer": get_gptoss_120b(),
            # "target": get_gptoss_120b(),
            "target": get_openai_gpt_5_2(),
            # "target": get_gptoss_20b(),
        },
        "dataset_loader": {
            "impl": "prompt_opt.dataset_loader.loader_common.DatasetLoaderJSONOut",
            "data_path": f"data/cite/{dataset_version}/{dataset_name}.json",
            "schema_path": f"data/cite/{dataset_version}/schemas/schema_{dataset_name}.json",
            # "trn_size": 7,
            # "tst_size": 6,
            "trn_size": 13,
            "tst_size": 0,
        },
        "optimizer": {
            "impl": "prompt_opt.optimizers.ea.EvolutionaryAlgorithm",
            "n_initial": 40,
            # "n_iters": 99,
            "n_iters": 49,
            "xval_trn_and_dev": True,
            "xval_permute": True,
            # "eval_splits": ["trn", "dev", "tst"],
            "eval_splits": ["trn", "dev"],
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
                    "score_key": "oa",
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
                            # {
                            #     "impl": "prompt_opt.ops.compare.DSeekCompareJSON",
                            #     "select_split": "trn",
                            #     "model": "optimizer",
                            #     "template_compare": "dseekdir/dseekdir_compare_01_single_example_v1.txt.jinja",
                            # },
                            {"impl": "prompt_opt.ops.compare.CompareScore", "select_split": "trn", "score_key": "oa"}
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
                            {"impl": "prompt_opt.ops.compare.CompareScore", "select_split": "trn", "score_key": "oa"}
                        ],
                    },
                },
                # "predict_op": {
                #     "impl": "prompt_opt.ops.predict.PredictCorrectedJSON",
                #     "model": "target",
                #     "template_process": "correct_json/correct_predict_01_process_json_schema_v1.txt.jinja",
                #     "template_correct": "correct_json/correct_predict_02_correct_json_schema_v1.txt.jinja",
                #     "max_corrections": 3,
                # },
                "predict_op": {
                    "impl": "prompt_opt.ops.predict.PredictReasoningJSON",
                    "model": "target",
                    # "model": "optimizer",
                    "template_process": "dseek/dseek_predict_01_process_json_schema_v2.txt.jinja",
                },
                "score_ops": [
                    {
                        "impl": "prompt_opt.ops.score_json.ScoreObjectAligner",
                        "score_key": "oa",
                        "schema": read_json(oa_schema_path)
                    }
                ],
            },
        },
    }
    cfg["exp_dir"] = get_exp_dir(cfg)
    return cfg
