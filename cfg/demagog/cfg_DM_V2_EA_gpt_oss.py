import numpy as np
from pathlib import Path

from aic_nlp_utils.json import read_json, read_jsonl

from prompt_opt.models.model_configs import *


def get_exp_dir(cfg):
    return str(Path(cfg["root"], cfg["experiment_name"], f'seed_{cfg["seed"]}').absolute())


def config():
    dataset_name, dataset_short, dataset_version = "demagog", "DM", "v4" 
    mutate_max_error_samples = 3
    # mutate_max_error_samples = 1
    cfg = {
        "root": "EXP",
        "experiment_name": f"{dataset_short}_V4",
        "experiment_note": f"""{dataset_short}_V4: uses v4 of the dataset + custom MBJ evaluation metric, testing GPT-OSS over VLLM""",
        "seed": np.random.randint(10000000),
        "models": {
            "optimizer": get_gptoss_120b(),
            # "predictor": get_gptoss_120b(),
            "scorer": get_openai_gpt_5_mini(),
        },
        "dataset_loader": {
            "impl": "prompt_opt.dataset_loader.loader_common.DatasetLoaderJSONOut",
            "data_path": f"data/demagog/{dataset_version}/{dataset_name}.jsonl",
            "schema_path": f"data/demagog/{dataset_version}/schemas/schema_{dataset_name}.json",
            "trn_size": 32,
            "tst_size": 24,
        },
        "optimizer": {
            "impl": "prompt_opt.optimizers.ea.EvolutionaryAlgorithm",
            "n_initial": 20,
            "n_iters": 9,
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
                    "trn_size": 16,  # out of 32
                    # MOVE ELSEWHERE
                    "template_init_using_all_examples": "dseek/dseek_init_01_using_all_examples_for_json_output_simple_v2.txt.jinja",
                },
                "mutate_op": {
                    "impl": "prompt_opt.ops.mutate.DSeekImproveJSON",
                    "model": "optimizer",
                    "select_split": "trn",
                    "score_key": "mbj",
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
                            {"impl": "prompt_opt.ops.compare.CompareScore", "select_split": "trn", "score_key": "mbj"}
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
                            {"impl": "prompt_opt.ops.compare.CompareScore", "select_split": "trn", "score_key": "mbj"}
                        ],
                    },
                },
                # "predict_op": {
                #     "impl": "prompt_opt.ops.predict.PredictCorrectedJSON",
                #     "model": "optimizer",
                #     "template_process": "correct_json/correct_predict_01_process_json_schema_v1.txt.jinja",
                #     "template_correct": "correct_json/correct_predict_02_correct_json_schema_v1.txt.jinja",
                #     "max_corrections": 3,
                # },
                "predict_op": {
                    "impl": "prompt_opt.ops.predict.PredictReasoningJSON",
                    "model": "optimizer",
                    "template_process": "dseek/dseek_predict_01_process_json_schema_v2.txt.jinja",
                },
                "score_ops": [
                    {
                        "impl": "prompt_opt.ops.score_json.ModelBasedDSeek",
                        "score_key": "mbj",
                        "model": "scorer",
                        "template_score": "metrics/dseek/demagog/dseek_model_based_metric_02_for_json.txt.jinja",
                    },
                ],
            },
        },
    }
    cfg["exp_dir"] = get_exp_dir(cfg)
    return cfg
