import numpy as np
from pathlib import Path

from aic_nlp_utils.json import read_json, read_jsonl

from prompt_opt.models.model_configs import get_dseek_llama8b, get_dseek_llama70b, get_qwq32b, get_qwen3_14b, get_qwen3_32b


def get_exp_dir(cfg):
    return str(Path(cfg["root"], cfg["experiment_name"], f'seed_{cfg["seed"]}').absolute())


def config():
    mutate_max_error_samples = 3
    # mutate_max_error_samples = 1
    cfg = {
        "root": "EXP",
        "experiment_name": "events_complete_V2_EAe",
        "experiment_note": """V2_EAe: fixed resume/caching""",
        "seed": np.random.randint(10000000),
        "models": {
            # "optimizer": get_qwen3_14b(reasoning=True),
            "optimizer": get_qwen3_32b(reasoning=True),
            # "optimizer": get_dseek_llama70b(reasoning=True),
            # "target": get_dseek_llama70b(reasoning=True),
            # "optimizer": get_qwq32b(reasoning=False, gpus=[2]), # reasoning means support for constrained generation - not needed here
        },
        "dataset_loader": {
            "impl": "prompt_opt.dataset_loader.loader_cro-v2.DatasetLoaderEventsCompleteV2",
            "merge_trn_and_dev": True,
        },
        "optimizer": {
            "impl": "prompt_opt.optimizers.ea.EvolutionaryAlgorithm",
            "n_initial": 10,
            "n_iters": 19,
            "xval_trn_and_dev": True,
            "xval_permute": True,
            "eval_splits": ["trn", "dev", "tst"],
            "prompt_format": "dseek",
            "ops": {
                # "init_op": {
                #     "impl": "prompt_opt.ops.init.InitExisting",
                #     "sources": [
                #         {"count": 1, "source": "EXP/XXX/seed_XXXXXXX", "metric": "mbj", "type": "archive", "select": "best", "split": "tst"}
                #     ]
                # },
                "init_op": {
                    "impl": "prompt_opt.ops.init.DSeekInitAllExamplesJSON",
                    "model": "optimizer",
                    "trn_size": 6,
                    "template_init_using_all_examples": "dseek/dseek_init_01_using_all_examples_for_json_output_simple_v1.txt.jinja",
                },
                "mutate_op": {
                    "impl": "prompt_opt.ops.mutate.MultiMutation",
                    "ops": [
                        {
                            "weight": 1.0,
                            "cfg": {
                                "impl": "prompt_opt.ops.mutate.DSeekImproveJSON",
                                "model": "optimizer",
                                "select_split": "trn",
                                "score_key": "mbj",
                                "max_error_samples": mutate_max_error_samples,
                                "template_improve_first_sample": "dseek/dseek_improve_01_first_sample_v1.txt.jinja",
                                "template_improve_next_sample": "dseek/dseek_improve_02_next_sample_v1.txt.jinja",
                                "template_suggest_changes_for_sample": "dseek/dseek_improve_03_suggest_changes_for_sample_v1.txt.jinja",
                                "template_generate_improved_prompt": (
                                    "dseek/dseek_improve_04_generate_prompt_v1.txt.jinja"
                                    if mutate_max_error_samples > 1
                                    else "dseek/dseek_improve_04_generate_prompt_single_example_v1.txt.jinja"
                                ),
                            },
                        },
                        {
                            "weight": 1.0,
                            "cfg": {
                                "impl": "prompt_opt.ops.mutate_unlabeled.DSeekImproveUnlabeledJSON",
                                "model": "optimizer",
                                "unlabeled_split": "unl",
                                "labeled_size": 2,
                                "template_predict_first": "dseek_unlabeled/dseek_unlabeled_improve_01_predict_first_v1.txt.jinja",
                                "template_improve_instructions": "dseek_unlabeled/dseek_unlabeled_improve_02_improve_instructions_v1.txt.jinja",
                            },
                        },
                    ],
                },
                "select_op": {
                    "impl": "prompt_opt.ops.select.Tournament",
                    "cache": "select_cache.jsonl",
                    "compare_op": {
                        "impl": "prompt_opt.ops.compare.DebugCompare",
                        "log": "compare_log_select.jsonl",
                        "ops": [
                            {
                                "impl": "prompt_opt.ops.compare.DSeekCompareJSON",
                                "select_split": "trn",
                                "model": "optimizer",
                                "template_compare": "dseekdir/dseekdir_compare_01_single_example_v1.txt.jinja",
                            },
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
                            {
                                "impl": "prompt_opt.ops.compare.DSeekCompareJSON",
                                "select_split": "trn",
                                "model": "optimizer",
                                "template_compare": "dseekdir/dseekdir_compare_01_single_example_v1.txt.jinja",
                            },
                            {
                                "impl": "prompt_opt.ops.compare.CompareScore",
                                "select_split": "trn",
                                "score_key": "mbj"
                            }
                        ],
                    }
                },
                "predict_op": {
                    "impl": "prompt_opt.ops.predict.PredictReasoningJSON",
                    "model": "optimizer",
                    "template_process": "dseek/dseek_predict_01_process_json_schema_v1.txt.jinja",
                },
                "score_ops": [
                    # {
                    #     "impl": "prompt_opt.ops.score_json.ScoreObjectAligner",
                    #     "score_key": "oa",
                    #     "schema": oa_metric_schema
                    # },
                    {
                        "impl": "prompt_opt.ops.score_json.ModelBasedDSeek",
                        "score_key": "mbj",
                        "model": "optimizer",
                        "template_score": "metrics/dseek/dseek_model_based_metric_01_for_json.txt.jinja",
                    },
                ],
            },
        },
    }
    cfg["exp_dir"] = get_exp_dir(cfg)
    return cfg
