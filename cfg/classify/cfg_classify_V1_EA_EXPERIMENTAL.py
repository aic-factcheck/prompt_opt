from ctypes import ArgumentError
import numpy as np
from pathlib import Path

from aic_nlp_utils.json import read_json, read_jsonl

from prompt_opt.models.model_configs import *


def get_exp_dir(cfg):
    return str(Path(cfg["root"], cfg["experiment_name"], f'seed_{cfg["seed"]}').absolute())


def dataset_loader_factory(dataset, trn_scale=1):
    if dataset == "liar":
        return {
            "impl": "prompt_opt.dataset_loader.loader_common.DatasetLoaderSplitsJSONOut",
            "data": {
                "trn": {"path": f"data/{dataset}/trn.json", "size": 16*trn_scale},
                "tst": {"path": f"data/{dataset}/tst.json", "size": 24},
            },
            "schema_path": f"data/{dataset}/schemas/schema_string_answer.json",
        }
    elif dataset.startswith("ethos"):
         return {
            "impl": "prompt_opt.dataset_loader.loader_common.DatasetLoaderJSONOut",
            "data_path": f"data/ethos/{dataset}.json",
            "schema_path": f"data/ethos/schemas/schema_{dataset}.json",
            "trn_size": 16*trn_scale,
            "tst_size": 24
        }
    else:
        raise ArgumentError(f"unknown dataset: {dataset}")
        
def score_oa_factory(dataset):
    dataset_dir = dataset.split("_")[0]
    if dataset == "ethos_multilabel_dct":
        schema = f"data/{dataset_dir}/oa/schema_ethos_multilabel_dct_exact.json"
    else:
        schema = f"data/{dataset_dir}/oa/schema_string_answer_exact.json"
    return {
        "impl": "prompt_opt.ops.score_json.ScoreObjectAligner",
        "score_key": "oa",
        "schema": read_json(schema)
    }
    

def config():
    # dataset = "liar"
    # dataset = "ethos_binary"
    dataset = "ethos_multilabel"
    # dataset = "ethos_multilabel_dct"
    dataset_type = ""
    mutate_max_error_samples = 3
    # mutate_max_error_samples = 1
    # trn_scale = 1
    trn_scale = 4
    cfg = {
        "root": "EXP",
        # "experiment_name": f"{dataset}{dataset_type}_V1_EA",
        # "experiment_note": f"""{dataset}{dataset_type}_V1_EA: initial, DSeekImproveJSON""",
        "experiment_name": f"{dataset}{dataset_type}_V1_EA_DIRts4",
        "experiment_note": f"""{dataset}{dataset_type}_V1_EA_DIRts4: initial, DSeekDirectImproveJSON, trn_scale=4""",
        # "experiment_name": f"{dataset}{dataset_type}_V1_EA_INIts2",
        # "experiment_note": f"""{dataset}{dataset_type}_V1_EA_INIts2: initial, init only, trn_scale=2""",
        "seed": np.random.randint(10000000),
        "models": {
            # "optimizer": get_dseek_llama70b(gpus=[0], reasoning=True),
            # "optimizer": get_dseek_r1_0527_685b(gpus=[0, 1, 2, 3], reasoning=True),
            # "optimizer": get_qwen3_14b(reasoning=True),
            "optimizer": get_qwen3_32b(reasoning=True),
            # "optimizer": get_qwen3_235b(reasoning=True),
        },
        "dataset_loader": dataset_loader_factory(dataset, trn_scale=trn_scale),
        "optimizer": {
            "impl": "prompt_opt.optimizers.ea.EvolutionaryAlgorithm",
            "n_initial": 5,
            # "n_initial": 100000, # init only
            "n_iters": 29,
            "xval_trn_and_dev": True,
            "xval_permute": True,
            "eval_splits": ["trn", "dev", "tst"],
            "prompt_format": "dseek",
            "ops": {
                "init_op": {
                    "impl": "prompt_opt.ops.init.DSeekInitAllExamplesJSON",
                    "model": "optimizer",
                    "trn_size": 6 * trn_scale,
                    "template_init_using_all_examples": "dseek/dseek_init_01_using_all_examples_for_json_output_simple_v1.txt.jinja",
                },
                # "mutate_op": {
                #     "impl": "prompt_opt.ops.mutate.DSeekImproveJSON",
                #     "model": "optimizer",
                #     "select_split": "trn",
                #     "score_key": "oa",
                #     "max_error_samples": mutate_max_error_samples,
                #     "template_improve_first_sample": "dseek/dseek_improve_01_first_sample_v1.txt.jinja",
                #     "template_improve_next_sample": "dseek/dseek_improve_02_next_sample_v1.txt.jinja",
                #     "template_suggest_changes_for_sample": "dseek/dseek_improve_03_suggest_changes_for_sample_v1.txt.jinja",
                #     "template_generate_improved_prompt": (
                #         "dseek/dseek_improve_04_generate_prompt_v1.txt.jinja"
                #         if mutate_max_error_samples > 1
                #         else "dseek/dseek_improve_04_generate_prompt_single_example_v1.txt.jinja"
                #     )
                # },
                "mutate_op": {
                    "impl": "prompt_opt.ops.mutate.DSeekDirectImproveJSON",
                    "model": "optimizer",
                    "select_split": "trn",
                    "score_key": "oa",
                    "max_error_samples": mutate_max_error_samples,
                    "template_give_samples": "dseekdir/dseekdir_improve_01_give_samples_v1.txt.jinja",
                    # "template_give_samples": "dseekdir/dseekdir_improve_01_give_samples_v2.txt.jinja",
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
                                "score_key": "oa"
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
                    score_oa_factory(dataset),
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
