import numpy as np
from pathlib import Path


def get_exp_dir(cfg):
    return str(Path(cfg["root"], cfg["experiment_name"], f'seed_{cfg["seed"]}').absolute())

def config():
    oa_metric_schema_05 = {
            "type": "array",
            "items": {
                "type": "string",
                "score": "jaro",
                "threshold": 0.5
            },
            "order": "align",
        }
    
    oa_metric_schema_07 = {
            "type": "array",
            "items": {
                "type": "string",
                "score": "jaro",
                "threshold": 0.7
            },
            "order": "align",
        }
    
    oa_metric_schema_no_treshold = {
            "type": "array",
            "items": {
                "type": "string",
                "score": "jaro",
            },
            "order": "align",
        }
    
    cfg = {
        "root": "EXP",
        "experiment_name": "sir_phrase-V0",
        "experiment_note": "v0: baseline random search",
        "seed": np.random.randint(10000000),
        "models": {
            # "optimizer": get_llama8b(),
            "optimizer": get_llama70b(),
            # "optimizer": get_ll32_3b(),
            # "optimizer": get_crp104b(),
            # "optimizer": get_ayaexp8b(),
            # "optimizer": get_ayaexp32b(),
            # "optimizer": get_minis8b(),
            # "optimizer": get_qwen25_72b(),
            # "optimizer": get_gem2_27b(),
        },
        
        "dataset_loader": {
            "impl": "prompt_opt.dataset_loader.loader_sir_phrase-v1.DatasetLoaderSiRPhraseV1",
            # "sizes": {
            #     "trn": 2,
            #     "dev": 2,
            #     "tst": 3
            # }
        },
        
        "optimizer": {
            "impl": "prompt_opt.optimizers.hill_climber.HillClimber",
            "n_initial": 100,
            "top_k": 2,
            "n_neighbors": 5,
            "n_iters": 0,
            "score_key": "oa-07",
            "select_split": "dev",
            "eval_splits": ["trn", "dev", "tst"],
            "ops": {
                "init_op": {
                    "impl": "prompt_opt.ops.init.TRRInitAllExamplesJSON",
                    "model": "optimizer",
                    "template_system_content": "md/trr_system_v1.txt.jinja",
                    "template_init_all_examples": "md/init_all_examples_for_json_output_simple_v1.txt.jinja",
                    "template_generate_prompt":"md/trr_generate_prompt_for_json_output_simple_v1.txt.jinja",
                },
                "mutate_op": {
                    "impl": "prompt_opt.ops.mutate.TRRImproveJSON",
                    "model": "optimizer",
                    "select_split": "dev",
                    "score_key": "oa-07",
                    "max_error_samples": 3,
                    "template_system_content": "md/trr_system_v1.txt.jinja",
                    "template_improve_first_sample": "md/trr_improve_first_sample_v1.txt.jinja",
                    "template_improve_next_sample": "md/trr_improve_next_sample_v1.txt.jinja",
                    "template_suggest_changes_for_sample": "md/trr_improve_suggest_changes_for_sample_v1.txt.jinja",
                    "template_generate_improved_prompt":"md/trr_generate_improved_prompt_v1.txt.jinja",
                },
                "predict_op": {
                    "impl": "prompt_opt.ops.predict.COTPredictSteppedJSON",
                    "model": "optimizer",
                    "template_system_content": "chat/system_v1.txt.jinja",
                    "template_think": "json/cot_stepped_think_json_schema_v1.txt.jinja",
                    "template_result": "json/cot_stepped_result_json_schema_v1.txt.jinja",
                },
                "score_ops": [
                    {
                        "impl": "prompt_opt.ops.score_json.ScoreObjectAligner",
                        "score_key": "oa-05",
                        "schema": oa_metric_schema_05
                    },
                    {
                        "impl": "prompt_opt.ops.score_json.ScoreObjectAligner",
                        "score_key": "oa-07",
                        "schema": oa_metric_schema_07
                    },
                    {
                        "impl": "prompt_opt.ops.score_json.ScoreObjectAligner",
                        "score_key": "oa-nt",
                        "schema": oa_metric_schema_no_treshold
                    }
                ]
            },
        },
    }
    cfg["exp_dir"] = get_exp_dir(cfg)
    return cfg


# this should be made more general, not SLURM-centered!
def get_llama8b(gpus=[0]):
    return {
        "short": "llama8b",
        "name": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "gpus": gpus,
        "guided_decoding_backend": "outlines",
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 65536}
    }


def get_llama70b(gpus=[0, 1]):
    return {
        "short": "llama70b",
        "name": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        "gpus": gpus,
        "guided_decoding_backend": "outlines",
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 65536}
    }
    

def get_ll32_3b(gpus=[0]):
    return {
        "short": "ll32_3b",
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "gpus": gpus,
        "guided_decoding_backend": "outlines",
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 131072}
    }

    
def get_crp104b(gpus=[0, 1]):
    return {
        "short": "crp104b",
        "name": "aixsatoshi/c4ai-command-r-plus-08-2024-awq",
        "gpus": gpus,
        "guided_decoding_backend": "outlines",
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 16384}
    }
    
    
def get_ayaexp8b(gpus=[0]):
    return {
        "short": "ayaexp8b",
        "name": "CohereForAI/aya-expanse-8b",
        "gpus": gpus,
        "guided_decoding_backend": "outlines",
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 8192}
    }
    
    
def get_ayaexp32b(gpus=[0, 1]):
    return {
        "short": "ayaexp32b",
        "name": "CohereForAI/aya-expanse-32b",
        "gpus": gpus,
        "guided_decoding_backend": "outlines",
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 16384}
    }
    
    
def get_minis8b(gpus=[0]):
    return { # not working for outlines
        "short": "minis8b",
        "name": "mistralai/Ministral-8B-Instruct-2410",
        "gpus": gpus,
        "guided_decoding_backend": "outlines",
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 32768,
                      "tokenizer_mode": "mistral", "config_format": "mistral", "load_format": "mistral"}
    }
    
    
def get_qwen25_72b(gpus=[0, 1]):
    return {
        "short": "qwen25_72b",
        "name": "Qwen/Qwen2.5-72B-Instruct-AWQ",
        "gpus": gpus,
        "guided_decoding_backend": "outlines",
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 32768}
    }
    
def get_gem2_27b(gpus=[0, 1]):
    return {
        "short": "gem2_27b",
        "name": "google/gemma-2-27b-it",
        "gpus": gpus,
        "guided_decoding_backend": "outlines",
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 4096}
    }
