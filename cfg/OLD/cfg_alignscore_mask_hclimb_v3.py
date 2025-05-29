import numpy as np
from pathlib import Path

from prompt_opt.models.model_configs import get_llama8b, get_llama70b

def get_exp_dir(cfg):
    return str(Path(cfg["root"], cfg["experiment_name"], f'seed_{cfg["seed"]}').absolute())

def config():
    oa_metric_schema = {
        "type": "object",
        "properties": {
            "class": {
            "type": "integer",
            "enum": [0, 1],
            "score": "exact"
            }
        },
        "required": ["class"]
    }

    
    cfg = {
        "root": "EXP",
        "experiment_name": "ascoreM-V3-mes3",
        "experiment_note": "v3: masked alignscore, mutate-improve, max_error_samples=3",
        "seed": np.random.randint(10000000),
        "models": {
            # "optimizer": get_llama8b(),
            "optimizer": get_llama70b(),
            # "optimizer": get_llama33_70b(),
            # "optimizer": get_llama33_70b_v2(),
            # "optimizer": get_llama33_70b_full(),
            # "optimizer": get_ll32_3b(),
            # "optimizer": get_crp104b(),
            # "optimizer": get_ayaexp8b(),
            # "optimizer": get_ayaexp32b(),
            # "optimizer": get_minis8b(),
            # "optimizer": get_qwen25_72b(),
            # "optimizer": get_gem2_27b(),
        },
        
        "dataset_loader": {
            "impl": "prompt_opt.dataset_loader.loader_alignscore-v1.DatasetLoaderAlignScoreMaskedV1",
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
            "n_iters": 9,
            # "n_iters": 2,
            "score_key": "oa",
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
                    "score_key": "oa",
                    "max_error_samples": 4,
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
                        "score_key": "oa",
                        "schema": oa_metric_schema
                    },
                ]
            },
        },
    }
    cfg["exp_dir"] = get_exp_dir(cfg)
    return cfg
