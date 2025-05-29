import numpy as np
from pathlib import Path

from prompt_opt.models.model_configs import get_llama8b, get_llama70b

def get_exp_dir(cfg):
    return str(Path(cfg["root"], cfg["experiment_name"], f'seed_{cfg["seed"]}').absolute())

def config():
    oa_metric_schema = {
        "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "score": "jaro"
                    },
                    "roles": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "score": "jaro"
                        },
                        "order": "align"
                    }
                },
                "required": ["name", "roles"],
            },
            "order": "align"
        }

    
    cfg = {
        "root": "EXP",
        "experiment_name": "people-V6-ts6",
        "experiment_note": """v6: same as people-V6-ts6 but using model-based-metric-for-json.""",
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
            # "optimizer": get_phi4_14b(),
            # "optimizer": get_openai_gpt_4o_mini(),
            # "optimizer": get_openai_gpt_4o(),
        },
        
        "dataset_loader": {
            "impl": "prompt_opt.dataset_loader.loader_people-v1.DatasetLoaderPeopleRolesV1",
            "merge_trn_and_dev": True
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
            "score_key": "mbj",
            "xval_trn_and_dev": True,
            "xval_permute": True,
            "select_split": "dev",
            "eval_splits": ["trn", "dev", "tst"],
            "ops": {
                "init_op": {
                    "impl": "prompt_opt.ops.init.TRRInitAllExamplesJSON",
                    "model": "optimizer",
                    "trn_size": 6,
                    "template_system_content": "md/trr_system_v1.txt.jinja",
                    "template_init_all_examples": "md/init_all_examples_for_json_output_simple_v1.txt.jinja",
                    "template_generate_prompt":"md/trr_generate_prompt_for_json_output_simple_v1.txt.jinja",
                },
                "mutate_op": {
                    "impl": "prompt_opt.ops.mutate.TRRImproveJSON",
                    "model": "optimizer",
                    "select_split": "trn",
                    "score_key": "mbj",
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
                        "score_key": "oa",
                        "schema": oa_metric_schema
                    },
                    {
                        "impl": "prompt_opt.ops.score_json.ModelBasedMD",
                        "score_key": "mbj",
                        "model": "optimizer",
                        "template_system_content": "chat/system_v1.txt.jinja",
                        "template_think": "metrics/md/cot_model_based_metric_for_json_think_01.txt.jinja",
                        "template_result": "metrics/md/cot_model_based_metric_for_json_result_01.txt.jinja",
                    },
                ]
            },
        },
    }
    cfg["exp_dir"] = get_exp_dir(cfg)
    return cfg
