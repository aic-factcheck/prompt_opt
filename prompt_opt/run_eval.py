from collections import Counter, defaultdict
from copy import deepcopy
from io import StringIO
import importlib
import os
from pathlib import Path
from pprint import pp
import re
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm
from prompt_opt.optimizers.predict_evaluate import PredictEvaluateAndLogCandidate
import wandb

from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl, process_to_jsonl
from aic_nlp_utils.encoding import nfc
from aic_nlp_utils.files import create_parent_dir
from aic_nlp_utils.pycfg import parse_pycfg_args, read_pycfg

from prompt_opt.models.predictor_loader import PredictorLoader
from prompt_opt.slurm_utils import rename_slurm_job_name, get_job_id
from prompt_opt.utils import *

sys.path.append("/home/drchajan/devel/python/FC/automated-fact-checking")

logger.remove()
logger.add(sys.stderr, colorize=True)

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
load_dotenv()

if __name__ == "__main__":
    
    def save_dir_fn(cfg):
        return cfg["exp_dir"]
    
    args = parse_pycfg_args()
    if Path(args.pycfg).is_file():
        cfg = read_json(args.pycfg)
    elif not Path(args.pycfg).is_dir():
        cfg = read_json(Path(args.pycfg, "config.json"))
    else:
        raise ValueError(f"The argument must be configuration file. Got this instead: {args.pycfg}")

    exp_path = Path(save_dir_fn(cfg))

    slurm_job_id = get_job_id()
    logger.info(f"slurm_job_id: {slurm_job_id}")
    logger.info(f"exp_path: {exp_path}")
    logger.info(f"experiment_name: {cfg['experiment_name']}")

    dataset_loader = get_class_instance_by_config(cfg["dataset_loader"])
    predictor_loader = PredictorLoader(cfg["models"], exp_path=exp_path)
    predictors = predictor_loader.load()
    
    output_schema = dataset_loader.get_output_schema()
    
    predict_op = get_class_instance_by_config(
        cfg["optimizer"]["ops"]["predict_op"],
        predictors=predictors)
    
    # PEOPLE
    # candidate_id = 87 # EXP/people-V0/seed_8875203
    # samples = read_jsonl("people_to_label.jsonl")
    # archive = read_jsonl(Path(exp_path, "archive.jsonl"))
    # candidate = next(filter(lambda c: c["id"] == candidate_id, archive), None)
    # assert candidate, f"Cannot find candidate id={candidate_id}"
    # prompt = candidate2prompt_md(candidate)
    # out_file = "data/labeled_datasets/people_roles/V2/predicted.jsonl"
    
    # EVENTS
    # candidate_id = 64 # EXP/events-V6-ts6/seed_5781810
    # samples = read_jsonl("data/labeled_datasets/events_V2_annotate_last_10.jsonl")
    # archive = read_jsonl(Path(exp_path, "archive.jsonl"))
    # candidate = next(filter(lambda c: c["id"] == candidate_id, archive), None)
    # assert candidate, f"Cannot find candidate id={candidate_id}"
    # prompt = candidate2prompt_md(candidate)
    # out_file = "data/labeled_datasets/events/V2/predicted_last_10.jsonl"
    
    # ORGS
    # prompt = Path("data/labeled_datasets/orgs/V1/bootstrap_prompt.md").read_text()
    # prompt = Path("data/labeled_datasets/orgs/V1/bootstrap_prompt_ner_types.md").read_text()
    # samples = read_jsonl("data/labeled_datasets/people_roles_V2.jsonl") # reuse people_V2
    # out_file = "data/labeled_datasets/orgs/V1/predicted.jsonl"
    # out_file = "data/labeled_datasets/orgs/V1/predicted_ner_types.jsonl"
    
    # LOCS
    # prompt = Path("data/labeled_datasets/locs/V1/bootstrap_prompt_ner_types.md").read_text()
    # samples = read_jsonl("data/labeled_datasets/people_roles_V2.jsonl") # reuse people_V2
    # out_file = "data/labeled_datasets/locs/V1/predicted_ner_types.jsonl"
    
    # EVENT Temporal Validity
    prompt = Path("data/labeled_datasets/event_temp_val/V1/bootstrap_prompt_temp_val.md").read_text()
    samples = read_jsonl("data/labeled_datasets/event_temp_val/V1/bootstrap_queries.jsonl")
    out_file = "data/labeled_datasets/event_temp_val/V1/predicted_temp_val.jsonl"
        
    logger.info(f"PROMPT:\n{prompt}")
    
    for i, sample in enumerate(samples):
        logger.info(f"evaluating sample {i+1}/{len(samples)}")

        response, messages = predict_op.predict(
            prompt=prompt, 
            query=sample["query"],
            output_schema=output_schema,
            examples=None)
        
        sample.update(response)

        response.update(
            {
                "gold": sample["answer"],
                "query": sample["query"],
                "messages": messages,
            }
        )  
        write_jsonl(out_file, samples)
    
    logger.info("DONE")
        
