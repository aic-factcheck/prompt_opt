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
import wandb

from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl, process_to_jsonl
from aic_nlp_utils.encoding import nfc
from aic_nlp_utils.files import create_parent_dir
from aic_nlp_utils.pycfg import parse_pycfg_args, read_pycfg

from prompt_opt.models.predictor_loader import PredictorLoader
from prompt_opt.slurm_utils import rename_slurm_job_name, get_job_id
from prompt_opt.utils import *

sys.path.append("/home/drchajan/devel/python/FC/automated-fact-checking")

# TQDM support turned off (non-tqdm logs not flushing)
# logger.remove()
# logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

logger.remove()
logger.add(sys.stderr, colorize=True)
# logger = logger.opt(colors=True)

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
load_dotenv()

if __name__ == "__main__":

    def save_dir_fn(cfg):
        return cfg["exp_dir"]


    args = parse_pycfg_args()
    if Path(args.pycfg).is_file(): # new run
        cfg = read_pycfg(args.pycfg, save_dir_fn=save_dir_fn)
        resume = False
    elif Path(args.pycfg).is_dir():
        cfg = read_json(Path(args.pycfg, "config.json"))
        resume = True
        logger.info("resuming previous run...")
    else:
        raise ValueError(f"The argument must be configuration file or existing run directory (to resume). Got this instead: {args.pycfg}")


    exp_path = Path(save_dir_fn(cfg))
    exp_path.mkdir(parents=True, exist_ok=True)

    slurm_job_id = get_job_id()
    logger.info(f"slurm_job_id: {slurm_job_id}")
    cfg["slurm_job_id"] = slurm_job_id
    rename_slurm_job_name(slurm_job_id, cfg["experiment_name"])
    rng = np.random.RandomState(cfg["seed"])
    logger.info(f"exp_path: {exp_path}")
    logger.info(f"experiment_name: {cfg['experiment_name']}")

    with open(Path(exp_path, f"slurm_job_id.{slurm_job_id}"), 'w') as fp:
        pass
    
    dataset_loader = get_class_instance_by_config(cfg["dataset_loader"])
    predictor_loader = PredictorLoader(cfg["models"], exp_path=exp_path)
    predictors = predictor_loader.load()
    optimizer = get_class_instance_by_config(cfg["optimizer"],
                                             exp_path=exp_path,
                                             dataset_loader=dataset_loader,
                                             predictors=predictors,
                                             resume=resume)
            
    if resume:
        wandb_ids = [f.name.split('.')[-1] for f in exp_path.iterdir() if f.is_file() and f.name.startswith('wandb_id.')]
        assert len(wandb_ids) == 1, wandb_ids
        wandb.init(project='Prompt Opt', name=cfg["experiment_name"], config=cfg, id=wandb_ids[0], resume="must")
    else:
        wandb_run = wandb.init(project='Prompt Opt', name=cfg["experiment_name"], config=cfg)
        with open(Path(exp_path, f"wandb_id.{wandb_run.id}"), 'w') as fp:
            fp.write(wandb_run.url)
        
    archive = optimizer.optimize(rng=rng)
    
    wandb.finish()
