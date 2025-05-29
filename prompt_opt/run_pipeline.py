from collections import Counter, defaultdict
from copy import deepcopy
from io import StringIO
import importlib
import os
from pathlib import Path
from pprint import pp, pprint, pformat
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

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
load_dotenv()

if __name__ == "__main__":

    def save_dir_fn(cfg):
        return cfg["out_dir"]

    args = parse_pycfg_args()
    if Path(args.pycfg).is_file():  # new run
        cfg = read_pycfg(args.pycfg, save_dir_fn=save_dir_fn)
    elif Path(args.pycfg).is_dir():
        cfg = read_json(Path(args.pycfg, "config.json"))
        logger.info("resuming previous run...")
    else:
        raise ValueError(f"The argument must be configuration file. Got this instead: {args.pycfg}")

    out_dir = Path(save_dir_fn(cfg))

    slurm_job_id = get_job_id()
    logger.info(f"pipeline_name: {cfg['pipeline_name']}")
    logger.info(f"slurm_job_id: {slurm_job_id}")
    logger.info(f"out_dir: {out_dir}")

    logger.info(f"seed: {cfg['seed']}")
    rng = np.random.RandomState(cfg["seed"])
    
    if "models" in cfg:
        predictor_loader = PredictorLoader(cfg["models"], exp_path=out_dir)
        predictors = predictor_loader.load()
        logger.debug(f"predictors=\n{predictors}")
    else:
        predictors = None
    

    # store keeps all persistent data
    state = {"predictors": predictors, "store": {}}

    id2cfg = {}
    id2stage = {}
    for cfg_stage in cfg["pipeline"]:
        logger.info(f'loading stage: "{cfg_stage["id"]}"')
        cfg_stage["seed"] = rng.randint(0, np.iinfo(np.int32).max, dtype=np.int32)
        cfg_stage["out_dir"] = out_dir
        id2cfg[cfg_stage["id"]] = cfg_stage
        id2stage[cfg_stage["id"]] = get_class_instance_by_config(cfg_stage)

    for i, (id_, stage) in enumerate(id2stage.items(), start=1):
        st = time.time()
        seed = id2cfg[id_]["seed"]
        logger.info(f'===== STARTING STAGE {i} "{id_}" seed={seed}')

        persistent = id2cfg[id_].get("persistent", True)
        
        output_format = id2cfg[id_].get("output_format", "json").lower()
        assert output_format in ["json", "jsonl"]
        
        output_name = id2cfg[id_]["output_name"] if "output_name" in id2cfg[id_] else f"store_{i:02d}_{id_}"
            
        if output_format == "json":
            fname = f"{output_name}.json"
        else:
            fname = f"{output_name}.jsonl"
            
        fpath = Path(out_dir, fname)
        fname_meta = f"{output_name}_meta.json"
        fpath_meta = Path(out_dir, fname_meta)

        def load_fn():
            if output_format == "json":
                output = read_json(fpath) if fpath.is_file() else None
            else:
                output = read_jsonl(fpath) if fpath.is_file() else None
                
            meta = read_json(fpath_meta) if output and fpath_meta.is_file() else None
            return output, meta

        def save_fn(output, meta):
            if not persistent:
                return output, meta
            
            if output_format == "json":
                write_json(fpath, output)
            else:
                write_jsonl(fpath, output)
            # logger.debug(f"saved: {str(fpath)}")
            if meta and any(e is not None for e in meta):
                assert len(output) == len(meta), (len(output), len(meta))
                write_json(fpath_meta, meta)
                logger.debug(f"saved meta: {str(fpath_meta)}")
            return output, meta
        
        filtered_state = {}
        for k, v in state.items():
            if k == "store":
                store = {}
                for k2, v2 in v.items():
                    scfg = id2cfg.get(id_, {})
                    if "deps" not in scfg:
                        scfg["deps"] = []
                    deps = set(scfg["deps"])
                    if k2 in deps:
                        store[k2] = v2
                v = store
            filtered_state[k] = v

        output, _ = stage.execute(filtered_state, load_fn, save_fn)
        duration = time.time() - st

        state["store"][id_] = output

        # logger.info(pformat(output))
        logger.info(f'      DONE STAGE {i} "{id_}": {len(output)} samples in {duration:.2f}s ({duration/len(output):.3f}s per sample)')

    logger.info("DONE")
