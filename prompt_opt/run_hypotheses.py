from collections import Counter, defaultdict, deque, namedtuple
from copy import deepcopy
import functools
import inspect
import json
import os
from pathlib import Path
import pickle
from pprint import pp, pprint
import re
import sys
import time
from typing import Dict, List

from jinja2 import Template
from loguru import logger
from more_itertools import chunked
import numpy as np
import openai
from openai import OpenAI
import pandas as pd
import plotly.express as px
from tqdm import tqdm

from colorutils import Color

from dotenv import load_dotenv

from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl, process_to_jsonl
from aic_nlp_utils.pycfg import parse_pycfg_args, read_pycfg


from prompt_opt.utils import *
from prompt_opt.hypoopt.evaluate import hypothesis_to_instructions, hypothesis_to_instructions2, compute_string_accuracy, compute_object_aligner_accuracy, get_ethos_accs, predict_samples
from prompt_opt.hypoopt.llm import PromptJSONByChat, PromptDefault
from prompt_opt.hypoopt.operations import Operations


logger.remove()
logger.add(sys.stderr, colorize=True)

load_dotenv()

string_answer_schema = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
        }
    },
    "required": ["answer"],
    "additionalProperties": False,
}


def connect_llms(opt_model, tgt_model, opt_model_base_url=None, tgt_model_base_url=None):
    client_opt = OpenAI(base_url=opt_model_base_url, api_key = "EMPTY") if opt_model_base_url else OpenAI()
    client_tgt = OpenAI(base_url=tgt_model_base_url, api_key = "EMPTY") if tgt_model_base_url else OpenAI()
    
    Prompt_opt = PromptJSONByChat if opt_model.startswith("gpt-oss:") else PromptDefault
    Prompt_tgt = PromptJSONByChat if tgt_model.startswith("gpt-oss:") else PromptDefault
    
    llm_opt = Prompt_opt(model=opt_model, client=client_opt)
    llm_tgt = Prompt_tgt(model=tgt_model, client=client_tgt)
    
    return llm_opt, llm_tgt


def test_updates(out_jsonl, parent, ops: Operations, start_idx, end_idx, bsize, start_cidx, k):
    out_jsonl = Path(out_jsonl)
    n = end_idx - start_idx
    assert n % bsize == 0
    n_batches = n//bsize
    
    candidates = read_jsonl(out_jsonl)  if out_jsonl.is_file() else []
    existing_cidxs = {c["id"]: c for c in candidates}
    if len(existing_cidxs) > 0:
        logger.info(f"existing ids: {existing_cidxs.keys()}")
        
    cidx = start_cidx
    for idx in range(k):
        candidate = parent
        logger.info(f"RUN: {idx+1}, train samples: {len(ops.trn_data)}")
        for bidx in range(n_batches):
            if cidx in existing_cidxs:
                logger.info("skipping...")
                candidate = existing_cidxs[cidx]
                cidx += 1
                continue
            
            sidx = start_idx+bidx*bsize 
            logger.info(f"batch: {bidx+1}/{n_batches}, indices: {sidx}:{sidx+bsize}")
            candidate = ops.update(
                id_=cidx,
                parent=candidate,
                start_idx=sidx,
                bsize=bsize,
                eval_indices=range(0, sidx+bsize) # evaluate on all known training data
                )
            candidates.append(candidate)
            write_jsonl(out_jsonl, candidates)
            cidx += 1
    return candidates

def test_updates_with_generalization(out_jsonl, parent, ops: Operations, 
                                     start_idx, end_idx, bsize, start_cidx,
                                     generalize_interval=1,
                                     k=1):
    # firstly updates using new training data,
    # then applies generalization
    # finaly, the parent for next is the better of the two
    # the generalization will run every generalize_interval update steps 
    out_jsonl = Path(out_jsonl)
    n = end_idx - start_idx
    assert n % bsize == 0
    n_batches = n//bsize
    
    candidates = read_jsonl(out_jsonl)  if out_jsonl.is_file() else []
    existing_cidxs = {c["id"]: c for c in candidates}
    if len(existing_cidxs) > 0:
        logger.info(f"existing ids: {existing_cidxs.keys()}")
        
    cidx = start_cidx
    for idx in range(k):
        candidate = parent
        logger.info(f"RUN: {idx+1}, train samples: {len(ops.trn_data)}")
        for bidx in range(n_batches):
            sidx = start_idx+bidx*bsize 
            
            # UPDATE phase
            if cidx in existing_cidxs:
                logger.info("skipping update...")
                candidate1 = existing_cidxs[cidx]
                assert candidate1["op"] == "update"
                cidx += 1
            else: 
                logger.info(f"update batch: {bidx+1}/{n_batches}, indices: {sidx}:{sidx+bsize}")
                candidate1 = ops.update(
                    id_=cidx,
                    parent=candidate,
                    start_idx=sidx,
                    bsize=bsize,
                    eval_indices=range(0, sidx+bsize) # evaluate on all known training data
                    )
                candidates.append(candidate1)
                write_jsonl(out_jsonl, candidates)
                cidx += 1
                
            if (bidx+1) % generalize_interval == 0:      
                # GENERALIZE phase
                if cidx in existing_cidxs:
                    logger.info("skipping generalize...")
                    candidate2 = existing_cidxs[cidx]
                    assert candidate2["op"] == "generalize"
                    cidx += 1
                else: 
                    logger.info(f"generalize batch: {bidx+1}/{n_batches}, indices: {sidx}:{sidx+bsize}")
                    candidate2 = ops.generalize(
                        id_=cidx,
                        parent=candidate1,
                        eval_indices=range(0, sidx+bsize)
                        )
                    candidates.append(candidate2)
                    write_jsonl(out_jsonl, candidates)
                    cidx += 1
                
                # CHOICE of the better, for draw prefer the generalized candidate
                if candidate1["eval"]["trn"]["accuracy"] <= candidate2["eval"]["trn"]["accuracy"]:
                    logger.info("choosing generalized candidate")
                    candidate = candidate2
                else:
                    logger.info("choosing updated candidate")
                    candidate = candidate1
            else:
                candidate = candidate1
                
    return candidates


def test_updates_with_fixes(out_jsonl, parent, ops: Operations, 
                                     start_idx, end_idx, bsize, start_cidx,
                                     fix_iters=1,
                                     k=1):
    # firstly updates using new training data,
    # then applies fix `fix_iters`-times
    # finaly, the parent for next is the best achieved
    out_jsonl = Path(out_jsonl)
    n = end_idx - start_idx
    assert n % bsize == 0
    n_batches = n//bsize
    
    candidates = read_jsonl(out_jsonl)  if out_jsonl.is_file() else []
    existing_cidxs = {c["id"]: c for c in candidates}
    if len(existing_cidxs) > 0:
        logger.info(f"existing ids: {existing_cidxs.keys()}")
        
    cidx = start_cidx
    for idx in range(k):
        candidate = parent
        logger.info(f"RUN: {idx+1}, train samples: {len(ops.trn_data)}")
        for bidx in range(n_batches):
            sidx = start_idx+bidx*bsize 
            
            # UPDATE phase
            if cidx in existing_cidxs:
                logger.info("skipping update...")
                candidate = existing_cidxs[cidx]
                assert candidate["op"] == "update"
                cidx += 1
            else: 
                logger.info(f"update batch: {bidx+1}/{n_batches}, indices: {sidx}:{sidx+bsize}")
                candidate = ops.update(
                    id_=cidx,
                    parent=candidate,
                    start_idx=sidx,
                    bsize=bsize,
                    eval_indices=range(0, sidx+bsize) # evaluate on all known training data
                    )
                candidates.append(candidate)
                write_jsonl(out_jsonl, candidates)
                cidx += 1
                
            # now `candidate`` is always the updated version of parent
            # TODO: try if using the better of parent/updated works better (more aligns with hillclimber approach)
            # note, however, that the parent must be evaluated for new train samples!
                
            for fixitr in range(fix_iters):
                logger.info(f"FIX: {fixitr+1}")
                # FIX phase
                if cidx in existing_cidxs:
                    logger.info("skipping fix...")
                    candidate_new = existing_cidxs[cidx]
                    assert candidate_new["op"] == "fix"
                    cidx += 1
                else: 
                    candidate_new = ops.fix(
                        id_=cidx,
                        parent=candidate,
                        eval_indices=range(0, sidx+bsize)
                        )
                    candidates.append(candidate_new)
                    write_jsonl(out_jsonl, candidates)
                    cidx += 1
                
                # CHOICE of the better, for draw prefer the generalized candidate
                if candidate["eval"]["trn"]["accuracy"] <= candidate_new["eval"]["trn"]["accuracy"]:
                    logger.info("choosing new candidate")
                    candidate = candidate_new
                else:
                    logger.info("keeping old candidate")
                
    return candidates


def load_wos11967():
    trn_data = read_json("data/wos/wos11967_trn.json")["examples"]
    tst_data = read_json("data/wos/wos11967_tst.json")["examples"]
        
    trn_data = [{"query": ex["query"], "answer": {"answer": "label" + str(ex["answer"])}, "idx": idx} for idx, ex in enumerate(trn_data)]
    tst_data = [{"query": ex["query"], "answer": {"answer": "label" + str(ex["answer"])}, "idx": idx} for idx, ex in enumerate(tst_data)]

    answer_schema = string_answer_schema
    score_samples = compute_string_accuracy
    tst_size = 66
    name = "wos11967"
    return trn_data, tst_data, answer_schema, score_samples, tst_size, name


def load_bbh_salient():
    # BBH salient_translation_error_detection, 6 classes
    data = read_json("data/BBH_PO/datasets/salient_translation_error_detection/task.json")["examples"]
    trn_data = data[:16] + data[64:] # original train data were 0:16, now exending to everything beyond test
    tst_data = data[16:64] # original test data were 24 samples 16:40, here extended to 48
        
    trn_data = [{**ex, "idx": idx} for idx, ex in enumerate(trn_data)]
    tst_data = [{**ex, "idx": idx} for idx, ex in enumerate(tst_data)]

    answer_schema = read_json("data/BBH_PO/schemas/schema_string_answer.json")
    score_samples = compute_string_accuracy
    name = "bbh_salient"
    return trn_data, tst_data, answer_schema, score_samples, len(tst_data), name



def load_ethos():
    data = read_json("data/ethos/ethos_multilabel.json")["examples"]       
    trn_data = data[:300]        
    tst_data = data[300:]        
    answer_schema = string_answer_schema
    score_samples = compute_string_accuracy
    tst_size = 72
    name = "ethos"
    return trn_data, tst_data, answer_schema, score_samples, tst_size, name

        
if __name__ == "__main__":
    llm_opt_name = "gpt-5"
    # llm_tgt_name = "gpt-5-mini"
    # llm_tgt_name = "gpt-oss:20b" # cidx wos11967:4
    # llm_tgt_name = "gpt-oss:120b" # cidx bbh_salient: 2, 2(size 4)
    # llm_tgt_name = "llama3.2:3b"
    llm_tgt_name = "qwen3:8b" # cidx wos11967:4, bbh_salient: 2, 2(size 4)
    # llm_tgt_name = "qwen3:14b" # cidx wos11967:0
    # llm_tgt_name = "qwen3:32b" # cidx wos11967:0, bbh_salient: 2
    # llm_tgt_name = "gemma3:27b" # cidx wos11967:1
    # llm_opt, llm_tgt = connect_llms(llm_opt_name, llm_tgt_name)
    llm_opt, llm_tgt = connect_llms(llm_opt_name, llm_tgt_name, tgt_model_base_url="http://h02:8222/v1")

    trn_data, tst_data, answer_schema, score_samples, tst_size, name = load_bbh_salient()
    # trn_data, tst_data, answer_schema, score_samples, tst_size, name = load_ethos()
    # trn_data, tst_data, answer_schema, score_samples, tst_size, name = load_wos11967()
    
    init_size = 4
    bsize = 4
    end_idx = 128
    generalize_interval = 0
    fix_iters = 2
    
    ops = Operations(trn_data, tst_data[:tst_size], llm_opt=llm_opt, llm_tgt=llm_tgt, score_samples=score_samples, predict_schema=answer_schema)
    
    init_candidates = read_jsonl(f"EXP/HYPO/init_{name}_{llm_opt_name}_{llm_tgt_name}_size{init_size}.jsonl")
    
    unused_idx = max([c["id"] for c in init_candidates]) + 1
    for cidx in [2]:
        parent = init_candidates[cidx]
        # out_jsonl = f"EXP/HYPO/updategen{generalize_interval}_{name}_{llm_opt_name}_{llm_tgt_name}_cidx{cidx}_size{init_size}_bsize{bsize}.jsonl"
        out_jsonl = f"EXP/HYPO/fix{fix_iters}_{name}_{llm_opt_name}_{llm_tgt_name}_cidx{cidx}_size{init_size}_bsize{bsize}.jsonl"
        logger.info(f"running for {out_jsonl}")
        # candidates = test_updates(
        # candidates = test_updates_with_generalization(
        candidates = test_updates_with_fixes(
            out_jsonl,
            parent,
            ops, 
            start_idx=init_size,
            end_idx=end_idx,
            bsize=bsize, 
            start_cidx=unused_idx,
            # generalize_interval=generalize_interval,
            fix_iters=fix_iters,
            k=5
        )
    logger.info("ALL DONE")