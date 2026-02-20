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

logger.remove()
logger.add(sys.stderr, colorize=True)

load_dotenv()

# based on notebooks_priv/playground_hypotheses_generation.ipynb

def init_openai():
    client = OpenAI()
    default_model = "o3-mini"
    # default_model = "gpt-4o-mini"
    return client, default_model


def init_local():
    client = OpenAI(
        base_url = "http://g06:8333/v1",
        api_key = "EMPTY"
    )
    models = client.models.list()
    logger.info("available models:")
    for model in models:
        logger.info(model.id)
    default_model = list(models)[0].id
        
    return client, default_model
    
    
# client, default_model = init_openai()
client, default_model = init_local()

    
def prompt_llm(prompt, model=default_model):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    # return completion
    return completion.choices[0].message.content    

init_template = """# Inferring Query-to-Answer Transformation Rules
You will be provided with one or more (query, answer) data pairs.
Your task is to generate a single most probable hypothesis that describe the reasoning or rules used to transform each query into its corresponding answer.
The hypothesis should offer a detailed explanation of the transformation process in a form of a high-level pseudo-code.

The examples follow:
{% for ex in dataset %}
<example id="{{ loop.index }}">
<query>{{ ex.query }}</query>
<answer>{{ ex.answer }}</answer>
</example>
{% endfor %}
"""

eval_template = """# Refine Query-to-Answer Transformation Hypothesis

You will receive one or more (query, answer) pairs along with a hypothesis that aims to explain how the transformation from query to answer occurs.

Your task consists of three parts:

1) **Evaluate**  
Assess whether each sample is consistent with the given hypothesis.  
List any samples that do not align, including their ID and a brief explanation.

2) **Improved Hypothesis**  
If there are any inconsistencies, revise the hypothesis to address them.  
Preserve the overall structure and formatting of the original hypothesis.  
Wrap the revised hypothesis in `<hypothesis>` and `</hypothesis>` tags.  
Do not indicate in the hypothesis text itself that it is a revision.  
You may include comments before or after the tags.  
Leave this section blank if all samples are aligned.

3) **Log**  
Report a single number indicating how many samples did not align with the hypothesis.

Format your response using Markdown with exactly three top-level sections:  
`# Evaluate`, `# Improved Hypothesis`, and `# Log`.  
Do not include any other sections or extra commentary.


The hypothesis follows:
<hypothesis>
{{ hypothesis }}
</hypothesis>

The examples follow:
{% for ex in dataset %}
<example id="{{ loop.index }}">
<query>{{ ex.query }}</query>
<answer>{{ ex.answer }}</answer>
</example>
{% endfor %}
"""

def extract_hypothesis(md_text):
    pattern = rf'^# {re.escape("Log")}\n(.*?)(?=^# |\Z)'
    match = re.search(pattern, md_text, re.DOTALL | re.MULTILINE)
    errors = int(match.group(1).strip()) if match else None
    
    if errors > 0:
        pattern = rf'^# {re.escape("Improved Hypothesis")}\n(.*?)(?=^# |\Z)'
        match = re.search(pattern, md_text, re.DOTALL | re.MULTILINE)
        improved_hypo = match.group(1).strip() if match else None
        assert improved_hypo, md_text
        
        match = re.search(r'<hypothesis>(.*?)</hypothesis>', improved_hypo, re.DOTALL)
        improved_hypo = match.group(1).strip() if match else None
        assert improved_hypo, md_text
    else:
        improved_hypo = None
    
    return improved_hypo, errors


def search_optimal_hypothesis(batch, log_jsonl=None, max_improvements=None, max_iters=None, init_hypothesis=None):
    log = []
    
    if not init_hypothesis:
        logger.info("generating initial hypothesis...")
        prompt = Template(init_template).render(dataset=batch)
        hypothesis = prompt_llm(prompt)
    else:
        prompt = None
        hypothesis = init_hypothesis
    
    lowest_error = None
    best_hypothesis = None
    n_improvements = 0
    n_iters = 0
    while True:
        logger.info("evaluating hypothesis...")
        eval_prompt = Template(eval_template).render(hypothesis=hypothesis, dataset=batch)
        hypothesis_eval = prompt_llm(eval_prompt)
        improved_hypothesis, n_errors = extract_hypothesis(hypothesis_eval)
        if not lowest_error:
            lowest_error = n_errors
        else:
            if n_errors < lowest_error:
                lowest_error = n_errors
                best_hypothesis = hypothesis
                n_improvements += 1
        log.append({"prompt": prompt, "hypothesis": hypothesis, "n_errors": n_errors})
        
        if log_jsonl:
            write_jsonl(log_jsonl, log)
            
        logger.info(f"#errors: {n_errors}")
        if n_errors == 0:
            break
        if max_improvements and max_improvements <= n_improvements:
            logger.debug("max_improvements reached")
            break
        if max_iters and max_iters <= n_iters:
            logger.debug("max_iters reached")
            break
        
        hypothesis = improved_hypothesis
        n_iters += 1
        
    return log, lowest_error, best_hypothesis


def batch_opt(batches, log_jsonl):
    hypothesis = None
    iteration = 0
    log = []
    while True:
        logger.info(f"--- ITERATION: {iteration}")
        lowest_errors = []
        first_errors = []
        for batch_id, batch in enumerate(batches):
            logger.info(f" -- BATCH: {batch_id}")
            batch_log, lowest_error, best_hypothesis = search_optimal_hypothesis(batch, 
                                                                                 init_hypothesis=hypothesis,
                                                                                 max_iters=5,
                                                                                 max_improvements=1)
            log.append({
                "iteration": iteration, 
                "batch_id": batch_id, 
                "lowest_error": lowest_error, 
                "best_hypothesis": best_hypothesis, 
                "log": batch_log})
            hypothesis = best_hypothesis
            write_jsonl(log_jsonl, log)
            lowest_errors.append(lowest_error)
            first_errors.append(batch_log[0]["n_errors"])
        logger.info(f"mean lowest error: {np.mean(lowest_errors)}, first error: {np.mean(first_errors)}")
        iteration += 1
        
        
if __name__ == "__main__":
    data = read_json("data/ethos/ethos_multilabel.json")["examples"]
    dataset = [{"query": ex["query"], "answer": ex["answer"]["answer"]} for ex in data]
    batches = list(chunked(data[:100], 20))
    batch_opt(batches, "EXP/hypotheses_log_qwen.jsonl")