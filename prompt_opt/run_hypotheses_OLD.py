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
    # default_model = "o4-mini"
    # default_model = "gpt-4o-mini"
    return client, default_model


def init_local():
    client = OpenAI(
        base_url = "http://g03:8333/v1",
        # base_url = "http://g07:8000/v1",
        api_key = "EMPTY"
    )
    models = client.models.list()
    logger.info("available models:")
    for model in models:
        logger.info(model.id)
    default_model = list(models)[0].id
        
    return client, default_model
    
    
client, default_model = init_openai()
# client, default_model = init_local()

    
def prompt_llm(prompt, model=default_model, schema=None):
    kwopts = {}
    if schema:
        kwopts["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "result", "schema": schema},
            # "strict": True
        }
    completion = client.chat.completions.create(
        model=model,
        messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ],
        # temperature=1.0,
        # top_p=0.95,
        **kwopts
    )
    # return completion
    content = completion.choices[0].message.content
    content = content if content else completion.choices[0].message.reasoning_content

    if schema:
        content = json.loads(content)
    return content


rule_schema = {
  "type": "object",
  "properties": {
    "rules": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "rule_id": { "type": "integer" },
          "rule": { "type": "string" }
        },
        "required": ["rule_id", "rule"],
        "additionalProperties": False
      }
    }
  },
  "required": ["rules"],
  "additionalProperties": False
}

init_template = """# Inferring Query-to-Answer Transformation Rules
You will be provided with one or more (query, answer) data pairs.
Your task is to generate a single most probable hypothesis that describe the reasoning or rules used to transform each query into its corresponding answer.
The hypothesis should offer a detailed explanation of the transformation process in a form of a rule set.
Notably the first rule should include general overview including a high-level pseudocode describing the transform.
All following rules should explain the details.

The output format JSON:
{
    "rules": [
        {"rule_id": 1, "rule": <text of a rule1, general overview and pseudocode>},
        {"rule_id": 2, "rule": <text of a rule2>},
        ...
    ]
}

The examples follow:
{% for ex in dataset %}
<example id="{{ loop.index }}">
<query>{{ ex.query }}</query>
<answer>{{ ex.answer }}</answer>
</example>
{% endfor %}
"""

eval_schema = {
    "type": "object",
    "properties": {
        "evaluation": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "example_id": {"type": "string"},
                    "rules_broken": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"rule_id": {"type": "integer"}, "explanation": {"type": "string"}},
                            "required": ["rule_id", "explanation"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["example_id", "rules_broken"],
                "additionalProperties": False,
            },
        },
        "improved_hypothesis": {
            "type": "object",
            "properties": {
                "rules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"rule_id": {"type": "integer"}, "rule": {"type": "string"}},
                        "required": ["rule_id", "rule"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["rules"],
            "additionalProperties": False,
        },
    },
    "required": ["evaluation", "improved_hypothesis"],
    "additionalProperties": False,
}


eval_template = """# Refine Query-to-Answer Transformation Hypothesis

You will receive one or more (query, answer) pairs along with a hypothesis that describes the transformation from query to answer.  
The hypothesis is expressed as a set of rules, each identified by a unique rule ID.

Your task has two parts:

---

## 1) **Evaluation**  
Evaluate whether each example is consistent with the given hypothesis.  
For any example that does **not** align with the hypothesis, list its ID and specify which rules were broken, along with a brief explanation for each.

Use the following JSON format:
```json
[
    {
        "example_id": <unaligned example ID>,
        "rules_broken": [
            {
                "rule_id": <broken rule ID, this rule must exist!>,
                "explanation": <short explanation of why the rule was broken>
            },
            ...
        ]
    },
    ...
]

## 2) **Improved Hypothesis**
If inconsistencies are found, revise the hypothesis accordingly.
Modify the text of existing rules (keeping their IDs) but do it **only** if needed.
If new rules are required, add them with new IDs in increasing order.
**Do not** annotate revised rules as changed in their text.
If no changes are needed, return an empty list of rules.

Use the following JSON format:
```json
{
    "rules": [
        {
            "rule_id": <existing or new rule ID>,
            "rule": <updated or new rule text>
        },
        ...
    ]
}
```

## 3) **Final Output**
Combine both parts into a single JSON object:
```json
{
    "evaluation": <Evaluation object from Part 1>,
    "improved_hypothesis": <Improved Hypothesis object from Part 2>
}
```

The hypothesis rules follow:
<hypothesis>
{% for rule in rules %}<rule id="{{ rule.rule_id }}">{{ rule.rule }}</rule>
{% endfor %}</hypothesis>

The examples follow:
{% for ex in dataset %}
<example id="{{ loop.index }}">
<query>{{ ex.query }}</query>
<answer>{{ ex.answer }}</answer>
</example>
{% endfor %}
"""


def update_hypothesis(hypothesis, hypothesis_eval):
    new_hypothesis = deepcopy(hypothesis)
    old_id2idx = {r["rule_id"]: idx for idx, r in enumerate(hypothesis["rules"])}
    # new_id2rule = {r["rule_id"]: r for r in hypothesis_eval["improved_hypothesis"]["rules"]}
    for new_rule in hypothesis_eval["improved_hypothesis"]["rules"]:
        new_id = new_rule["rule_id"]
        if new_id in old_id2idx:
            logger.debug(f'updating rule {new_rule["rule_id"]}')
            # print("=============")
            # pp(new_rule)
            new_hypothesis["rules"][old_id2idx[new_id]] = new_rule
        else:
            logger.debug(f'adding rule {new_rule["rule_id"]}')
            new_hypothesis["rules"].append(new_rule)
    return new_hypothesis        
    
    
def get_errors(hypothesis_eval):
    ev = hypothesis_eval["evaluation"]
    n_errors = len(ev)
    broken_rules = Counter([rb["rule_id"] for e in ev for rb in e["rules_broken"]])
    error_indices = [int(e["example_id"])-1 for e in ev] # this is internal batch index
    return n_errors, broken_rules, error_indices


def search_optimal_hypothesis(batch, log_jsonl=None, max_improvements=None, max_iters=None, init_hypothesis=None):
    log = []
    
    if not init_hypothesis:
        logger.info("generating initial hypothesis...")
        prompt = Template(init_template).render(dataset=batch)
        hypothesis = prompt_llm(prompt, schema=rule_schema)
    else:
        prompt = None
        hypothesis = init_hypothesis
    
    lowest_error = None
    best_hypothesis = None
    n_improvements = 0
    n_iters = 0
    while True:
        logger.info("evaluating & improving hypothesis...")
        eval_prompt = Template(eval_template).render(dataset=batch, rules=hypothesis["rules"])
        hypothesis_eval = prompt_llm(eval_prompt, schema=eval_schema)
        improved_hypothesis = update_hypothesis(hypothesis, hypothesis_eval)
        n_errors, broken_rules, error_indices = get_errors(hypothesis_eval)
        error_indices = [batch[eidx]["idx"] for eidx in error_indices] # convert internal batch index to unique sample id
        logger.debug(f'broken rules: {broken_rules}')
        logger.debug(f'error indices: {error_indices}')
        
        if not best_hypothesis:
            best_hypothesis = hypothesis
        
        if not lowest_error:
            lowest_error = n_errors
        else:
            if n_errors < lowest_error:
                lowest_error = n_errors
                best_hypothesis = hypothesis
                n_improvements += 1
        log.append({"prompt": prompt, "hypothesis": hypothesis, 
                    "n_errors": n_errors, "broken_rules": broken_rules, "error_indices": error_indices, 
                    "eval_prompt": eval_prompt, "hypothesis_eval": hypothesis_eval})
        
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
    while iteration < 5:
        logger.info(f"--- ITERATION: {iteration}")
        lowest_errors = []
        first_errors = []
        for batch_id, batch in enumerate(batches):
            logger.info(f" -- BATCH: {batch_id}")
            batch_log, lowest_error, best_hypothesis = search_optimal_hypothesis(batch, 
                                                                                 init_hypothesis=hypothesis,
                                                                                 max_iters=5,
                                                                                 max_improvements=None)
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
    dataset = [{"query": ex["query"], "answer": ex["answer"]["answer"], "idx": ex["idx"]} for ex in data]
    batches = list(chunked(data[:100], 10))
    # batch_opt(batches, f"EXP/hypotheses_rules_o3mini_log.jsonl")
    batch_opt(batches, f"EXP/hypotheses_rules_o3mini_MI_log.jsonl") # max_improvements=None