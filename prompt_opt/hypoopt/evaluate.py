import json
from joblib import Parallel, delayed
from pathlib import Path
import time

from jinja2 import Environment, FileSystemLoader, Template
from loguru import logger
import numpy as np
from tqdm import tqdm


from aic_nlp_utils.json import read_json

from prompt_opt.metrics.object_aligner import ObjectAligner

instructions_template = """# {{ title }}
{{ overview }}

{% for rule in rules %}
## {{ rule.title }}
{{ rule.rule }}
{% endfor %}
"""


def hypothesis_to_instructions(hypothesis, instructions_template=instructions_template):
    rules = hypothesis["rules"]
    instructions = Template(instructions_template).render(
        title=rules[0]["title"],
        overview=rules[0]["rule"],
        rules=rules[1:])
    return instructions


instructions_template2 = """# {{ title }}
{{ overview }}

---
{% for rule in rules %}
## {{ rule.title }} (rule {{ rule.rule_id }})
{{ rule.rule }}
{% endfor %}
"""


def hypothesis_to_instructions2(hypothesis, instructions_template=instructions_template2):
    rules = hypothesis["rules"]
    instructions = Template(instructions_template).render(
        title=rules[0]["title"],
        overview=rules[0]["rule"],
        rules=rules[1:])
    return instructions

init_template_titles = """{{ instructions }}

## Query to transform
Now use the above instructions to find an answer for the following query:
<query>{{ query }}</query>

Answer using this JSON schema:
{{ schema }}
"""


# def predict_samples(llm, instructions, batch, predict_schema):
#     predict_schema_str = json.dumps(predict_schema, indent=3)
#     answers = []
#     messages_lst = []
#     for ex in tqdm(batch):
#         prompt = Template(init_template_titles).render(instructions=instructions.strip(), query=ex["query"], schema=predict_schema_str)
#         answer, messages = llm.prompt(prompt, predict_schema)
#         answers.append(answer)
#         messages_lst.append(messages)
#     return answers, messages


def predict_samples(llm, instructions, batch, predict_schema, n_jobs=0):
    """
    Run predictions in parallel (if n_jobs > 1) or sequentially (if n_jobs == 1) using joblib.
    For n_jobs == 0, there is a job for each sample, i,e., n_jobs = len(batch)
    Suitable for I/O-bound work like API calls.
    """
    if n_jobs == 0:
        n_jobs = len(batch)
        
    predict_schema_str = json.dumps(predict_schema, indent=3)

    def worker(iex):
        idx, ex = iex
        logger.debug(f"predict_samples start {idx+1}/{len(batch)}")
        st = time.time()
        prompt = Template(init_template_titles).render(
            instructions=instructions.strip(),
            query=ex["query"],
            schema=predict_schema_str
        )
        res = llm.prompt(prompt, predict_schema)
        duration = time.time() - st
        logger.debug(f"predict_samples done {idx+1}/{len(batch)} in {duration:.1f}s")
        return res

    # joblib handles n_jobs=1 as sequential automatically
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(worker)((idx, ex)) for idx, ex in enumerate(batch)
    )

    # Unpack answers and messages
    answers, messages_lst = zip(*results)
    return list(answers), list(messages_lst)



def compute_string_accuracy(preds, batch, fn=lambda x: x, verbose=False):
    
    assert len(preds) >= len(batch)
    correct_indices = []
    wrong_indices = []
    for idx, (pred, gold) in enumerate(zip(preds, batch)):
        p = pred.get("answer", "ERROR") if pred else "ERROR"
        g = gold["answer"]["answer"]
        try:
            p = fn(p)
        except:
            p = "ERROR"
        g = fn(g)
        # print(p, g, p==g, type(p), type(g))
        if p == g:
            correct_indices.append(idx)
        else:
            wrong_indices.append(idx)
            if verbose:
                print(f"P:{p}", f"G:{g}", gold["query"])
            
    accuracy = len(correct_indices)/len(preds)
    return accuracy, correct_indices, wrong_indices


def compute_object_aligner_accuracy(preds, batch, oa_schema, fn=lambda x: x, correct_threshold=1.0):
    oa = ObjectAligner(id_="object_aligner_metric", schema=oa_schema)
    assert len(preds) >= len(batch)
    scores = []
    correct_indices = []
    wrong_indices = []
    for idx, (pred, gold) in enumerate(zip(preds, batch)):
        p = pred
        g = gold["answer"]
        try:
            p = fn(p)
        except:
            p = "ERROR"
        g = fn(g)
        score = float(oa.metric(g, p)["score"])
        scores.append(score)
        if score >= correct_threshold:
            correct_indices.append(idx)
        else:
            wrong_indices.append(idx)

    mean_score = np.mean(scores)
    return float(mean_score), correct_indices, wrong_indices
    
    
def get_ethos_accs(answers, batch):
    total_acc = np.round(compute_string_accuracy(answers, batch) * 100.0)
    A = np.array([
        compute_string_accuracy(answers, batch, verbose=False, fn=lambda x: x.split(",")[i])[0] for i in range(7)])
    A = np.round(100.*A)
    return f"ETHOS {total_acc}: V:{A[0]}%, DG:{A[1]}%, G:{A[2]}%, R:{A[3]}%, N:{A[4]}%, D:{A[5]}%, S:{A[6]}%"