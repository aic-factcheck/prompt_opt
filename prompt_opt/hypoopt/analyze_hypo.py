import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template
from loguru import logger
from tqdm import tqdm

from aic_nlp_utils.json import read_json


def analyze_hypothesis_split(llm, rules, evals, batch, analyze_split_template_src):
  # this does not show gold values only consistent/inconsistent samples
  # NOTE: not tested after moving here from the notebook
  breaking = []
  consistent = []
  for ev, ex in zip(evals, batch):
      rules_broken = [rb["rule_id"] for rb in ev["rules_broken"]]
      if len(rules_broken) > 0:
          breaking.append({"example_id": f"B{len(breaking)+1}", "query": ex["query"], "answer": ex["answer"]})
      else:
          consistent.append({"example_id": f"C{len(consistent)+1}", "query": ex["query"], "answer": ex["answer"]})
            
  cfg_dir = "data/templates/hypo/analyse_hypo"    
  env = Environment(loader=FileSystemLoader(cfg_dir))
  template = env.get_template(f"{analyze_split_template_src}.txt.jinja")
  schema = read_json(Path(cfg_dir, f"{analyze_split_template_src}_schema.json"))
  
  analysis_prompt = template.render(
      rules=rules, breaking=breaking, consistent=consistent
  )
  analysis, messages = llm.prompt(analysis_prompt, schema=schema)
  return analysis, messages



def analyze_hypothesis(llm, hypothesis, answers, batch, analyze_template_src, single=False):
  examples = []
  for i, (answer, ex) in enumerate(zip(answers, batch)):
    gold = json.dumps(ex["answer"], indent=3)
    pred = json.dumps(answer, indent=3)
    examples.append({"example_id": f"E{i+1}", "query": ex["query"], "gold": gold, "prediction": pred})
          
  cfg_dir = "data/templates/hypo/analyze_hypo"    
  env = Environment(loader=FileSystemLoader(cfg_dir))
  template = env.get_template(f"{analyze_template_src}.txt.jinja")
  schema = read_json(Path(cfg_dir, f"{analyze_template_src}_schema.json"))
  
  prompt = template.render(
      rules=hypothesis["rules"], examples=examples
  )
  analysis, messages = llm.prompt(prompt, schema=schema)
  
  if single:
    analysis = {
      "rules": [{"rule_id": "R0", "explanation": analysis["explanation"], "correction": analysis["correction"]}]
    }
  return analysis, messages