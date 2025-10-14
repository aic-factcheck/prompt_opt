from copy import deepcopy
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template
from loguru import logger
import numpy as np
from tqdm import tqdm

from aic_nlp_utils.json import read_json

def correct_hypothesis(llm, hypothesis, analysis, correct_template_src, single=False):
  cfg_dir = "data/templates/hypo/correct_hypo"
  env = Environment(loader=FileSystemLoader(cfg_dir))
  template = env.get_template(f"{correct_template_src}.txt.jinja")
  schema = read_json(Path(cfg_dir, f"{correct_template_src}_schema.json"))
  
  prompt = template.render(
      rules=hypothesis["rules"], corrections=analysis["rules"]
  )
  corrected, messages = llm.prompt(prompt, schema=schema)

  if single:
    corrected = {
      "rules": [{"rule_id": "R0", "title": corrected["title"], "rule": corrected["hypothesis"]}]
  }
      
  # merge back
  id2rec = {c["rule_id"]: c for c in corrected["rules"]}
  hypothesis = deepcopy(hypothesis)
  for r in hypothesis["rules"]:
    rid = r["rule_id"]
    if rid in id2rec:
      nr = id2rec[rid]
      assert len(nr) > 0
      r["title"] = nr["title"]
      r["rule"] = nr["rule"]
    
  return hypothesis, messages