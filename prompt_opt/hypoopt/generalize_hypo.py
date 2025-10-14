from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template
from loguru import logger
from aic_nlp_utils.json import read_json

def generalize_hypo(llm, rules, template_src="generalize_v1", single=False):
  cfg_dir = "data/templates/hypo/generalize_hypo"
  env = Environment(loader=FileSystemLoader(cfg_dir))
  template = env.get_template(f"{template_src}.txt.jinja")
  prompt = template.render(rules=rules)
  schema = read_json(Path(cfg_dir, f"{template_src}_schema.json"))
  hypothesis, messages = llm.prompt(prompt, schema)
  if single:
    hypothesis = {
      "rules": [{"rule_id": "R0", "title": hypothesis["title"], "rule": hypothesis["hypothesis"]}]
    }
  return hypothesis, messages
