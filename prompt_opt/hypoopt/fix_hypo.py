from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template
from loguru import logger
from aic_nlp_utils.json import read_json

def fix_hypothesis(llm, data, rules, template_src="fix_classify_v1", single=False):
  cfg_dir = "data/templates/hypo/fix_hypo"
  env = Environment(loader=FileSystemLoader(cfg_dir))
  template = env.get_template(f"{template_src}.txt.jinja")
  dataset_with_preds = [{"query": ex["query"], "gold": ex["gold"], "pred": ex["pred"]} for ex in data]
  prompt = template.render(dataset=dataset_with_preds, rules=rules)
  schema = read_json(Path(cfg_dir, f"{template_src}_schema.json"))
  hypothesis, messages = llm.prompt(prompt, schema)
  if single:
    hypothesis = {
      "rules": [{"rule_id": "R0", "title": hypothesis["title"], "rule": hypothesis["hypothesis"]}]
    }
  return hypothesis, messages
