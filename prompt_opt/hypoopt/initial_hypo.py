from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template
from loguru import logger
from aic_nlp_utils.json import read_json

def generate_initial_hypo(llm, batch, init_template_src="modular_v1", single=False):
  cfg_dir = "data/templates/hypo/initial_hypo"
  env = Environment(loader=FileSystemLoader(cfg_dir))
  init_template = env.get_template(f"{init_template_src}.txt.jinja")
  init_prompt = init_template.render(dataset=batch)
  schema = read_json(Path(cfg_dir, f"{init_template_src}_schema.json"))
  init_hypothesis, messages = llm.prompt(init_prompt, schema)
  if single:
    init_hypothesis = {
      "rules": [{"rule_id": "R0", "title": init_hypothesis["title"], "rule": init_hypothesis["hypothesis"]}]
    }
  return init_hypothesis, messages
