from jinja2 import Template
from loguru import logger
from tqdm import tqdm

eval_template = """# ✅ Task: Evaluate Hypothesis Consistency for a Query-Answer Example

You will be given:

1. A **hypothesis**, which is a list of transformation rules (each with a unique `rule_id`) that describe how to transform a query into an answer.
2. A **single example**, consisting of a `query` and its corresponding `answer`.

---

### 🎯 Your Goal:

Determine whether the example is **consistent** with the hypothesis.

If **any rules were broken** in the transformation from query to answer, identify them.

---

### 🔍 What to Return:

A JSON object listing the **broken rule(s)** with:

* `rule_id`: The ID of the broken rule.
* `explanation`: A brief explanation of **how** and **why** the rule was violated.

---

### 🔤 Input Format:

First, you will see the hypothesis:

```xml
<hypothesis>
{% for rule in rules %}
  <rule id="{{ rule.rule_id }}">{{ rule.rule }}</rule>
{% endfor %}
</hypothesis>
```

Then, the example:

```xml
<query>{{ example.query }}</query>
<answer>{{ example.answer }}</answer>
```

---

### 📤 Output Format:

Return your evaluation in the following JSON structure:

```json
{
  "rules_broken": [
    {
      "rule_id": "<broken rule ID>",
      "explanation": "<short explanation of why the rule was broken>"
    }
    // ... more broken rules if any
  ]
}
```

If **no rules were broken**, return:

```json
{
  "rules_broken": []
}
```
"""

eval_schema = {
    "type": "object",
    "properties": {
        "rules_broken": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rule_id": {
                        "type": "string",
                    },
                    "explanation": {
                        "type": "string",
                    },
                },
                "required": ["rule_id", "explanation"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rules_broken"],
    "additionalProperties": False,
}

def eval_single(llm, batch, rules):
    evals = []
    for example in tqdm(batch):
        eval_prompt = Template(eval_template).render(example=example, rules=rules)
        hypothesis_eval, _ = llm.prompt(eval_prompt, schema=eval_schema)
        evals.append(hypothesis_eval)
    return evals