from jinja2 import Template
from loguru import logger
from tqdm import tqdm

improve_template = """# ✅ Task: Rewrite a Hypothesis Rule Based on Query-Answer Examples

You are given the following inputs:

1. A **hypothesis**: a list of transformation rules (each with a unique `rule_id`) that describe how to transform a **query** into an **answer**.
2. A **target rule**: one specific rule from the hypothesis that needs improvement.
3. A list of **breaking examples**: query-answer pairs that likely **violate** the target rule.
4. A list of **consistent examples**: query-answer pairs that likely **follow** the target rule (but may violate other rules).

---

### 🎯 Objective

- Carefully analyze the **target rule** and how it performs on the provided examples.
- Use **breaking examples** to identify the flaws in the current rule.
- Use **consistent examples** to understand what the rule is attempting to capture correctly.
- **Rewrite the target rule** to better align with both the breaking and consistent examples.
- If necessary, update other rules in the hypothesis that are also affected by this change.

---

### 🔍 What to Return

Output a JSON object with updated rule definitions. This must include at least one entry for the **target rule**, and may include additional updates for other impacted rules.

---

### 🔤 Input Format

The **hypothesis**:

```xml
<hypothesis>
{% for rule in rules %}
  <rule id="{{ rule.rule_id }}">{{ rule.rule }}</rule>
{% endfor %}
</hypothesis>
````

The **target rule**:

```xml
<target-rule id="{{ target_rule.rule_id }}">{{ target_rule.rule }}</target-rule>
```

The **breaking examples**:

```xml
<breaking>
{% for example in breaking %}
  <example id="{{ example.example_id }}">
    <query>{{ example.query }}</query>
    <answer>{{ example.answer }}</answer>
  </example>
{% endfor %}
</breaking>
```

The **consistent examples**:

```xml
<consistent>
{% for example in consistent %}
  <example id="{{ example.example_id }}">
    <query>{{ example.query }}</query>
    <answer>{{ example.answer }}</answer>
  </example>
{% endfor %}
</consistent>
```

---

### 📤 Output Format

Return a JSON object like this:

```json
{
  "rules": [
    {
      "rule_id": "<target_rule_id>",
      "rule": "<updated target rule>"
    },
    {
      "rule_id": "<other_rule_id>",
      "rule": "<updated related rule, if needed>"
    }
  ]
}
```
"""


improve_schema = {
    "type": "object",
    "properties": {
        "rules": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "rule_id": {"type": "string", "description": "The unique identifier of the rule being updated"},
                    "rule": {"type": "string", "description": "The updated text of the rule"},
                },
                "required": ["rule_id", "rule"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rules"],
    "additionalProperties": False,
}


def improve_hypothesis(llm, rules, target_rule_id, evals, batch):
    target_rule = [r for r in rules if r["rule_id"] == target_rule_id]
    assert len(target_rule) == 1
    target_rule = target_rule[0]
    logger.debug(f"improving: {target_rule}")
    breaking = []
    consistent = []
    for ev, ex in zip(evals, batch):
        rules_broken = [rb["rule_id"] for rb in ev["rules_broken"]]
        if target_rule_id in rules_broken:
            breaking.append({"example_id": f"B{len(breaking)+1}", "query": ex["query"], "answer": ex["answer"]})
        else:
            consistent.append({"example_id": f"C{len(consistent)+1}", "query": ex["query"], "answer": ex["answer"]})
            
    improve_prompt = Template(improve_template).render(
        rules=rules, target_rule=target_rule, breaking=breaking, consistent=consistent
    )
    updated_target_rules, messages = llm.prompt(improve_prompt, schema=improve_schema)
    return updated_target_rules, messages
