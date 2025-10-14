from jinja2 import Template
from loguru import logger
from tqdm import tqdm

improve_template = """
# ✅ Task: Rewrite a Hypothesis Rule Based on Query-Answer Examples

You will be given the following:

1. A **hypothesis**, which is a list of transformation rules (each with a unique `rule_id`) that describe how to transform a query into an answer.
2. A **target rule**, which is a specific rule from the hypothesis selected for improvement (or complete rewrite).
3. A list of **breaking examples**, which are query-answer pairs that likely **violate the target rule**.
4. A list of **consistent examples**, which are query-answer pairs that likely **follow the target rule** (though they may violate other rules).

---

### 🎯 Your Goal:

- Analyze the provided examples and the hypothesis.
- Focus on identifying issues in the **target rule**.
- Use the **breaking examples** to identify why and how the rule fails.
- Use the **consistent examples** to identify what the rule is trying to capture correctly.
- **Rewrite the target rule from scratch** so the issue gets fixed.
- Do not worry if the new rule becomes inconsistent with other rules -- these will be fixed later.

You may use other hypothesis rules as context, but you **can not** modify them. Fix only the target rule.

---

### 🔍 What to Return:

Return a JSON object with a single key, `target_rule`, containing the **revised version of the rule** as a string.

---

### 🔤 Input Format:

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

### 📤 Output Format:

Return the new **target rule** in this JSON format:

```json
{
  "target_rule": "<updated or completely changed target rule>"
}
```
"""


improve_schema = {
    "type": "object",
    "properties": {
        "target_rule": {
            "type": "string",
        }
    },
    "required": ["target_rule"],
    "additionalProperties": False,
}


def improve_rule(llm, rules, target_rule_id, evals, batch):
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
    updated_target_rule, messages = llm.prompt(improve_prompt, schema=improve_schema)
    return updated_target_rule, messages
