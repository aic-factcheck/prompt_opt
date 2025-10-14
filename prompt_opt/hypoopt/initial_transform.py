from jinja2 import Template
from loguru import logger


# rule_schema_transform = {
#     "type": "object",
#     "properties": {"title": {"type": "string"}, "rule": {"type": "string"}},
#     "required": ["title", "rule"],
#     "additionalProperties": False,
# }

def generate_initial_transform(llm, batch):
    init_template = '''**Inferring Query-to-Answer Transformation – Python Implementation**

You will receive **one or more** `(query, answer)` example pairs.

Your task is to write a **single Python function**:

```python
def transform(query: str) -> str:
    ...
```

This function must take a `query` string as input and return the correct transformed `answer` string.

**Rules and constraints:**

1. **Allowed resources**:

   * Python standard library
   * Jinja2 templates
   * The following helper function:

     ```python
     def ask_llm(prompt: str, output_json_schema: object):
         """Send `prompt` to another LLM (likely less capable than you) and receive a JSON-formatted response matching `output_json_schema`."""
         return <JSON formatted response>
     ```

2. **Generalization over memorization**:

   * The `transform` function should work for **unseen queries**, even using different languages.
   * Exact pattern matching (e.g., using `re` module) is PROHIBITED use `ask_llm` for semantic matching instead.
   * The `prompt` for `ask_llm` must be ZERO-SHOT, i.e., do not copy examples from the `query`.
   * Use Jinja2 templates if prompt needs any parameters.

3. **Output format**:

   * Your final answer must contain **exactly one** Python code block.
   * The code block should contain a **complete, working implementation** of `transform` using the provided examples as guidance.
   * The only exception is the `ask_llm` method - do not output any placeholder for it.

**Examples**:
{% for ex in dataset %}
<example id="{{ loop.index }}">
<query>{{ ex.query }}</query>
<answer>{{ ex.answer }}</answer>
</example>
{% endfor %}

**Now, output the complete Python implementation of `transform`.**
'''

    init_prompt = Template(init_template).render(dataset=batch)
    logger.debug(init_prompt)
    response, messages = llm.prompt(init_prompt, None)
    return response, messages


def generate_initial_transform_with_decompose(llm, batch):
    init_template = '''# Inferring and Decomposing Query-to-Answer Transformations – Python Implementation

You will receive **one or more** `(query, answer)` example pairs.

Your task:
Write a **single Python function**:

```python
def transform(query: str) -> str:
    ...
```

This function must take a `query` string as input and return the correctly transformed `answer` string.

---

## **Core Objective: Transformation Decomposition**

You are **not** just guessing an answer — you are **inferring the transformation process** that maps the query to the answer.
Your implementation should:

* Identify and explicitly decompose the transformation into **clear, logical steps** (e.g., extract entities → normalize → reformat → translate).
* Use **multiple sub-calls to `ask_llm`** when different reasoning steps are needed.
* Treat transformations as **semantic** (meaning-based) rather than purely textual.
* Be robust enough to handle **unseen queries** (possibly in other languages).

---

## **Rules and Constraints**

1. **Allowed Resources**:

   * Python standard library
   * Jinja2 templates
   * The following helper function:

     ```python
     def ask_llm(prompt: str, output_json_schema: object):
         """Send `prompt` to another LLM (likely less capable than you) and receive a JSON-formatted response matching `output_json_schema`."""
         return <JSON formatted response>
     ```

2. **Generalization over Memorization**:

   * **Do not** hardcode query→answer mappings.
   * Exact string or regex matching (`re` module) is **prohibited**.
   * Always prefer **semantic prompting** with `ask_llm`.
   * Prompts for `ask_llm` must be **zero-shot** — never paste example queries or answers into them.
   * Use **Jinja2** templates if a prompt needs dynamic parameter insertion.

3. **Transformation Decomposition Guidance**:

   * Think of each `ask_llm` call as a **subroutine** that handles a single, well-defined transformation step.
   * If the task involves:

     * Rewording → call `ask_llm` with a rephrasing prompt.
     * Extracting key information → call `ask_llm` with an extraction schema.
     * Formatting → call `ask_llm` with formatting instructions.
   * Chain these steps together so the transformation logic is explicit in code.

4. **Output Format**:

   * Your answer must contain **exactly one** Python code block.
   * The code block must include a **complete, working** implementation of `transform`.
   * You may **call** `ask_llm`, but must not define it.
   * No explanatory text outside the code block.

---

### **Examples**:

{% for ex in dataset %}
<example id="{{ loop.index }}">
<query>{{ ex.query }}</query>
<answer>{{ ex.answer }}</answer>
</example>
{% endfor %}
---

**Now, output the complete Python implementation of `transform`, ensuring the transformation is clearly decomposed into multiple reasoning steps where applicable.**
'''

    init_prompt = Template(init_template).render(dataset=batch)
    logger.debug(init_prompt)
    response, messages = llm.prompt(init_prompt, None)
    return response, messages


def generate_initial_transform_with_classes(llm, batch):
    init_template = '''# Inferring and Decomposing Query→Answer Transformations – Python Implementation

You will be given **one or more** `(query, answer)` example pairs.

Your goal: **Infer the transformation process** that maps each query to its corresponding answer, then implement it as **one complete Python function**:

```python
def transform(query: str) -> str:
    ...
```

---

## **Objective: Transformation Process Discovery**

You are **not** guessing answers — you must **explicitly identify, decompose, and code the transformation steps** that convert a given `query` into the correct `answer`.

Your implementation must:

* Break the process into **clear, sequential, semantic steps** (e.g., *extract entities → normalize → reformat → translate*).
* Use **multiple sub-calls to `ask_llm`** when different reasoning steps are needed.
* Treat transformations as **meaning-based** (semantic) rather than purely textual.
* Work robustly for **unseen queries**, potentially in other languages.

---

## **Rules & Constraints**

1. **Allowed Tools**

   * Python **standard library**
   * **Jinja2** templates
   * The following provided helper function:

     ```python
     def ask_llm(prompt: str, output_json_schema: object):
         """Send `prompt` to another LLM (less capable than you) and receive a JSON-formatted response matching `output_json_schema`."""
         return <JSON-formatted response>
     ```

2. **Generalization, Not Memorization**

   * **Never** hardcode query→answer mappings.
   * **Prohibited**: exact string matching, regex matching (`re` module).
   * Prefer **semantic prompting** via `ask_llm`.
   * Prompts for `ask_llm` must be **zero-shot** — no training examples embedded.
   * If prompts require dynamic data, use **Jinja2 templates**.

3. **Encapsulation Requirement**

   * Every `ask_llm` call must be inside a dedicated **ASK-class**:

     * Class name: starts with `ASK` followed by a brief description.
     * Must have **two methods**:

       ```python
       def ask(self, input):
           # Prepare input, build prompt & schema, call ask_llm, postprocess result
           ...

       def dataset(self):
           # Return synthesized (query, answer) pairs allowing to optimize the subproblem represented by this class later
           # → Base them directly on the <example> records provided at the end of this prompt
           # → Preserve 1-to-1 correspondence in most cases
           # → The number of synthesized pairs will be the same (or larger) than the number of <example> records below
           return [
               {"query": <synthetic_query>, "answer": <synthetic_answer>},
               ...
           ]
       ```

4. **Step-by-Step Decomposition**

   * Each `ask_llm` call should handle **one distinct transformation step**.
   * Example mapping:

     * *Rewording*: prompt for paraphrase.
     * *Information extraction*: prompt with extraction schema.
     * *Formatting*: prompt with explicit formatting instructions.
   * Chain subroutines to make the transformation logic **transparent and traceable**.

5. **Output Requirements**

   * Your response must include **exactly one** Python code block.
   * That block must contain:

     * All ASK-classes you define.
     * A complete, functional `transform(query)` implementation.
   * You may **call** `ask_llm`, but must not redefine it.
   * No explanation outside the code block.

---

## **Examples**

{% for ex in dataset %}
<example id="{{ loop.index }}">
<query>{{ ex.query }}</query>
<answer>{{ ex.answer }}</answer>
</example>
{% endfor %}

---

**Now, output the complete Python implementation of `transform`, ensuring the transformation is clearly decomposed into multiple reasoning steps where applicable.**
'''

    init_prompt = Template(init_template).render(dataset=batch)
    logger.debug(init_prompt)
    response, messages = llm.prompt(init_prompt, None)
    return response, messages