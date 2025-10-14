import re
from jinja2 import Template
import json
from jsonschema import validate, ValidationError
from loguru import logger

from prompt_opt.hypoopt.streaming_completion import create_completion_streaming

def md_extract_code_block(text, lang="json") -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    start_marker = f"```{lang}"
    end_marker = "```"

    start_index = text.find(start_marker)
    if start_index != -1:
        start_index += len(start_marker)

        end_index = text.rfind(end_marker)
        if end_index == -1 or end_index <= start_index:
            return None

        json_str = text[start_index:end_index].strip()
    else:
        json_str = text

    try:
        json_data = json.loads(json_str)
    except:
        logger.debug(f"Can't convert to '{lang}'")
        return None

    return json_data


def validate_json(json_data, schema):
    try:
        validate(instance=json_data, schema=schema)
    except ValidationError as e:
        logger.debug(f"JSON validation error: {e.message}")
        return False
    return True


class PromptDefault:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def prompt(self, prompt, schema=None):
        kwopts = {}
        if schema:
            kwopts["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema},
                # "strict": True
            }

            # kwopts["format"] = {"type": "json_schema", "schema": schema}
        messages = [
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwopts,
        )
        content = completion.choices[0].message.content
        content = content if content else completion.choices[0].message.reasoning_content
        if schema:
            content = json.loads(content)

        messages.append({"role": "assistant", "content": content})
        return content, messages


class PromptJSONByChat:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def prompt(self, prompt, schema, max_iters=5):
        # this version creates conversation with LLM to get valid JSON output
        # WITHOUT using constrained generation
        # meant for models not yet supporting it

        fix_json_template = """I need your answer in JSON with following schema:
    {{ schema }}
    """
        schema_str = json.dumps(schema, indent=3)

        messages = [{"role": "user", "content": prompt}]

        # use streaming as a workaround for Ollama timeouts
        completion = self.client.chat.completions.create(model=self.model, messages=messages)
        # completion = create_completion_streaming(self.client, model=self.model, messages=messages)
        
        content = completion.choices[0].message.content
        messages.append({"role": "assistant", "content": content})
        json_data = md_extract_code_block(content, lang="json")

        n_iters = 1
        while n_iters <= max_iters:
            if json_data and validate_json(json_data, schema):
                break
            else:
                logger.info(f"IT: {n_iters}")
                fix_json_prompt = Template(fix_json_template).render(schema=schema_str)
                messages.append({"role": "user", "content": fix_json_prompt})
                
                completion = self.client.chat.completions.create(model=self.model, messages=messages)
                # completion = create_completion_streaming(self.client, model=self.model, messages=messages)
                
                content = completion.choices[0].message.content
                messages.append({"role": "assistant", "content": content})
                json_data = md_extract_code_block(content, lang="json")
            n_iters += 1

        if not validate_json(json_data, schema):
            json_data = None
        return json_data, messages
