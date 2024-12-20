import json
from jinja2 import Template

from .agent_chat import AgentChat


class AgentJSONDirect:
    def __init__(self, parent: AgentChat):
        self._parent = parent
        
        
    def history(self):
        return self._parent.history()
        
        
    def add_user(self, prompt: str):
        self._parent.add_user(prompt)
        
        
    def add_assistant(self, prompt: str):
        self._parent.add_assistant(prompt)
        
        
    def query(self, prompt: str, schema, **kwargs):
        model_response = self._parent.query(prompt, guided_json=schema, **kwargs)
        try:
            return json.loads(model_response)
        except Exception as e:
            return {"error": str(e)}
    
    
class AgentJSONSteppedCoT:
    def __init__(self, parent: AgentChat):
        self._parent = parent
        
        
    def history(self):
        return self._parent.history()
        
        
    def add_user(self, prompt: str):
        self._parent.add_user(prompt)
        
        
    def add_assistant(self, prompt: str):
        self._parent.add_assistant(prompt)
        
        
    def query(self, think_prompt: str, result_prompt: str, schema, **kwargs):
        think_response = self._parent.query(think_prompt, **kwargs)
        result = self._parent.query(result_prompt, guided_json=schema, **kwargs)
        return {"think": think_response, "pred": json.loads(result)}


# # slow
# trr_schema = {
#     "type": "object",
#     "properties": {
#         "thinking": {"type": "string", "minLength": 500, "maxLength": 5000},
#         "reflection": {"type": "string", "minLength": 500, "maxLength": 5000},
#         "response": {"type": "string", "minLength": 500, "maxLength": 5000},
#     },
#     "required": ["thinking", "reflection", "response"]
# }