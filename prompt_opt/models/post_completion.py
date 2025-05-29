import time

from prompt_opt.models.llm_predictor import LLMOutput

from prompt_opt.utils import extract_response_dseek, pf, ld, lw


class BasicCompletionPostProcessor:
    def __init__(self, cfg):
        pass
    
    def postprocess(self, completion, duration_completion: float) -> LLMOutput:
        st = time.time()
        msg = completion.choices[0].message
        content = msg.content
        reasoning_content = msg.reasoning_content if hasattr(msg, "reasoning_content") else None
        
        if content is None and reasoning_content is not None:
            content = reasoning_content
            reasoning_content = "ERROR: The content was empty, only reasoning_content was returned."

        duration = time.time() - st
        return LLMOutput(content=content, reasoning_content=reasoning_content, duration=duration + duration_completion)
    
    
class SimulatedThinkingCompletionPostProcessor:
    def __init__(self, cfg):
        pass
    
    def postprocess(self, completion, duration_completion: float) -> LLMOutput:
        st = time.time()
        msg = completion.choices[0].message
        content = msg.content
        # ld(content)
        res = extract_response_dseek(content)
        if res["answer"] == "":
            answer, think = res["think"], ""
        else:
            answer, think = res["answer"], res["think"]
        # ld("answer=\n", answer)
        # ld("think=\n", think)
        duration = time.time() - st
        return LLMOutput(content=answer, reasoning_content=think, duration=duration_completion + duration)