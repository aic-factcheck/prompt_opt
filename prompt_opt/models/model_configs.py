def get_llama8b(gpus=[0]):
    return {
        "short": "llama8b",
        "name": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "gpus": gpus,
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 65536},
    }


def get_llama70b(gpus=[0, 1]):
    return {
        "short": "llama70b",
        "name": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        "gpus": gpus,
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 65536},
    }


def get_llama33_70b(gpus=[0, 1]):
    return {
        "short": "llama33_70b",
        "name": "casperhansen/llama-3.3-70b-instruct-awq",
        "gpus": gpus,
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 65536},
    }


def get_llama33_70b_full(gpus=[0, 1, 2, 3, 4, 5, 6, 7]):
    return {
        "short": "llama33_70bF",
        "name": "meta-llama/Llama-3.3-70B-Instruct",
        "gpus": gpus,
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 16384},
    }


def get_llama33_70b_v2(gpus=[0, 1]):
    return {
        "short": "llama33_70b_v2",
        "name": "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
        "gpus": gpus,
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 65536},
    }


def get_ll32_3b(gpus=[0]):
    return {
        "short": "ll32_3b",
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "gpus": gpus,
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 131072},
    }


def get_crp104b(gpus=[0, 1]):
    return {
        "short": "crp104b",
        "name": "aixsatoshi/c4ai-command-r-plus-08-2024-awq",
        "gpus": gpus,
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 16384},
    }


def get_ayaexp8b(gpus=[0]):
    return {
        "short": "ayaexp8b",
        "name": "CohereForAI/aya-expanse-8b",
        "gpus": gpus,
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 8192},
    }


def get_ayaexp32b(gpus=[0, 1]):
    return {
        "short": "ayaexp32b",
        "name": "CohereForAI/aya-expanse-32b",
        "gpus": gpus,
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 16384},
    }


def get_minis8b(gpus=[0]):
    return {  # not working for outlines
        "short": "minis8b",
        "name": "mistralai/Ministral-8B-Instruct-2410",
        "gpus": gpus,
        "template_dir": "data/templates/agents",
        "vllm_opts": {
            "gpu-memory-utilization": 0.95,
            "max-model-len": 32768,
            "tokenizer_mode": "mistral",
            "config_format": "mistral",
            "load_format": "mistral",
        },
    }


def get_qwen25_72b(gpus=[0, 1]):
    return {
        "short": "qwen25_72b",
        "name": "Qwen/Qwen2.5-72B-Instruct-AWQ",
        "gpus": gpus,
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 32768},
    }


def get_gem2_27b(gpus=[0, 1]):
    return {
        "short": "gem2_27b",
        "name": "google/gemma-2-27b-it",
        "gpus": gpus,
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 4096},
    }


def get_phi4_14b(gpus=[0]):
    return {
        "short": "phi4_14b",
        "name": "stelterlab/phi-4-AWQ",
        "gpus": gpus,
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 16384},
    }


def get_dseek_llama8b(gpus=[0], reasoning=False):
    extra_opts = {"enable-reasoning": None, "reasoning-parser": "deepseek_r1"} if reasoning else {}
    return {
        "short": "dseek_llama8b",
        "name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "gpus": gpus,
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 65536, "enable-prefix-caching": None, **extra_opts},
    }


def get_dseek_llama70b(gpus=[0, 1], reasoning=False):
    extra_opts = {"enable-reasoning": None, "reasoning-parser": "deepseek_r1"} if reasoning else {}
    return {
        "short": "dseek_llama70b",
        # "name": "casperhansen/deepseek-r1-distill-llama-70b-awq",
        "name": "/home/drchajan/models/casperhansen/deepseek-r1-distill-llama-70b-awq",
        "gpus": gpus,
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 65536, "enable-prefix-caching": None, **extra_opts},
    }

def get_gemma3_12b(gpus=[0], reasoning=False):
    extra_opts = {"enable-reasoning": None, "reasoning-parser": "deepseek_r1"} if reasoning else {}
    return {
        "short": "gemma3_12b",
        "name": "google/gemma-3-12b-it",
        "gpus": gpus,
        "sampling_params": {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "min_p": 0.0},
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 65536, **extra_opts},
    }


def get_gemma3_27b(gpus=[0], reasoning=False):
    extra_opts = {"enable-reasoning": None, "reasoning-parser": "deepseek_r1"} if reasoning else {}
    return {
        "short": "gemma3_27b",
        "name": "gaunernst/gemma-3-27b-it-qat-compressed-tensors",
        "gpus": gpus,
        "sampling_params": {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "min_p": 0.0},
        "template_dir": "data/templates/agents",
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 65536, 'dtype': 'bfloat16', **extra_opts},
        "post_completion": {"impl": "prompt_opt.models.post_completion.SimulatedThinkingCompletionPostProcessor"}
    }


def get_qwq32b(gpus=[0], reasoning=False):
    extra_opts = {"enable-reasoning": None, "reasoning-parser": "deepseek_r1"} if reasoning else {}
    return {
        "short": "qwq32b",
        "name": "Qwen/QwQ-32B-AWQ",
        "gpus": gpus,
        "template_dir": "data/templates/agents",
        # "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 65536, **extra_opts},
        "vllm_opts": {"gpu-memory-utilization": 0.95, "max-model-len": 40960, **extra_opts},
    }


def get_qwen3_14b(gpus=[0], reasoning=False):
    extra_opts = {"enable-reasoning": None, "reasoning-parser": "deepseek_r1"} if reasoning else {}
    vllm_opts = {
        "gpu-memory-utilization": 0.95,
        "max-model-len": 40960 if len(gpus) == 1 else 131072,
        "enable-prefix-caching": None,
        **extra_opts,
    }
    if len(gpus) > 1:
        vllm_opts["rope-scaling"] = '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
    return {
        "short": "qwen3_14b",
        "name": "Qwen/Qwen3-14B",
        "gpus": gpus,
        "template_dir": "data/templates/agents",
        "vllm_opts": vllm_opts,
        }


def get_qwen3_32b(gpus=[0], reasoning=False):
    extra_opts = {"enable-reasoning": None, "reasoning-parser": "deepseek_r1"} if reasoning else {}
    vllm_opts = {
        "gpu-memory-utilization": 0.95,
        "max-model-len": 40960 if len(gpus) == 1 else 131072,
        "enable-prefix-caching": None,
        **extra_opts,
    }
    if len(gpus) > 1:
        vllm_opts["rope-scaling"] = '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
    return {
        "short": "qwen3_32b",
        "name": "CobraMamba/Qwen3-32B-AWQ",
        "gpus": gpus,
        "template_dir": "data/templates/agents",
        "vllm_opts": vllm_opts,
    }


def get_openai_gpt_4o_mini():
    return {
        "type": "openai",
        "short": "gpt4om",
        "name": "gpt-4o-mini-2024-07-18",
        "template_dir": "data/templates/agents",
    }


def get_openai_gpt_o3_mini():
    return {
        "type": "openai",
        "short": "o3m",
        "name": "o3-mini-2025-01-31",
        "template_dir": "data/templates/agents",
        "ignore_temperature": True
    }


def get_openai_gpt_4o():
    return {"type": "openai", "short": "gpt4o", "name": "gpt-4o-2024-11-20", "template_dir": "data/templates/agents"}


def get_debug_model():
    return {"type": "debug", "short": "debug_model", "name": "debug_model", "template_dir": "data/templates/agents"}
