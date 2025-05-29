import os
from pathlib import Path
from pprint import pp
import subprocess
import time
from typing import Dict, List, Union


from loguru import logger
import numpy as np


from ..slurm_utils import get_job_id, get_idle_gpus, get_allocated_nodes_and_gpus
from .llm_predictor import LLMPredictor, OpenAIPredictor, VLLMPredictor, DebugPredictor
from ..agents.agent_chat import AgentChat
from ..utils import get_class_instance_by_config

DEFAULT_SAMPLING_PARAMS = {"temperature": 0.3}

class PredictorLoader:
    def __init__(self, cfg_models, exp_path, port_min=8000, port_max=8999):
        cfg_openai_models = {}
        cfg_local_models = {}
        cfg_debug_models = {}
        for model_id, cfg_model in cfg_models.items():
            type_ = cfg_model.get("type", "local")
            assert type_ in ["local", "openai", "debug"]
            if type_ == "local":
                cfg_local_models[model_id] = cfg_model
            elif type_ == "openai":
                cfg_openai_models[model_id] = cfg_model
            else:
                cfg_debug_models[model_id] = cfg_model
                
        self.openai_predictor_loader = OpenAIPredictorLoader(cfg_openai_models, exp_path) if len(cfg_openai_models) > 0 else DummyPredictorLoader()
        self.local_predictor_loader = LocalPredictorLoader(cfg_local_models, exp_path, port_min=8000, port_max=8999) if len(cfg_local_models) > 0 else DummyPredictorLoader()
        self.debug_predictor_loader = DebugPredictorLoader(cfg_debug_models) if len(cfg_debug_models) > 0 else DummyPredictorLoader()
    
    
    def killall(self):
        self.openai_predictor_loader.killall()
        self.local_predictor_loader.killall()
        self.debug_predictor_loader.killall()
    
    
    def load(self):
        predictors = self.openai_predictor_loader.load()
        predictors.update(self.local_predictor_loader.load())
        predictors.update(self.debug_predictor_loader.load())
        return predictors
    
    
class DummyPredictorLoader:
    def __init__(self):
        pass
    
    
    def killall(self):
        pass
    
    
    def load(self):
        return {}
    
    
class DebugPredictorLoader:
    def __init__(self, cfg_models):
        self.cfg_models = cfg_models
    
    
    def killall(self):
        pass
    
    
    def load(self):
        logger.info("loading...")
        predictors = {}
        for model_id, cfg_model in self.cfg_models.items():
            model_name = cfg_model["name"]
            llm_predictor = DebugPredictor(
                model_name=model_name,
                sampling_params=cfg_model.get("sampling_params", DEFAULT_SAMPLING_PARAMS),
                template_dir=self.cfg_models[model_id]["template_dir"]
            )
            predictors[model_id] = llm_predictor
        return predictors
        
    
class OpenAIPredictorLoader:
    def __init__(self, cfg_models, exp_path):
        self.cfg_models = cfg_models
        self.exp_path = exp_path
        logger.info("init()")
    
    
    def killall(self):
        pass
    
    
    def load(self, debug=False):
        logger.info("loading...")
        predictors = {}
        for model_id, cfg_model in self.cfg_models.items():
            model_name = cfg_model["name"]
            llm_predictor = OpenAIPredictor(
                model_name=model_name,
                template_dir=self.cfg_models[model_id]["template_dir"],
                base_url=cfg_model.get("base_url"),
                sampling_params=cfg_model.get("sampling_params", DEFAULT_SAMPLING_PARAMS),
                ignore_temperature=cfg_model.get("ignore_temperature", False)
            )
            predictors[model_id] = llm_predictor
            
            if debug: 
                system_content = llm_predictor.get_template('chat/system_v1.txt.jinja').render()
                agent = AgentChat(llm_predictor, system_content)
                test_schema = {"type": "object",
                    "properties": {
                        "name": {
                        "type": "string"
                        }
                    },
                    "additionalProperties": False,
                    "required": ["name"]
                }
                test_answer = agent.query("Capital of GB?", temperature=0.0, guided_json=test_schema)
                logger.info(f'loaded {model_name}, test answer: "{test_answer}"')
            else:
                logger.info(f'loaded {model_name},')
        logger.info("all models loaded", flush=True)
        logger.info(predictors)
        return predictors
        

class LocalPredictorLoader:
    def __init__(self, cfg_models, exp_path, port_min=8000, port_max=8999):
        self.cfg_models = cfg_models
        self.exp_path = exp_path
        self.slurm_job_id = get_job_id()
        self.gpus = get_idle_gpus() # cuda devices
        self.model_ids = list(cfg_models.keys())
        self.ports = np.random.choice(np.arange(port_min, port_max+1), size=len(self.model_ids), replace=False)
        
        # requested_gpus a list [0, 1, ...] for each model
        requested_gpu_idxs = [cfg_models[model_id]["gpus"] for model_id in self.model_ids]
        # flatten to a set
        all_requested_gpu_idxs = sorted(set([gidx for glist in requested_gpu_idxs for gidx in glist]))
        for i, g in enumerate(all_requested_gpu_idxs):
            assert i == g, f"incorrectly numbered gpu indices: {all_requested_gpu_idxs}"

        # remap to actual cuda devices
        self.requested_gpus = []
        for model_requested_gpus in requested_gpu_idxs:
            self.requested_gpus.append([self.gpus[gidx] for gidx in model_requested_gpus])

        logger.info("init()")
        logger.info(f" allocated GPUs: {self.gpus}")
        logger.info(f" requested GPUs: {self.requested_gpus}")
        
        assert len(self.gpus) == len(all_requested_gpu_idxs)
        
        self.processes = []


    def killall(self):
        for process in self.processes:
            logger.info(f"killing PID: {process.pid}")
            process.kill()
            

    def load(self, debug=True):
        logger.info("loading...")
        log_files = []
        for model_id, port, req_gpus in zip(self.model_ids, self.ports, self.requested_gpus):
            logger.info(f'running "{model_id}" model VLLM')
            log_file = f"{self.exp_path}/{get_job_id()}.{model_id}.vllm_server.out"
            log_files.append(log_file)

            model_name = self.cfg_models[model_id]["name"]
            vllm_opts = self.cfg_models[model_id]["vllm_opts"]
            vllm_opts = [["--" + k, str(v)] if v else ["--" + k] for k, v in vllm_opts.items()]
            vllm_opts = [e for kv in vllm_opts for e in kv]
            
            cuda_visible_devices = ','.join([g for g in req_gpus])
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            logger.info(f"CUDA_VISIBLE_DEVICES = {cuda_visible_devices}")
            
            command_line = ["nohup", "vllm", "serve", model_name, "--port", str(port), "--tensor-parallel-size", str(len(req_gpus))] + vllm_opts
            logger.debug(f"VLLM command line: {command_line}")
            # logger.debug(f"VLLM command line: {command_line}", flush=True) # fails for string with curly braces
            
            with open(log_file, "a") as log_file:
                process = subprocess.Popen(
                    command_line, 
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    # preexec_fn=os.setpgrp
                    )
                self.processes.append(process)
            time.sleep(3)
        
        logger.info("waiting for VLLMs to start...", flush=True)
        while len(log_files) > 0:
            for log_file in log_files.copy():
                 with open(log_file, 'r') as file:
                    contents = file.read()
        
                    # if "Avg prompt throughput:" in contents: # used for older version
                    if "Application startup complete." in contents: # this should work for older as well
                        logger.info(f"VLLM initialized for: {log_file}")
                        log_files.remove(log_file)
            time.sleep(3)
            print(".", end="", flush=True)
            
        predictors = {}
        for model_id, port in zip(self.model_ids, self.ports):
            model_name = self.cfg_models[model_id]["name"]
            cfg_model = self.cfg_models[model_id]
            post_completion_cfg = self.cfg_models[model_id].get("post_completion", {"impl": "prompt_opt.models.post_completion.BasicCompletionPostProcessor"})
            post_completion = get_class_instance_by_config(post_completion_cfg)
            completion_log_file = f"{self.exp_path}/{get_job_id()}.{model_id}.vllm_completions.jsonl"
            
            llm_predictor = VLLMPredictor(
                model_name=model_name,
                post_completion=post_completion,
                openai_base_url=f"http://localhost:{port}/v1",
                sampling_params=cfg_model.get("sampling_params", DEFAULT_SAMPLING_PARAMS),
                guided_decoding_backend=cfg_model.get("guided_decoding_backend"), 
                template_dir=cfg_model["template_dir"],
                log_jsonl=completion_log_file
            )
            predictors[model_id] = llm_predictor
            
            if debug: 
                system_content = llm_predictor.get_template('chat/system_v1.txt.jinja').render()
                agent = AgentChat(llm_predictor, system_content)
                test_schema = {"type": "object",
                    "properties": {
                        "name": {
                        "type": "string"
                        }
                    },
                    "required": ["name"]
                }
                # test_answer = agent.query("Capital of GB?", frequency_penalty=0.05, guided_json=test_schema)
                test_answer = agent.query("Capital of GB?", temperature=0.1, frequency_penalty=0.05, guided_json=test_schema)

                logger.info(f'loaded {model_name}, test answer: "{test_answer}"')
        logger.info("all models loaded", flush=True)

        return predictors