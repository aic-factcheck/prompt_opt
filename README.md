# PromptOpt
Prompt Optimization library

## Installation
These instructions log installation to RCI cluster running on SLURM batch system, but should work on other configurations. 

### Run interactive session
```bash
srun -p amdgpufast --time=4:00:00 --gres=gpu:1 --mem=32G --pty bash -i
srun -p amdfast --time=4:00:00 --mem=32G --pty bash -i
```

### Create Python irtual environment
Load Python module, VLLM currently needs <3.13.
```bash
ml Python/3.12.3-GCCcore-13.3.0 
# in your virtual environments' dir:
python -m venv promptopt
```

### Clone this repository
```bash
git clone git@github.com:aic-factcheck/prompt_opt.git
cd prompt_opt
```

### Create environment init script
```bash
cat << 'EOF' > init_environment_promptopt_amd.sh
echo "initializing environment..."
uname -a

ml Python/3.12.3-GCCcore-13.3.0
ml CUDA/12.6.0
ml cuDNN/9.5.0.50-CUDA-12.6.0

source /home/drchajan/devel/python/FC/VENV/promptopt/bin/activate

echo "done"
EOF
```
Note: Do not forget to change the virtual environment path.

Create symlink for convenience of changing different module and virtual environment configurations. 
```bash
ln -s init_environment_promptopt_amd.sh init_environment_default.sh
```
Activate the environment.
```bash
source init_environment_default.sh
```

### Install modules
Install packages. The most important part is the VLLM installation.
I am using bleeding-edge version so newest models are supported.
My current tested version is `vllm==0.9.1.dev35+g1661a9c28`.
I'll create `requirements.txt` later.

```bash
pip install --upgrade pip
pip install wheel setuptools
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
pip install git+https://github.com/aic-factcheck/aic-nlp-utils.git
pip install loguru pandas rapidfuzz scikit-learn wandb
```

Do not forget to [setup Weights & Biases](https://docs.wandb.ai/quickstart/) for logging. 

### Jupyter install
```bash
pip install jupyterlab
python -m ipykernel install --user --name=promptopt
```

## Big-Bench-Hard (BBH) for Prompt Optimization Test Run
BBH datasets are stored in `data/BBH_PO`. There are multiple versions:
- `task_orig.json`: original BBH data converted to common format used by this module,
- `task.json`: harder(?) version created by stripping task descriptions and converting answer format from class id choice to string (in some cases),
- `task_obs.json`: hardest version(?): same as the previous but the answer class names are obfuscated to give least information on the task.
See `notebooks/data_bbh_promptopt_create.ipynb` for more details.

Run from `slurm/` directory:
```bash
sbatch run_opt_vllm_amdgpu_2gpu.batch cfg/BBH/cfg_BBH_V1_EA.py
```

## "55 cases" Medical Data Test Run
The SLURM example scripts can be found in `slurm/` directory.

This runs a simplified version of the task with 3 classes only (521 and 521A merged) using DeepSeek fine-tunned LLama model (needs 2 A100 GPUs).
```bash
sbatch run_opt_vllm_amdgpu_2gpu.batch cfg/MamaAI/cfg_55cases_V1_EA_simple.py
```

The `cfg/MamaAI/cfg_55cases_V1_EA_simple.py` argument points to a configuration file, which has a form of a Python script (evaluated by `aic_nlp_utils.pycfg`).

The following runs the 4 class full task using Qwen3 32B on a single A100 GPU.
```bash
sbatch run_opt_vllm_amdgpu_1gpu.batch cfg/MamaAI/cfg_55cases_V1_EA.py
```

The experiment output directory is defined by `get_exp_dir(cfg)` in the configuration files. It is now set to `EXP/[experiment_name]/seed_XXXXX`.

## Optimization Archive
The prompt candidates explored during the optimization run are stored in `archive.jsonl`, where each line corresponds to a single **candidate**.
The format of each candidate is as follows (long strings cropped):
```json5
{
  "messages": [ // chat messages leading to this candidate's prompt
    {
      "role": "user",
      "content": "# Instructions\nI wil...",
      "duration": 0.0,
      "desc": ""
    },
    {
      "role": "assistant",
      "content": "<think>\n\nOkay, I nee...", // this contains the prompt
      "duration": 43.97779560089111,
      "desc": ""
    }
  ],
  "split2indices": { // sample indices refering to original splits
  // here the original "trn" split is randomly resampled to new "trn" and "dev" splits
  // to promote diversity, it works similarly to cross-validation
    "trn": {"source": "trn", "indices": [13, 0, ...]}, 
    "dev": {"source": "trn", "indices": [10, 6, ...]}
  },
  "duration": 43.981162786483765, // total time to create this candidate
  "id": 21, // candidate identifier
  "parent_id": 1, // candidate's parent identifier
  "split": { // candidate evaluation for each split
    "trn": [ // the list of all "trn" split samples
      {
        "think": "\nOkay, I need to det...", // reasoning model's thinking extracted
        "pred": {"label": "other"}, // JSON prediction
        "gold": {"label": "520: rány nešité, ..."}, // GOLD label JSON
        "query": "<document id=\"1\">\nZd...", // the query part of the prompt
        "messages": [ // evaluation chat messages
          {
            "role": "user",
            "content": "To transform a query...",
            "duration": 0.0,
            "desc": ""
          },
          {
            "role": "assistant",
            "content": "<think>\n\nOkay, I nee...", // this contains the prediction extracted above
            "duration": 14.473679304122925,
            "desc": "AgentJSONForReasonin..."
          }
        ],
        "eval": { // scores for this sample 
          "oa": { // here it is only the ObjectAligner score
            "reasoning": "The predicted output...", // textual explanation of the score
            "score": 0.0 // score value (0-1 range)
          }
        }
      }, ...
    ],
    "dev": [...],
    "tst": [...]
  }
}
```

See methods like `utils.candidate2prompt_dseek` to extract the candidate's prompt.
