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
pip install loguru pandas rapidfuzz wandb
```

Do not forget to [setup Weights & Biases](https://docs.wandb.ai/quickstart/) for logging. 

### Jupyter install
```bash
pip install jupyterlab
python -m ipykernel install --user --name=promptopt
```
## "55 cases" medical data test run
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