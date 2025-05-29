import os
import re
import subprocess

from prompt_opt.utils import lw

def rename_slurm_job_name(job_id, new_job_name):
    cmd = f"scontrol update JobId={job_id} JobName={new_job_name}"
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"Job name changed to '{new_job_name}' for JobId {job_id}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to change job name: {e}")


def get_job_id():
    return os.getenv('SLURM_JOB_ID')


def get_idle_gpus():
    gpu_info_result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,pci.bus_id', '--format=csv,noheader'],
        stdout=subprocess.PIPE, universal_newlines=True
    )

    process_info_result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=gpu_bus_id', '--format=csv,noheader'],
        stdout=subprocess.PIPE, universal_newlines=True
    )

    try:
        bus2gpu = {line.strip().split(",")[1].strip(): line.strip().split(",")[0].strip()  for line in gpu_info_result.stdout.splitlines()}
        busy_gpus = set(line.strip() for line in process_info_result.stdout.splitlines())

        idle_gpus = [gpu for bus, gpu in bus2gpu.items() if bus not in busy_gpus]
    except:
        lw("No GPUs found!")
        idle_gpus = []

    return idle_gpus


def has_gpus():
    return len(get_idle_gpus()) > 0


def get_allocated_nodes_and_gpus(job_id):
    assert False, "IMPLEMENT SUPPORT for multiple nodes!"
    cmd = f"scontrol show job {job_id}"
    output = subprocess.check_output(cmd, shell=True).decode('utf-8')
    node_list_pattern = re.compile(r' NodeList=(\S+)')
    node_list_match = node_list_pattern.search(output)
    
    if node_list_match:
        node_list = node_list_match.group(1)
        expanded_node_list = subprocess.check_output(f"scontrol show hostnames {node_list}", shell=True).decode('utf-8').splitlines()
    else:
        raise ValueError("NodeList not found in the job information.")

    node_gpu_dict = {}

    for node in expanded_node_list:
        node_info_cmd = f"scontrol show node={node}"
        node_info_output = subprocess.check_output(node_info_cmd, shell=True).decode('utf-8')
        print(node_info_output)
        
        gpu_pattern = re.compile(r' Gres=gpu:(\d+)')
        gpu_match = gpu_pattern.search(node_info_output)
        
        if gpu_match:
            num_gpus = int(gpu_match.group(1))
            node_gpu_dict[node] = num_gpus
        else:
            node_gpu_dict[node] = 0

    return node_gpu_dict