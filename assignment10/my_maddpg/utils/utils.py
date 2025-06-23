import os
import time
import random
import torch
import numpy as np
from typing import Dict


EPS = 1e-8


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def combine_actions(group_of_policy_out: list, n_envs: int):
    actions_groups = [group['actions'] for group in group_of_policy_out]
    actions_combined = [{} for _ in range(n_envs)]
    for i_env in range(n_envs):
        for actions in actions_groups:
            actions_combined[i_env].update(actions[i_env])
    return actions_combined


def space2shape(observation_space):
    if isinstance(observation_space, Dict) or isinstance(observation_space, dict):
        return {key: observation_space[key].shape for key in observation_space.keys()}
    elif isinstance(observation_space, tuple):
        return observation_space
    else:
        return observation_space.shape



def create_directory(path):
    dir_split = path.split("/")
    current_dir = dir_split[0] + "/"
    for i in range(1, len(dir_split)):
        if not os.path.exists(current_dir):
            os.mkdir(current_dir)
        current_dir = current_dir + dir_split[i] + "/"


def get_time_string():
    t_now = time.localtime(time.time())
    t_year = str(t_now.tm_year).zfill(4)
    t_month = str(t_now.tm_mon).zfill(2)
    t_day = str(t_now.tm_mday).zfill(2)
    t_hour = str(t_now.tm_hour).zfill(2)
    t_min = str(t_now.tm_min).zfill(2)
    t_sec = str(t_now.tm_sec).zfill(2)
    time_string = f"{t_year}_{t_month}{t_day}_{t_hour}{t_min}{t_sec}"
    return time_string


def create_memory(shape,
                  n_envs,
                  n_size,
                  dtype=np.float32):
    if shape is None:
        return None
    elif isinstance(shape, dict):
        memory = {}
        for key, value in shape.items():
            if value is None:  # save an object type
                memory[key] = np.zeros([n_envs, n_size], dtype=object)
            else:
                memory[key] = np.zeros(
                    [n_envs, n_size] + list(value), dtype=dtype)
        return memory
    elif isinstance(shape, tuple):
        return np.zeros([n_envs, n_size] + list(shape), dtype)
    else:
        raise NotImplementedError

