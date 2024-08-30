import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
import pybullet as p

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

EFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('one_d_rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 2
output_folder=DEFAULT_OUTPUT_FOLDER
colab=DEFAULT_COLAB
plot=True

model_path = 'results/save-08.26.2024_16.45.51/final_model.zip'  


test_env = MultiHoverAviary(gui=True, num_drones=2, obs=ObservationType('kin'), act=ActionType('one_d_rpm'))
model = PPO.load(model_path)


obs, info = test_env.reset(seed=42, options={})



logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=2,
                output_folder=output_folder,
                colab=colab
                )

for i in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    obs2 = obs.squeeze()
    act2 = action.squeeze()
    # 
    drone_position = obs[0][:3]  # 

 
    for d in range(DEFAULT_AGENTS):
        logger.log(drone=d,
                   timestamp=i / test_env.CTRL_FREQ,
                   state=np.hstack([obs2[d][0:3],
                                    np.zeros(4),
                                    obs2[d][3:15],
                                    act2[d]
                                    ]),
                   control=np.zeros(12)
                   )

    test_env.render()

    time.sleep(1.0 / test_env.CTRL_FREQ)

    if terminated:
        obs, info = test_env.reset(seed=42, options={})

test_env.close()

if plot and DEFAULT_OBS == ObservationType.KIN:
    logger.plot()