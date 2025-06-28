import collections
import itertools
import logging
import random
import sys
from collections import deque
import json

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import gym
import pfrl
import trp_env
import conditioned_trp_env

import torch
from tqdm import tqdm

from torch import nn
from gym.wrappers import RescaleAction

from util.ppo import PPO_KL
from util.modules import ortho_init, BetaPolicyModel
from conditioned_trp_env.envs.conditioned_trp_env import FoodClass

sns.set()
sns.set_context("talk")

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# group1 : 0, 3  --> ankle 1 & 2 forward
# group2 : 1, 2, 4 --> ankle 2 & 3 forward
AGENT_ID = 4
SEED = 0

# new paramas
param_dirs = [
    "data/result_trp_therm_oct2021/trp-homeostatic_shaped2021-09-30-14-54-11/150000000_finish",
    "data/result_trp_therm_oct2021/trp-homeostatic_shaped2021-09-30-14-40-15/150000000_finish",
    "data/result_trp_therm_oct2021/trp-homeostatic_shaped2021-09-28-17-38-42/150000000_finish",
    "data/result_trp_therm_oct2021/trp-homeostatic_shaped2021-09-28-17-26-31/150000000_finish",
    "data/result_trp_therm_oct2021/trp-homeostatic_shaped2021-10-02-12-18-57/150000000_finish",
]

seeds = np.arange(6)

env = gym.make(
    "SmallLowGearAntTRP-v0",
    max_episode_steps=np.inf,
    internal_reset="setpoint",
    n_bins=20,
    sensor_range=16,
)
env = RescaleAction(env, 0, 1)
env = pfrl.wrappers.CastObservationToFloat32(env)

obs_space = env.observation_space
action_space = env.action_space

obs_size = obs_space.low.size
action_size = action_space.low.size

policy = BetaPolicyModel(obs_size=obs_size,
                         action_size=action_size,
                         hidden1=256,
                         hidden2=64)

value_func = torch.nn.Sequential(
    nn.Linear(obs_size, 256),
    nn.Tanh(),
    nn.Linear(256, 64),
    nn.Tanh(),
    nn.Linear(64, 1),
)

model = pfrl.nn.Branched(policy, value_func)

opt = torch.optim.Adam(model.parameters())

agent = PPO_KL(
    model=model,
    optimizer=opt,
    gpu=-1,
)

agent.load(param_dirs[AGENT_ID])
env.seed(SEED)

obs = env.reset()

print("initial intero: ", env.get_interoception())

while True:
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    
    if done:
        break


print("finish.")
