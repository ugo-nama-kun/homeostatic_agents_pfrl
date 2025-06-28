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
from matplotlib.patches import FancyArrowPatch
import seaborn as sns

import gym
import pfrl
import trp_env
from trp_env.wrappers import TRPVisionEnvWrapper, to_multi_modal

import torch
from tqdm import tqdm

from torch import nn
from gym.wrappers import RescaleAction

# from util.env import VisionEnvWrapper, to_multi_modal
from util.ppo import PPO_KL
from util.modules import ortho_init, BetaPolicyModel
from conditioned_trp_env.envs.conditioned_trp_env import FoodClass

sns.set()
sns.set_context("talk")

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

DO_SAMPLING = False
TIME_STEPS = 10_000
N_SEED = 4

# new paramas
param_dirs = [
    "data/success_vision/vision-homeostatic_shaped2021-11-26-14-13-12/20000000_finish/",
    "data/success_vision/vision-homeostatic_shaped2021-11-26-14-16-12/20000000_finish/",  # best agent
    "data/success_vision/vision-homeostatic_shaped2021-12-02-10-54-22/20000000_finish/",
]

seeds = np.arange(N_SEED)


if DO_SAMPLING:
    n_channel = 4
    n_frame_stack = 3
    dim_vision_flatten = 200
    im_size = (64, 64)
    
    env = gym.make(
        id="SmallLowGearAntTRP-v0",
        max_episode_steps=np.inf,
        internal_reset="setpoint",
        n_bins=2,
        sensor_range=0.1,
    )
    env = RescaleAction(env, 0, 1)
    env = TRPVisionEnvWrapper(env, im_size=im_size, n_frame=n_frame_stack, mode="rgbd_array")
    env = pfrl.wrappers.CastObservationToFloat32(env)
    
    obs_size_prop = 27
    obs_size_intero = 2
    action_size = 8
    
    
    # vision encoder
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            
            self.flatten_size = 26912  # 35 * 35 * 32
            
            # random shift augmentation
            # self.aug = RandomShiftsAug(pad=2)
            
            self.vision_encoder = nn.Sequential(
                nn.Conv2d(in_channels=n_channel * n_frame_stack, out_channels=32, kernel_size=3, stride=2, bias=False),
                nn.ELU(inplace=True),
                nn.Conv2d(32, 32, 3, stride=1, bias=False),
                nn.ELU(inplace=True),
                nn.Flatten(),
                nn.Linear(in_features=self.flatten_size, out_features=dim_vision_flatten),
                nn.LayerNorm(normalized_shape=dim_vision_flatten),
                nn.Tanh()
            )
            
            # Initialization
            for l in self.vision_encoder:
                if isinstance(l, nn.Conv2d):
                    gain = nn.init.calculate_gain('relu')
                    nn.init.orthogonal_(l.weight, gain=gain)
                    if hasattr(l.bias, "data"):
                        l.bias.data.fill_(0.0)
                if isinstance(l, nn.Linear):
                    nn.init.orthogonal_(l.weight, gain=1)
                    nn.init.zeros_(l.bias)
        
        def forward(self, x, enable_aug=False):
            x_prop, x_intero, x_vision = to_multi_modal(x,
                                                        im_size=im_size,
                                                        n_frame=n_frame_stack,
                                                        n_channel=n_channel)
                        
            feature_vision = self.vision_encoder(x_vision)
            
            return feature_vision, x_prop, x_intero
    
    
    dim_joint_feature = dim_vision_flatten + 27 + 2
    
    # compose a total model
    class FFModel(nn.Module):
        def __init__(self):
            super(FFModel, self).__init__()
            
            self.encoder = Encoder()
            
            self.policy_network = BetaPolicyModel(obs_size=dim_joint_feature,
                                                  action_size=action_size,
                                                  activation=nn.Tanh,
                                                  hidden1=300,
                                                  hidden2=200)
            
            self.value_network = nn.Sequential(
                nn.Linear(dim_joint_feature, 400),
                nn.ReLU(inplace=True),
                nn.Linear(400, 300),
                nn.ReLU(inplace=True),
                nn.Linear(300, 1)
            )
            
            ortho_init(self.value_network[0], gain=1)
            ortho_init(self.value_network[2], gain=1)
            ortho_init(self.value_network[4], gain=1)
        
        def forward(self, x, enable_aug=False):
            # enable data augmentation while training
            feature_vision, x_prop, x_intero = self.encoder(x, enable_aug)
            
            # policy path (removing detach op for vision)
            x_policy = torch.cat((feature_vision, x_prop, x_intero), dim=1)
            distribs = self.policy_network(x_policy)
            
            # value path
            x_value = torch.cat((feature_vision, x_prop, x_intero), dim=1)
            vs_pred = self.value_network(x_value)
            
            return (distribs, vs_pred)
    
    model = FFModel()
    
    opt = torch.optim.Adam(model.parameters())
    
    if torch.cuda.is_available():
        gpu_id = 1
        print(torch.cuda.get_device_name(gpu_id))
    else:
        gpu_id = None
    
    agent = PPO_KL(
        model=model,
        optimizer=opt,
        gpu=gpu_id,
    )
    
    # data hist
    position_hist = np.zeros((len(param_dirs), N_SEED, TIME_STEPS, 2))  # xy positions
    orientation_hist = np.zeros((len(param_dirs), N_SEED, TIME_STEPS, 1))
    joint_hist = np.zeros((len(param_dirs), N_SEED, TIME_STEPS, 8))
    joint_vel_hist = np.zeros((len(param_dirs), N_SEED, TIME_STEPS, 8))
    intero_hist = np.zeros((len(param_dirs), N_SEED, TIME_STEPS, 2))
    food_positions = [[set() for k in range(N_SEED)] for _ in range(len(param_dirs))]
    
    def add_object(id_, seed, objects_):
        for o_ in objects_:
            if len(o_) != 0:
                food_positions[id_][seed].add(o_)
    
    for id_, d in enumerate(param_dirs):
        
        agent.load(d)
        
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
        
            print(f"START NEW PARAM: {d}")
            
            env.seed(int(seed))
    
            obs = env.reset()
            add_object(id_, seed, env.objects)
            
            print("initial intero: ", env.get_interoception())
    
            for t in tqdm(range(TIME_STEPS), desc=f"{id_ + 1}/{len(param_dirs)} sampling:"):
                
                position_hist[id_, seed, t] = env.wrapped_env.get_body_com("torso")[:2]
                orientation_hist[id_, seed, t] = env.get_ori()
                current_joint_obs = np.array(env.wrapped_env.get_current_obs())
                # print(current_joint_obs)
                joint_hist[id_, seed, t] = current_joint_obs[5:13]
                joint_vel_hist[id_, seed, t] = current_joint_obs[19:27]
                intero_hist[id_, seed, t] = env.get_interoception()
                
                # print(env.get_interoception())
                
                with torch.no_grad():
                    action = agent.act(obs)
                    
                obs, reward, done, info = env.step(action)
                add_object(id_, seed, env.objects)
                # env.render()
                
                if done:
                    break
    
    env.close()

    import os
    os.makedirs("individual_data_vision", exist_ok=True)
    np.save("individual_data_vision/position_hist", position_hist)
    np.save("individual_data_vision/orientation_hist", orientation_hist)
    np.save("individual_data_vision/joint_hist", joint_hist)
    np.save("individual_data_vision/joint_vel_hist", joint_vel_hist)
    np.save("individual_data_vision/intero_hist", intero_hist)

    food_positions = [[list(s) for s in fp] for fp in food_positions]
    for i in range(len(param_dirs)):
        for j in range(N_SEED):
            food_positions[i][j] = [(obj[0], obj[1], obj[2].value) for obj in food_positions[i][j]]

    with open('individual_data_vision/food_positions.json', 'w') as file:
        json.dump(food_positions, file)


# PLOTTING
position_hist = np.load("individual_data_vision/position_hist.npy").astype(float)
orientation_hist = np.load("individual_data_vision/orientation_hist.npy").astype(float)
joint_hist = np.load("individual_data_vision/joint_hist.npy").astype(float)
joint_vel_hist = np.load("individual_data_vision/joint_vel_hist.npy").astype(float)
intero_hist = np.load("individual_data_vision/intero_hist.npy").astype(float)

with open('individual_data_vision/food_positions.json', 'r') as file:
    food_positions = json.load(file)

cmap = cm.get_cmap('viridis')
num_colors = len(param_dirs)
t = np.linspace(0, 1, TIME_STEPS)

fig1 = plt.figure(figsize=(7, 6), dpi=100)
for i in range(len(param_dirs)):
    for j in range(4):
        plt.subplot(4, len(param_dirs), i+len(param_dirs)*j+1)
        if j == 0:
            plt.title(f"ID={i}v")
    
        food_positions_ij = list(food_positions[i][j])
        pos_red = []
        pos_blue = []
        for obj in food_positions_ij:
            if obj[2] == 1:
                pos_blue.append(obj[:2])
            else:
                pos_red.append(obj[:2])
    
        pos_blue, pos_red = np.array(pos_blue), np.array(pos_red)
        plt.scatter(pos_red[:, 0], pos_red[:, 1], c="r", alpha=1, s=20, edgecolors="none")
        plt.scatter(pos_blue[:, 0], pos_blue[:, 1], c="b", alpha=1, s=20, edgecolors="none")
    
        plt.scatter(position_hist[i, j, :, 0],
                    position_hist[i, j, :, 1],
                    c=t, cmap='viridis', s=1, edgecolors="none")
            
        plt.ylim([-7, 7])
        plt.xlim([-7, 7])
        plt.axis("square")
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])

plt.tight_layout()
plt.savefig("individual_same_seed_vision.pdf")
# plt.show()

print("finish.")
