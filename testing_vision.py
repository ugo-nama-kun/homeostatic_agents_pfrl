"""
Testing/prototyping code for visual environment
"""
import math
import time

import gym
import numpy as np
import pfrl

import matplotlib.pyplot as plt
import torch

import imageio

from util.env import VisionEnvWrapper, to_multi_modal

env_id = "trp_env:SmallLowGearAntTRP-v0"
reward_name = "homeostatic"
reward_bias = None
reward_scale = 1.0

env_config = {"max_episode_steps": np.inf,
              "internal_reset": "random",
              "reward_setting": reward_name,
              "reward_bias": reward_bias,
              "coef_main_rew": reward_scale,
              "coef_ctrl_cost": 0.001,
              "coef_head_angle": 0.005,
              "internal_random_range": (-1. / 6, 1. / 6)}

env = gym.make(
    env_id,
    max_episode_steps=env_config["max_episode_steps"],
    internal_reset=env_config["internal_reset"],
    reward_setting=env_config["reward_setting"],
    reward_bias=env_config["reward_bias"],
    coef_main_rew=env_config["coef_main_rew"],
    coef_ctrl_cost=env_config["coef_ctrl_cost"],
    coef_head_angle=env_config["coef_head_angle"],
    internal_random_range=env_config["internal_random_range"],
    n_bins=20,
    sensor_range=16,
)

n_frame = 4
n_channel = 4
use_step_by_step = False
use_dummy = False

if use_dummy:
    im_size = (64, 64)  # dummy image is 64x64x4 (rgba)
else:
    im_size = (84, 84)

env = VisionEnvWrapper(env, im_size=im_size, n_frame=n_frame, mode="rgb_array" if n_channel == 3 else "rgbd_array")
env = pfrl.wrappers.CastObservationToFloat32(env)

obs = env.reset()
for i in range(100):
    obs, _, _, _ = env.step(env.action_space.sample())
env.close()
time.sleep(0.1)

print(obs.shape)

im_original = obs[29:]
ims = [im_original[np.prod(im_size) * n_channel * i:np.prod(im_size) * n_channel * (i+1)] for i in range(n_frame)]
for i in range(n_frame):
    ims[i] = ims[i].reshape(im_size[0], im_size[1], n_channel)
im_original = np.array(ims)

print("im original shape: ", im_original.shape)

plt.figure()
for i in range(n_frame):
    plt.subplot(n_frame, 1, i + 1)
    plt.ylabel(f"$t = {i}$")
    plt.imshow(env.decode_vision(im_original[i][:, :, :3]))
plt.pause(0.01)

obs_tensor = torch.tensor(obs).unsqueeze(0)

print(obs_tensor.shape)

if not use_step_by_step:
    input_prop, input_intero, input_vision = to_multi_modal(obs_tensor,
                                                            im_size=im_size,
                                                            n_frame=n_frame,
                                                            n_channel=n_channel)
else:
    # step-by-step definition
    input_prop, input_intero, input_vision = torch.split(obs_tensor, split_size_or_sections=(27, 2, np.prod(im_size) * 3 * n_frame), dim=1)

    # dummy input
    if use_dummy:
        input_vision_dummy = 2 * (np.array(imageio.imread("data/color_wheel.png")[:, :, :3])/255. - 0.5)
        print("dummy vision", input_vision_dummy.shape)

        plt.figure()
        for i in range(n_frame):
            plt.subplot(n_frame, 1, i + 1)
            plt.title(f"$t = {i}$")
            plt.imshow(env.decode_vision(input_vision_dummy))
        plt.pause(0.01)

        input_vision = torch.tensor(np.concatenate([input_vision_dummy.flatten()] * n_frame))
        input_vision = torch.stack([input_vision, input_vision])

    print("input_vision raw", input_vision.shape, [np.prod(im_size) * n_channel] * n_frame)

    input_vision = torch.split(input_vision, split_size_or_sections=[np.prod(im_size) * n_channel] * n_frame, dim=1)
    print("input_vision split", [im.shape for im in input_vision])

    input_vision = [im.reshape(-1, im_size[0], im_size[1], n_channel) for im in input_vision]
    print("input_vision reshape", [im.shape for im in input_vision])

    input_vision = torch.cat(input_vision, dim=3)
    print("input_vision stack", input_vision.shape)

    input_vision = input_vision.view(-1, im_size[0], im_size[1], n_channel * n_frame).permute(0, 3, 1, 2)
    print("input_vision cnn data", input_vision.shape)

print("proprioception:", input_prop)
print("interoception: ", input_intero, env.get_interoception())
print("vision_shape:", input_vision.shape)

im = np.hstack([input_vision[0][i].detach().numpy() for i in range(n_channel * n_frame)])

plt.figure(figsize=(5, 5))
for i in range(n_frame):
    for j in range(n_channel):
        plt.subplot(n_frame, n_channel, n_channel*i + j + 1)
        if j == 0:
            plt.ylabel(f"$t = {i}$")
            plt.title("R")
        elif j == 1:
            plt.title("G")
        elif j == 2:
            plt.title("B")
        elif j == 3:
            plt.title("D")
        plt.imshow(env.decode_vision(input_vision[0, n_channel*i + j]))
plt.show()

