import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import seaborn as sns

import gym
import pfrl

import torch
from tqdm import tqdm

from torch import nn
from gym.wrappers import RescaleAction

from util.ppo import PPO_KL
from util.modules import ortho_init, BetaPolicyModel
from util.env import VisionEnvWrapper, to_multi_modal

sns.set()
sns.set_context("talk")

seed = 100
n_channel = 4
n_frame_stack = 3
dim_vision_flatten = 200
im_size = (64, 64)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)


########################################
# Environment
########################################

def get_env(n_blue_red=(6, 4), blue_nutrient=(0.1, 0), red_nutrient=(0, 0.1)):
    env = gym.make(
        id="trp_env:SmallLowGearAntTRP-v0",
        max_episode_steps=np.inf,
        internal_reset="setpoint",
        n_bins=2,
        sensor_range=0.1,
        n_blue=n_blue_red[0],
        n_red=n_blue_red[1],
        blue_nutrient=blue_nutrient,
        red_nutrient=red_nutrient,
    )
    env = RescaleAction(env, 0, 1)
    env = VisionEnvWrapper(env, im_size=im_size, n_frame=n_frame_stack, mode="rgbd_array")
    env = pfrl.wrappers.CastObservationToFloat32(env)
    return env


########################################
# Agent
########################################
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
        
        #         if enable_aug:
        #             x_vision = self.aug(x_vision.float())
        
        feature_vision = self.vision_encoder(x_vision)
        # print("feature", feature_vision.shape)
        
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
    
    def feature(self, x, layer=2):
        feature_vision, x_prop, x_intero = self.encoder(x)
        x_policy = torch.cat((feature_vision, x_prop, x_intero), dim=1)
        if layer == 1:
            latent = self.policy_network.fc[0](x_policy)
            latent = self.policy_network.fc[1](latent)
        elif layer == 2:
            latent = self.policy_network.fc(x_policy)
        else:
            raise ValueError(f"layer value should be 1 or 2. given {layer}")
        return latent.detach()
    
    def feature_vision(self, x):
        feature_vision, x_prop, x_intero = self.encoder(x)
        return feature_vision.detach()


model = FFModel()

opt = torch.optim.Adam(model.parameters())

if torch.cuda.is_available():
    gpu_id = 1
    print(torch.cuda.get_device_name(gpu_id))
else:
    gpu_id = None

if gpu_id is not None:
    device = torch.device(f'cuda:{gpu_id}')
else:
    device = "cpu"

agent = PPO_KL(
    model=model,
    optimizer=opt,
    gpu=gpu_id,
)

agent.load("data/success_vision/vision-homeostatic_shaped2021-11-26-14-16-12/20000000_finish")


def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0], interpolation="nearest")
    
    def update(frame):
        im.set_data(frame)
        return [im]
    
    interval = 1000 / framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    return anim


MAX_STEPS = 20000

env = get_env()

obs = env.reset()
frames = []

interoception = np.zeros((MAX_STEPS, 2))

for i in tqdm(range(MAX_STEPS)):
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
    
    interoception[i] = env.get_interoception()
    
    _, _, vision = to_multi_modal(torch.tensor([obs]), im_size=im_size, n_frame=n_frame_stack, n_channel=n_channel)
    vision = env.decode_vision(vision)
    vision = (255. * vision.permute(0, 2, 3, 1)[0].numpy()).astype(np.uint8)[:, :, :3]
    frames.append(vision)
    # vision = env.wrapped_env._get_viewer("rgb_array").read_pixels(64, 64, False)[::-1, :, :]
    frames.append(vision.copy())
    plt.imshow(vision); plt.pause(0.0001)
    if done:
        break

plt.imshow(frames[10])

FPS = 60
anim = display_video(frames, FPS)
writervideo = animation.FFMpegWriter(fps=FPS)
anim.save('ecocentroc_vision.mp4', writer=writervideo)

plt.plot(interoception)
