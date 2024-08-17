# diffusion policy import
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import pickle
import random
import sys
sys.path.insert(0, '/home/hongyic/3D_Learning/Kodex/')
import pdb 
import torch
import torch.nn as nn
import torchvision
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from utils.gym_env import GymEnv
from utils.unet1D import *
from utils.resnet18 import ClassificationNetwork18
from utils.fc_network import FCNetwork
from utils.Diffusion_evaluation_gt import Diffusion_policy_control_reorientation
from tqdm.auto import tqdm
import torchvision.transforms as transforms
# env import
import cv2
import os
import gc
import robohive
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


tasks = ["grasp"]
task_name = "reori"
sample_rate = 1.0
dagger = False

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(robot_state):
    # customized for delta hands
    #pdb.set_trace()
    stats = {
        'min': np.min(robot_state,axis=0),
        'max': np.max(robot_state,axis=0)
    }
    for i in range(len(stats['min'])):
        if (stats['min'][i] == stats['max'][i]):
            stats['min'][i] = 0
            stats['max'][i] = stats['min'][i] + 1
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def normalize_images(images):
    # resize image to (120, 160)
    # nomalize to [0,1]
    nimages = images / 255.0
    return nimages

def add_noise(inputs):
     noise = torch.randn_like(inputs) * 0.2 - 0.1 #[-0.1, 0.1]
     return torch.clamp(inputs + noise, min=-1.0, max=1.0)

# dataset
class DiffusionDataset(torch.utils.data.Dataset):
    def __init__(self,
                 demo_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int):
        
        # load all traj's data
        actions = []
        states = []
        episode_ends = [] # the end idx of each traj
        cur_cnt = 0
        demo_data = pickle.load(open(demo_path, 'rb'))
        #demo_traj_length = len(demo_data[0]['actions'])
        # data_list = new_data_list
        # ## only use partially data
        # num_sample = int(sample_rate * len(demo_data))
        # sample_list = random.sample(data_list, num_sample)
        # print("Samples: ", len(sample_list), sample_list)
        for i in range(10): #len(demo_data)
            path = demo_data[i]
            observations_visualize = path['observations_visualization']
            observations = path['observations']  
            length = len(path['actions'])
            cur_state = []
            img_counts = []
            for t in range(length):
                obs_visual = observations_visualize[t]
                obs = observations[t]
                handpos = obs[:24] 
                obj_pos = np.append(obs[33:36], obs[36:39])
                cur_state.append(np.append(handpos, obj_pos))
            cur_state = np.array(cur_state)
            next_state = np.zeros_like(cur_state)
            next_state[:-1,:] = cur_state[1:,:]
            next_state[-1,:] = next_state[-2,:]
            action = next_state 
            
            states.append(cur_state) # pos
            actions.append(action) # d pos
            cur_cnt += len(cur_state)
            episode_ends.append(cur_cnt)

        print("Concatenating images...")
        episode_ends = np.array(episode_ends)
        print("Concatenating actions...")
        actions = np.concatenate(actions)
        print("Concatenating states...")
        states = np.concatenate(states)

        # (N, D)
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            'robot_state': states,
            'action': actions
        }

        # compute start and end of each state-action sequence
        # also handles padding
        print("Creating sample indices...")
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(train_data[key])
            normalized_train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        nsample['robot_state'] = torch.from_numpy(nsample['robot_state'][:self.obs_horizon,:])#add_noise()
        nsample['action'] = torch.from_numpy(nsample['action'][:self.pred_horizon,:])
        return nsample
    


# parameters
pred_horizon = 16
obs_horizon = 2
action_horizon = 8
#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

# create dataset from file
dataset = DiffusionDataset(
    demo_path="./Reorientation/Data/Pen_task.pickle",
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)
# save training data statistics (min, max) for each dim
stats = dataset.stats

# create dataloader
print("Creating dataloader...")
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    num_workers=1,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)

# visualize data in batch
# batch = next(iter(dataloader))
# print("batch['robot_state'].shape:", batch['robot_state'].shape)
# print("batch['action'].shape", batch['action'].shape)
# pdb.set_trace()



#### **Network Demo**

num_hand = 24
num_obj = 6
obs_dim = num_hand + num_obj
action_dim = obs_dim


# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

# for this demo, we use DDPMScheduler with 100 diffusion iterations
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

# device transfer
device = torch.device('cuda:1') #'cuda' # error on nvidia version
_ = noise_pred_net.to(device)


num_epochs = 1000

# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(
    parameters=noise_pred_net.parameters(),
    power=0.75)

# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=noise_pred_net.parameters(),
    lr=1e-4, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

NN_Input_size = 2 * num_hand
NN_Output_size = num_hand
NN_size = [NN_Input_size, 2 * NN_Input_size, 2 * NN_Input_size, NN_Input_size, NN_Output_size] 
Controller_loc = './Reorientation/controller/NN_controller_best.pt'
Controller = FCNetwork(NN_size, nonlinearity='relu') 
Controller.load_state_dict(torch.load(Controller_loc))
Controller.eval() 

e = GymEnv('pen-v0')
e.reset()
Testing_data = e.generate_unseen_data_reorientate(133) # CHANGE

# save_path = "/home/iam-lab/research/Delta-robot/ckpts/block_walking.pt"
save_path = "./Results/Diffusion/ckpts/" + task_name + ".pt"

cur_loss = 0.0
min_loss = np.inf


# epoch loop
for epoch_idx in range(num_epochs):
    epoch_loss = list()
    # batch loop
    for nbatch in dataloader:
        # data normalized in dataset
        # device transfer
        nagent_pos = nbatch['robot_state'].to(device)
        naction = nbatch['action'].to(device)
        B = nagent_pos.shape[0]

        obs_cond = nagent_pos[:,:obs_horizon,:]
        obs_cond = obs_cond.flatten(start_dim=1)
        # print(obs_cond.type())
        # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn(naction.shape, device=device, dtype=torch.float)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        # add noise to the clean images according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_actions = noise_scheduler.add_noise(
            naction, noise, timesteps)
        noisy_actions = noisy_actions.to(device, dtype=torch.float)
        obs_cond = obs_cond.to(device, dtype=torch.float)
        #print(naction.shape, noisy_actions.shape, obs_cond.shape)
        # predict the noise residual
        #pdb.set_trace()
        noise_pred = noise_pred_net(
            noisy_actions, timesteps, global_cond=obs_cond)

        # L2 loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        # optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # step lr scheduler every batch
        # this is different from standard pytorch behavior
        lr_scheduler.step()

        # update Exponential Moving Average of the model weights
        ema.step(noise_pred_net.parameters())

        # logging
        loss_cpu = loss.item()
        #print(f"loss_cpu {loss_cpu}")
        epoch_loss.append(loss_cpu)

    cur_loss = np.mean(epoch_loss)
    print("# epoch, loss: ", epoch_idx, cur_loss)

    if (epoch_idx >= 100 and epoch_idx % 100 == 0):
        with torch.no_grad():
            ema_nets = noise_pred_net
            ema.copy_to(ema_nets.parameters())

            Diffusion_policy_control_reorientation('pen-v0', Controller, ema_nets, Testing_data, dataset.stats, device, obs_horizon, pred_horizon, action_horizon, num_diffusion_iters, noise_scheduler, use_gt = False)


# Weights of the EMA model
# is used for inference
# ema_nets = ema.averaged_model
