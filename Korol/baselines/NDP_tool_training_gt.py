from cProfile import label
from glob import escape
from attr import asdict
import torch
import robohive
import click 
import json
import os
import numpy as np
import gym
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.insert(0, '/home/hongyic/3D_Learning/Kodex/')
from utils.gym_env import GymEnv
from utils.Observables import *
from utils.NDP_evaluation import NDP_policy_control_tool
from utils.Controller import *
from utils.quatmath import quat2euler, euler2quat
from utils.coord_trans import ori_transform, ori_transform_inverse
from utils.gaussian_rnn import RNN
from utils.fc_network import FCNetwork
from utils.demo_loader import hammer_demo_playback # CHANGE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dmp_layer import DMPIntegrator, DMPParameters
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.linalg import logm
import sys
import os
import random
import time
import shutil
import pdb 

class BCDataset(Dataset):
    def __init__(self, demo, img_path):
        self.data = demo
        self.img_path = img_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index]
        current_idx = random.randint(0, len(path)-2)
        robot_current = path[current_idx]['handpos']
        robot_next = path[current_idx+1]['handpos'] 
        
        count = path[current_idx]['count']
        img_path = os.path.join(self.img_path, "rgbd_"+str(count)+".npy") 
        rgbd = np.load(img_path)
        rgbd = np.transpose(rgbd, (2, 0, 1))

        return rgbd, robot_current, robot_next

def main():
    num_demo = 200
    seed = 1 #int(seed)
    number_sample = 200  
    batch_size = 8
    lr = 1e-3
    iter = 200 
    num_obj_feature = 6 #8 #8 #num_objpos + num_objvel + num_init_pos
    num_obj = num_obj_feature
    num_hand = 26
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    
    demo_data_path = pickle.load(open("./Tool/Data/Hammer_task.pickle", 'rb'))
    img_path = "./Tool/Data"
    feature_data_path = './Tool/feature_Tool_demo_full.pickle'
    # Keep using this controller
    Controller_loc = './Tool/controller/NN_controller_best.pt'
    ndp_policy_path = './Tool/Results/NDP/NDP_agent.pt'
    
    # define the input and outputs of the neural network
    NN_Input_size = 2 * num_hand
    NN_Output_size = num_hand
    # NN_size may vary
    NN_size = [NN_Input_size, 2 * NN_Input_size, 2 * NN_Input_size, NN_Input_size, NN_Output_size]  # define a neural network to learn the PID mapping
    Controller = FCNetwork(NN_size, nonlinearity='relu') # load the controller
    Controller.load_state_dict(torch.load(Controller_loc))
    Controller.eval() # eval mode
    # define the input and outputs of the NDPs
    k = 1 
    # only one rollout (only one input image, but the whole traj with T steps)
    # the number of rollout = the number of DMP/NDP trained in each traj.
    # if we only have one rollout, only one NDP, so all the demo data along each traj is used to train the param for this NDP.
    # define the input and outputs of NDPs    
    NN_Input_size = num_obj + num_hand 

    env_name = 'hammer-v0'
    Training_data = hammer_demo_playback(env_name, demo_data_path, feature_data_path, num_demo)
    e = GymEnv(env_name)
    e.reset()
    Testing_data = e.generate_unseen_data_hammer(1000)
    k = 1 # only one rollout (only one input image, but the whole traj with T steps)
    # the number of rollout = the number of DMP/NDP trained in each traj.
    # if we only have one rollout, only one NDP, so all the demo data along each traj is used to train the param for this NDP.
    T = int(len(Training_data[0])/k) # simulation steps
    NN_Input_size = num_obj + num_hand 
    N = 30
    hidden_layer = [NN_Input_size, 128, 256, 128]
    policy = NdpCnn(T=T,l=1, N=N, state_index=np.arange(NN_Input_size), layer_sizes = hidden_layer, seed = seed)
    trainable_params = list(policy.parameters())

    Raw_Input_data = np.zeros([num_demo, len(Training_data[0]), NN_Input_size])   
    NDP_loss = []
    Best_loss = 10000
    print("NDP training starts!\n")
    for k in tqdm(range(num_demo)):
        for t in range(len(Training_data[k])):
            hand_OriState = Training_data[k][t]['handpos']
            # np.append(Training_data[k][t]['toolpos'], 
            obj_OriState = np.append(Training_data[k][t]['objpos'], Training_data[k][t]['nail_goal'])
            Raw_Input_data[k, t] = np.append(hand_OriState, obj_OriState)

    Raw_Output_data = Raw_Input_data.copy()
    batch_iter = int(Raw_Input_data.shape[0] / batch_size)
    shuffle_index = np.arange(0, batch_iter * batch_size)
    batch_loss = 0
    policy = policy.to(device)
    for t in tqdm(range(1001)):
        if t != 0 and t % 20 == 0:
            lr *= 0.9
        print("This is %d iteration and the learning rate is %f."%(t, lr))
        np.random.seed(t)  # fixed the initial seed value for reproducible results
        np.random.shuffle(shuffle_index)     
        policy.train()
        optimizer = torch.optim.Adam(trainable_params, lr=lr)  # try with momentum term
        for mb in range(batch_iter):
            rand_idx = shuffle_index[mb * batch_size: (mb + 1)* batch_size]
            batch_input = torch.from_numpy(Raw_Input_data[rand_idx]).float().to(device)
            batch_label = torch.from_numpy(Raw_Output_data[rand_idx]).float().to(device)
            optimizer.zero_grad()  
            batch_output = policy(batch_input[:, 0], batch_input[:, 0]) 
            loss_criterion = torch.nn.MSELoss()
            #pdb.set_trace()
            loss = loss_criterion(batch_output, batch_label.detach().to(torch.float32))
            batch_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        print(f"t {t}: batch_loss {batch_loss}")
        batch_loss = 0  
        if (t % 100 == 0 and t >= 400): 
            torch.save(policy.state_dict(), ndp_policy_path)
            NN_Input_size = num_obj + num_hand 
            N = 30
            hidden_layer = [NN_Input_size, 128, 256, 128]
            NDP_policy = NdpCnn(T=T,l=1, N=N, state_index=np.arange(NN_Input_size), layer_sizes = hidden_layer)
            NDP_policy.load_state_dict(torch.load(ndp_policy_path))
            NDP_policy = NDP_policy.to("cpu")
            NDP_policy.eval()
            NDP_policy_control_tool(env_name, Controller, NDP_policy, Testing_data, False, num_hand, num_obj, "Drafted", None, None)

def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class NdpCnn(nn.Module):
    # ptu.fanin_init -> uniform distribution
    def __init__(
            self,
            init_w=3e-3,
            layer_sizes=[784, 200, 100],
            hidden_activation=F.relu,
            output_activation=None,
            hidden_init=fanin_init, 
            b_init_value=0.1,
            state_index=np.arange(1),
            N = 5, 
            T = 10, 
            l = 10,
            seed = 0,
            *args,
            **kwargs
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.N = N 
        self.l = l
        self.input_size = len(state_index)
        self.output_size = N*len(state_index) + 2*len(state_index) # 30 * 2 + 2 * 2 = 64
        # output_size should be: omage + goal (params for NDP)
        output_size = self.output_size
        self.T = T
        self.output_activation=torch.tanh
        self.state_index = state_index
        self.output_dim = output_size
        tau = 1
        dt = 0.01  # read from Mujuco env
        self.output_activation=torch.tanh
        self.DMPparam = DMPParameters(N, tau, dt, len(state_index), None, T)  # non-trainable params
        self.func = DMPIntegrator()
        self.register_buffer('DMPp', self.DMPparam.data_tensor)
        self.register_buffer('param_grad', self.DMPparam.grad_tensor)
        # we can fine-tine the layer sizes - NN to learn the parameters for dynamics policy
        self.hidden_activation = hidden_activation
        self.N = N
        self.middle_layers = []
        for i in range(len(layer_sizes)-1): 
            layer = torch.nn.Linear(layer_sizes[i], layer_sizes[i+1])
            hidden_init(layer.weight)
            layer.bias.data.fill_(b_init_value)
            self.middle_layers.append(layer)
            self.add_module("middle_layer_" + str(i), layer)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.last_fc = init_(nn.Linear(layer_sizes[-1], output_size)) # output_size -> output_size

    def forward(self, input, y0, return_preactivations=False):
        x = input
        x = x.view(-1, self.input_size).float()
        activation_fn = self.hidden_activation  # F.relu
        # x.shape = (batch_size, self.input_size)
        for layer in self.middle_layers:
            x = activation_fn(layer(x))
        # x.size = output_size = N*len(state_index) + 2*len(state_index)
        output = self.last_fc(x)*1000  
        y0 = y0.reshape(-1, 1)[:, 0] # y0.shape -> [dim*batch_size]
        dy0 = torch.zeros_like(y0) + 0.01 # set to be small values
        # self.DMPp -> DMP parameters, including: \alpha, \beta, +a_x
        y, dy, ddy = self.func.forward(output, self.DMPp, self.param_grad, None, y0, dy0) # y -> # [batch_size * dim, T]
        y = y.view(input.shape[0], len(self.state_index), -1) # project back to [batch_size, dim, T]
        return y.transpose(2, 1) # [batch_size, T, dim]

    def execute(self, input, y0, return_preactivations=False):
        x = input
        #pdb.set_trace()
        x = x.view(-1, self.input_size).float()
        activation_fn = self.hidden_activation  # F.relu
        # x.shape = (batch_size, self.input_size)
        for layer in self.middle_layers:
            x = activation_fn(layer(x))
        # x.size = output_size = N*len(state_index) + 2*len(state_index)
        output = self.last_fc(x)*1000  
        y0 = y0.reshape(-1, 1)[:, 0] # y0.shape -> [dim*batch_size]
        dy0 = torch.zeros_like(y0) + 0.01 # set to be small values
        # self.DMPp -> DMP parameters, including: \alpha, \beta, a_x
        y, dy, ddy = self.func.forward(output, self.DMPp, self.param_grad, None, y0, dy0) # y -> # [batch_size * dim, T]
        return y.transpose(1, 0).detach().numpy() # [batch_size, T, dim]

if __name__ == '__main__':
    main()

