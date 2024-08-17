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
import time
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.insert(0, '/home/hongyic/3D_Learning/Kodex/')
from utils.gym_env import GymEnv
from utils.Observables import *
from utils.NDP_evaluation import NDP_policy_control_tool
#from utils.resnet import ClassificationNetwork18
from utils.resnet18 import ClassificationNetwork18
from utils.demo_loader import * 
from utils.Controller import *
from utils.quatmath import quat2euler, euler2quat
from utils.coord_trans import ori_transform, ori_transform_inverse
from utils.Koopman_evaluation import generate_feature
from utils.gaussian_rnn import RNN
from utils.fc_network import FCNetwork
#from utils.resnet_vis import *
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
import json
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class NDPDataset(Dataset):
    def __init__(self, demo, img_path):
        self.data = demo
        self.img_path = img_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index]
        current_idx = 0 #random.randint(0, len(path)-52)
        robot_current = path[current_idx]['handpos']
        obj_current = path[current_idx]['objpos']
        robot_next = [path[current_idx+i]['handpos'] for i in range(0,70)]
        obj_next = [path[current_idx+i]['objpos'] for i in range(0,70)]
        count = path[current_idx]['count']
        img_path = os.path.join(self.img_path, "rgbd_"+str(count)+".npy") 
        rgbd = np.load(img_path)
        rgbd = np.transpose(rgbd, (2, 0, 1))

        return rgbd, robot_current, robot_next, obj_current, obj_next

def main():
    num_demo = 50#200
    seed = 1 #int(seed)
    number_sample = 200  
    batch_size = 8
    lr = 1e-3
    iter = 200 
    num_obj_feature = 8 #8 #num_objpos + num_objvel + num_init_pos
    num_obj = num_obj_feature
    num_hand = 26
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    demo_data_path = pickle.load(open("./Tool/Data/Hammer_task.pickle", 'rb'))
    img_path = "./Tool/Data"
    feature_data_path = './Tool/feature_Tool_demo_full.pickle'
    # Keep using this controller
    Controller_loc = './Tool/controller/NN_controller_best.pt'
    ndp_policy_path = './Tool/Results/NDP/NDP_agent_50.pt'
    
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
    NN_Input_size = num_obj + num_hand 

    resnet_model = ClassificationNetwork18(feat_dim = 8)
    resnet_model = resnet_model.float()
    resnet_model = resnet_model.to(device)
    resnet_model.eval()

    '''
    Above: Test the trained koopman dynamics
    Below: Train the koopman dynamics from demo data
    '''
    Training_data = hammer_demo_playback('hammer-v0', demo_data_path, feature_data_path, num_demo)
    ndp_batch_num = 8
    ndp_train_dataset = NDPDataset(Training_data, img_path) #, obj_embedder=obj_embedder
    ndp_train_dataloader = DataLoader(ndp_train_dataset, batch_size=ndp_batch_num, shuffle=True, num_workers=4)

    e = GymEnv('hammer-v0')
    e.reset()

    Testing_data = e.generate_unseen_data_hammer(133) 
    k = 1 # only one rollout (only one input image, but the whole traj with T steps)
    # the number of rollout = the number of DMP/NDP trained in each traj.
    # if we only have one rollout, only one NDP, so all the demo data along each traj is used to train the param for this NDP.
    T = 70 # simulation steps
    NN_Input_size = num_obj + num_hand 
    N = 30
    hidden_layer = [NN_Input_size, 128, 256, 128]
    policy = NdpCnn(T=T,l=1, N=N, state_index=np.arange(NN_Input_size), layer_sizes = hidden_layer, seed = seed)
    #trainable_params = list(policy.parameters())

    Raw_Input_data = np.zeros([200, len(Training_data[0]), NN_Input_size])   
    NDP_loss = []
    Best_loss = 10000
    print("NDP training starts!\n")

    batch_iter = int(Raw_Input_data.shape[0] / batch_size)
    shuffle_index = np.arange(0, batch_iter * batch_size)
    batch_loss = 0

    # optimizer = torch.optim.Adam(trainable_params, lr=lr)  # try with momentum term
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120], gamma=0.4) #milestones=[2,4,6,8,12,20,25,30,50], gamma=0.5
    loss_criterion = torch.nn.MSELoss()
    policy = policy.to(device)
    interval = 60 # eval every 60 seconds
    results = []
    start = time.time()
    for iter_num in range(3001):
        policy.train()
        resnet_model.train()
        total_loss = 0
        iteration_num = 1
        ErrorInOriginalRobot = 0
        if iter_num != 0 and iter_num % 100 == 0:
            lr *= 0.9
        optimizer = torch.optim.Adam(list(policy.parameters()) + list(resnet_model.parameters()), lr=lr) 
        for batch_num, (rgbds, robot_currents, robot_nexts, obj_currents, obj_nexts) in enumerate(ndp_train_dataloader):
            robot_currents = robot_currents.float().to(device)
            # obj_currents = obj_currents.float().to(device)
            rgbds = rgbds.float().to(device)
            _, obj_features = resnet_model(rgbds)
            batch_input = torch.concat([robot_currents, obj_features], dim=1)
            optimizer.zero_grad() 
            batch_output = policy(batch_input, batch_input) # batch_output -> [batch_size, T, dim]
            robot_nexts = torch.stack(robot_nexts).permute(1,0,2)
            obj_nexts = torch.stack(obj_nexts).permute(1,0,2)
            batch_label = torch.concat([robot_nexts, obj_nexts], dim=2).float().to(device)
            #pdb.set_trace()
            loss = loss_criterion(batch_output[:,:,:num_hand], batch_label[:,:,:num_hand])
            total_loss += loss.item()
            loss.backward()
            
            optimizer.step()

        print(f"iter_num {iter_num}, total_loss {total_loss}")
        #print(f"current_time {current_time} start {start}, interval {interval}")
        # if (iter_num % 200 == 0 and iter_num > 0):
        #     resnet_model.eval()
        #     generate_feature(resnet_model, feature_data_path, img_path, device=device, epoch=iter_num)
        #     resnet_model.train()
        if (iter_num % 100 == 0 and iter_num >= 1000):
            #print(f"start eval {iter_num}")
            torch.save(policy.state_dict(), ndp_policy_path)
            NN_Input_size = num_obj + num_hand 
            N = 30
            hidden_layer = [NN_Input_size, 128, 256, 128]
            NDP_policy = NdpCnn(T=T,l=1, N=N, state_index=np.arange(NN_Input_size), layer_sizes = hidden_layer)
            NDP_policy.load_state_dict(torch.load(ndp_policy_path))
            NDP_policy.eval()
            resnet_model.eval()
            success_rate = NDP_policy_control_tool("hammer-v0", Controller, NDP_policy, Testing_data, False, num_hand, num_obj, "Drafted", resnet_model, device)
            results.append(success_rate)
            np.save("NDP_result_train.npy", np.array(results))
            policy.train()
            resnet_model.train()
            start = time.time()


def door_demo_playback(env_name, demo_paths, feature_paths, num_demo, device):
    with open(feature_paths, 'rb') as handle:
        feature_data = pickle.load(handle)
    # return Training_data
    e = GymEnv(env_name)
    e.reset()
    Training_data = []
    Training_data_rgbd = []
    print("Begin loading demo data!")
    print(num_demo)
    sample_index = np.arange(num_demo)
    count = 0
    for i in tqdm(sample_index):
        path = demo_paths[i]
        path_data = []
        e.set_env_state(path['init_state_dict'])
        actions = path['actions']
        observations = path['observations']  
        observations_visualize = path['observations_visualization']
        for tt in range(len(actions)):
            tmp = dict()
            if tt == 0:
                tmp['init'] = path['init_state_dict']
            obs = observations[tt] 
            obs_visual = observations_visualize[tt]
            handpos = obs_visual[:28] 
            tmp['handpos'] = handpos
            tmp['handvel'] = obs_visual[30:58]
            tmp['objpos'] = obs[32:35]
            
            tmp['objvel'] = obs_visual[58:59]
            tmp['handle_init'] = path['init_state_dict']['door_body_pos'] 
            tmp['observation'] = obs[35:38]
            tmp['action'] = actions[tt]

            # dict_value = feature_data[count].values()
            # predict = list(dict_value)[0]
            # tmp['rgbd_feature'] = tmp['objpos'] #predict
            tmp['count'] = count
            count += 1
            path_data.append(tmp)
        Training_data.append(path_data)
    return Training_data


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

