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
import sys
sys.path.insert(0, '/home/hongyic/3D_Learning/Kodex/')
from utils.gym_env import GymEnv
from utils.Observables import *
from utils.BC_evaluation import BC_policy_control_door
from utils.Controller import *
from utils.resnet import *
#from utils.resnet_vis import *
from utils.Koopman_evaluation import generate_feature
from utils.quatmath import quat2euler, euler2quat
from utils.coord_trans import ori_transform, ori_transform_inverse
from utils.fc_network import FCNetwork
from torch.utils.data import Dataset, DataLoader
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
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

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
        obj_current = np.append(path[current_idx]['objpos'], np.append(path[current_idx]['objvel'], path[current_idx]['handle_init']))

        robot_next = path[current_idx+1]['handpos']         
        count = path[current_idx]['count']
        img_path = os.path.join(self.img_path, "rgbd_"+str(count)+".npy") 
        rgbd = np.load(img_path)
        rgbd = np.transpose(rgbd, (2, 0, 1))
        return rgbd, robot_current, obj_current, robot_next

def main():
    num_demo = 200
    seed = 1 
    number_sample = 200  
    batch_size = 8
    lr = 1e-4
    iter = 101
    num_obj_feature = 8 #num_objpos + num_objvel + num_init_pos
    num_hand = 28
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    demo_data_path = pickle.load(open("./Door/Data/Testing.pickle", 'rb'))
    img_path = "./Door/Data"
    feature_data_path = './Door/feature_door_demo_full.pickle'
    # resnet model for image
    resnet_model = ClassificationNetwork18(feat_dim = 8)
    resnet_model = resnet_model.float()
    resnet_model = resnet_model.to(device)
    resnet_model.eval()
    
    env_name = 'door-v0'
    Training_data = door_demo_playback("door-v0", demo_data_path, feature_data_path, num_demo, device)

    e = GymEnv("door-v0")
    e.reset()
    Testing_data = e.generate_unseen_data_door(233) 

    bc_batch_num = 8
    bc_train_dataset = BCDataset(Training_data, img_path) #, obj_embedder=obj_embedder
    bc_train_dataloader = DataLoader(bc_train_dataset, batch_size=bc_batch_num, shuffle=True, num_workers=4)

    Raw_Input_data = np.zeros([num_demo * (len(Training_data[0]) - 1), num_obj_feature + num_hand])  #num_obj_feature + 
    Raw_Output_data = np.zeros([num_demo * (len(Training_data[0]) - 1), num_obj_feature + num_hand])   

    index = 0
    
    NN_Input_size = 2 * num_hand
    NN_Output_size = num_hand
    NN_size = [NN_Input_size, 2 * NN_Input_size, 2 * NN_Input_size, NN_Input_size, NN_Output_size] 
    Controller_loc = './Door/controller/NN_controller_best.pt'
    Controller = FCNetwork(NN_size, nonlinearity='relu') 
    Controller.load_state_dict(torch.load(Controller_loc))
    Controller.eval() 
    
    # BC Agent
    NN_Input_size = num_hand + num_obj_feature
    NN_Output_size = NN_Input_size
    hidden_size = (64, 128, 64)
    NN_size = (NN_Input_size, ) + hidden_size + (NN_Output_size, )
    BC_agent = FCNetwork(NN_size, seed = seed, nonlinearity='relu') # define the NN network
    BC_agent = BC_agent.to(device)
    
    # optimize both bc and resnet
    optimizer = torch.optim.Adam(list(BC_agent.parameters()) + list(resnet_model.parameters()), lr=lr) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15,20,25,30,35,40,45,50,60,90,120], gamma=0.9) #milestones=[2,4,6,8,12,20,25,30,50], gamma=0.5

    
    BC_loss = []
    Best_loss = 10000

    batch_iter = int(Raw_Input_data.shape[0] / batch_size)
    shuffle_index = np.arange(0, batch_iter * batch_size)

    BC_agent = BC_agent.eval().to('cpu')
    resnet_model.eval()
    #BC_policy_control("door-v0", Controller, BC_agent, Training_data, num_hand, device, resnet_model)
    resnet_model.train()
    
    
    BC_agent = BC_agent.to(device)
    loss_criterion = torch.nn.MSELoss()
    for t in range(iter): 
        BC_agent.train()
        resnet_model.train()
        total_loss = 0
        iteration_num = 10
        for _ in range(iteration_num):
            for batch_num, (rgbds, robot_currents, obj_currents, robot_nexts) in enumerate(bc_train_dataloader):
                robot_currents = robot_currents.float().to(device)
                rgbds = rgbds.float().to(device)
                _, obj_features = resnet_model(rgbds)
                robot_nexts = robot_nexts.float().to(device)
                # obj_features = obj_currents.float().to(device)
                bc_input = torch.concat([robot_currents, obj_features], dim=1)
                optimizer.zero_grad() 
                bc_output = BC_agent(bc_input)
                loss = loss_criterion(bc_output[:, :num_hand], robot_nexts)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
        scheduler.step()
        print(f"This is {t} iteration and total_loss {total_loss}, the learning rate is {scheduler.get_lr()}")
        if (t % 10 == 0 and t > 0):
            BC_agent.eval() 
            resnet_model.eval()
            BC_agent = BC_agent.eval().to('cpu')
            with torch.no_grad():
                BC_policy_control_door("door-v0", Controller, BC_agent, Testing_data, num_hand, device, resnet_model, use_gt = False)
            #generate_feature(resnet_model, './Door/feature_door_demo_full.pickle', "./Door/Data", device=device, epoch=t)
            BC_agent = BC_agent.to(device)
            resnet_model.train()
            
    print("BC training ends!\n")




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

            dict_value = feature_data[count].values()
            predict = list(dict_value)[0]
            tmp['rgbd_feature'] = tmp['objpos'] #predict
            tmp['count'] = count
            count += 1
            path_data.append(tmp)
        Training_data.append(path_data)
    return Training_data


if __name__ == '__main__':
    main()