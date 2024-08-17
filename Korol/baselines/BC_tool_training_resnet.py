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
from utils.BC_evaluation import BC_policy_control_tool # CHANGE
from utils.Controller import *
from utils.resnet_vis import *
from utils.Koopman_evaluation import generate_feature
#from utils.resnet import *
from utils.quatmath import quat2euler, euler2quat
from utils.coord_trans import ori_transform, ori_transform_inverse
from utils.fc_network import FCNetwork
from utils.demo_loader import hammer_demo_playback # CHANGE
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
        obj_current = np.append(path[current_idx]['objpos'],  path[current_idx]['nail_goal'])
        robot_next = path[current_idx+1]['handpos']         
        count = path[current_idx]['count']
        img_path = os.path.join(self.img_path, "rgbd_"+str(count)+".npy") 
        rgbd = np.load(img_path)
        rgbd = np.transpose(rgbd, (2, 0, 1))
        return rgbd, robot_current, obj_current, robot_next

def main():
    num_demo = 200
    seed = 1 #int(seed)
    number_sample = 200  
    batch_size = 8
    lr = 1e-4
    num_obj_feature = 8 #8 #num_objpos + num_objvel + num_init_pos CHANGE
    num_hand = 26 # CHANGE
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    
    demo_data_path = pickle.load(open("./Tool/Data/Hammer_task.pickle", 'rb'))
    img_path = "./Tool/Data"
    feature_data_path = './Tool/feature_Tool_demo_full.pickle'

    env_name = 'hammer-v0'
    Training_data = hammer_demo_playback(env_name, demo_data_path, feature_data_path, num_demo)

    e = GymEnv(env_name)
    e.reset()
    Testing_data = e.generate_unseen_data_hammer(1000) # CHANGE

    bc_batch_num = 8
    bc_train_dataset = BCDataset(Training_data, img_path) #, obj_embedder=obj_embedder
    bc_train_dataloader = DataLoader(bc_train_dataset, batch_size=bc_batch_num, shuffle=True, num_workers=4)

    Raw_Input_data = np.zeros([num_demo * (len(Training_data[0]) - 1), num_obj_feature + num_hand])  #num_obj_feature + 
    Raw_Output_data = np.zeros([num_demo * (len(Training_data[0]) - 1), num_obj_feature + num_hand])   

    index = 0
    NN_Input_size = 2 * num_hand
    NN_Output_size = num_hand
    NN_size = [NN_Input_size, 2 * NN_Input_size, 2 * NN_Input_size, NN_Input_size, NN_Output_size] 
    Controller_loc = './Tool/controller/NN_controller_best.pt'
    Controller = FCNetwork(NN_size, nonlinearity='relu') 
    Controller.load_state_dict(torch.load(Controller_loc))
    Controller.eval() 

    resnet_model = ClassificationNetwork18(feat_dim = 8)
    resnet_model = resnet_model.float()
    resnet_model = resnet_model.to(device)
    resnet_model.train()
    
    # BC Agent
    NN_Input_size = num_hand + num_obj_feature
    NN_Output_size = NN_Input_size
    hidden_size = (64, 128, 64)
    NN_size = (NN_Input_size, ) + hidden_size + (NN_Output_size, )
    BC_agent = FCNetwork(NN_size, seed = seed, nonlinearity='relu') # define the NN network
    BC_agent = BC_agent.to(device)
    
    BC_loss = []
    Best_loss = 10000

    batch_iter = int(Raw_Input_data.shape[0] / batch_size)
    shuffle_index = np.arange(0, batch_iter * batch_size)

    BC_agent = BC_agent.to(device)
    loss_criterion = torch.nn.MSELoss()
    for t in range(300):
        #print("This is %d iteration and the learning rate is %f."%(t, ))   
        BC_agent.train()
        resnet_model.train()
        total_loss = 0
        iteration_num = 10
        if t != 0 and t % 20 == 0:
            lr *= 0.9
        optimizer = torch.optim.Adam(list(BC_agent.parameters()) + list(resnet_model.parameters()), lr=lr)
        for batch_num, (rgbds, robot_currents, obj_currents, robot_nexts) in enumerate(bc_train_dataloader):
            rgbds = rgbds.float().to(device)
            robot_currents = robot_currents.float().to(device)
            robot_nexts = robot_nexts.float().to(device)
            _, obj_features = resnet_model(rgbds)
            #obj_features = obj_currents.float().to(device)
            bc_input = torch.concat([robot_currents, obj_features], dim=1)
            optimizer.zero_grad() 
            bc_output = BC_agent(bc_input)
            loss = loss_criterion(bc_output[:, :num_hand], robot_nexts)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"This is {t} iteration and total_loss {total_loss}, the learning rate is {lr}")
    print("BC training ends!\n")
    BC_agent.eval()   
    resnet_model.eval()  
    BC_agent = BC_agent.to('cpu')
    # with torch.no_grad():
    #     BC_policy_control_tool("hammer-v0", Controller, BC_agent, Testing_data, num_hand, device, resnet_model, use_gt=False)
    generate_feature(resnet_model, './Tool/feature_tool_demo_full.pickle', "./Tool/Data", device=device, epoch=t)
    


if __name__ == '__main__':
    main()