from cProfile import label
from glob import escape
from attr import asdict
import torch
import click 
import json
import os
import numpy as np
import gym
import pickle
from utils.Observables import *
from utils.Koopman_evaluation import *
from utils.Controller import *
from utils.quatmath import quat2euler, euler2quat
from utils.coord_trans import ori_transform, ori_transform_inverse
from utils.resnet import *
#from utils.resnet18 import *
from utils.demo_loader import *
from scipy.linalg import logm
import scipy
import sys
import os
import random
import time 
import shutil
import robohive
import pdb 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import pdb 

class DynamicsPredictionDataset(Dataset):
    def __init__(self, demo, img_path, length = 20, step_interval = 10):
        self.data = demo
        self.img_path = img_path
        self.length = length
        self.step_interval = step_interval

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index]
        current_idx = random.randint(0, len(path)-(self.length+2))
        robot_current = path[current_idx]['handpos']
        count = path[current_idx]['count']
        # img_path = "./Door/Distinct_Data/rgbd_"+str(count)+".npy"
        
        rgbds = []
        for i in range(0, self.length+1, self.step_interval):
            img_path = self.img_path + "/rgbd_" + str(count+i) + ".npy"
            rgbd = np.load(img_path)
            rgbd = np.transpose(rgbd, (2, 0, 1))
            rgbds.append(rgbd)
        rgbds = np.array(rgbds)
        robot_next = [path[current_idx+i]['handpos'] for i in range(1,self.length+1)]
        return rgbds, robot_current, robot_next


def main(args):
    learningRate = 1e-4 
    num_demo = 200
    device = "cuda" if torch.cuda.is_available() else "cpu"
    folder_name = os.getcwd()
    num_hand = 28 
    num_obj = 8 
    step_interval = 10 # Object feature loss step interval


    img_path = "./Door/Data"
    Controller_loc = './Door/controller/NN_controller_best.pt'
    NN_Input_size = 2 * num_hand
    NN_Output_size = num_hand
    NN_size = [NN_Input_size, 2 * NN_Input_size, 2 * NN_Input_size, NN_Input_size, NN_Output_size]  # define a neural network to learn the PID mapping
    Controller = FCNetwork(NN_size, nonlinearity='relu')
    Controller.load_state_dict(torch.load(Controller_loc))
    Controller.eval() # eval mode

    # training parameters
    resnet_model = ClassificationNetwork18(feat_dim = 8)
    resnet_model = resnet_model.float().to(device)
    resnet_model.eval()
    
    env = "Door"

    e = GymEnv("door-v0")
    e.reset()
    Testing_data = e.generate_unseen_data_door(200) 

    koopman_save_path = os.path.join(folder_name, "Door/koopmanMatrix.npy")
    demo_data = pickle.load(open("./Door/Data/Testing.pickle", 'rb'))
    feature_data_path = './Door/feature_door_demo_full.pickle'
    Koopman = DraftedObservable(num_hand, num_obj)

    generate_feature(resnet_model, feature_data_path, img_path, device=device, img_size=num_demo*70)
    Training_data = door_demo_playback("door-v0", demo_data, feature_data_path, num_demo)
    Training_data_length = np.sum([len(demo) for demo in Training_data])
    # Resnet Dynamics Training dataset
    dynamics_batch_num = 8
    dynamics_train_dataset = DynamicsPredictionDataset(Training_data, img_path, length = int(args.N), step_interval = step_interval) 
    dynamics_train_dataloader = DataLoader(dynamics_train_dataset, batch_size=dynamics_batch_num, shuffle=True, num_workers=4)

    train_koopman(Training_data, num_hand, num_obj, koopman_save_path)
    cont_koopman_operator = np.load(koopman_save_path) # matrix_file
    success_rate = koopman_policy_control("door-v0", Controller, Koopman, cont_koopman_operator, Testing_data, False, num_hand, num_obj, "Drafted", resnet_model=resnet_model, device=device)
    cont_koopman_operator = torch.from_numpy(cont_koopman_operator).to(device)
    
    loss = torch.nn.L1Loss()
    loss_results = []
    # Train 
    for epoch in range(201):
        resnet_model.train()
        if epoch != 0 and epoch % 10 == 0:
            learningRate *= 0.9
        optimizer_feature = torch.optim.Adam(resnet_model.parameters(), lr=learningRate)
        ErrorInOriginalRobot = 0
        total_loss = 0
        for batch_num, (rgbds, robot_currents, robot_nexts) in enumerate(dynamics_train_dataloader):
            robot_currents = robot_currents.float().to(device)
            #pdb.set_trace()
            
            batch_size = len(robot_currents)
            for i in range(batch_size):
                hand_OriState = robot_currents[i]
                rgbds_single = rgbds[i].float().to(device)
                _, pred_feats = resnet_model(rgbds_single)
                obj_OriState = pred_feats[0]
                z_t_computed = Koopman.z_torch(hand_OriState, obj_OriState).to(device)
                for t in range(len(robot_nexts)):
                    robot_next = robot_nexts[t][i].float().to(device)
                    z_t_1_computed = torch.matmul(cont_koopman_operator.float(), z_t_computed.float()) 
                    loss_val = loss(z_t_1_computed[:num_hand], robot_next)
                    ErrorInOriginalRobot += loss_val
                    if (t > 0 and (t+1) % step_interval == 0):
                        #pdb.set_trace()
                        loss_val = loss(z_t_1_computed[2*num_hand:2*num_hand+num_obj], pred_feats[int((t+1)/step_interval)])
                        ErrorInOriginalRobot += loss_val
                        #pdb.set_trace()
                    z_t_computed = z_t_1_computed
                    total_loss += loss_val.item()
            ErrorInOriginalRobot *= 0.05 # weights
            ErrorInOriginalRobot.backward()
            optimizer_feature.step()
            ErrorInOriginalRobot = 0
            optimizer_feature.zero_grad()

        loss_results.append(total_loss)
        print(f"epoch {epoch}, total_loss {total_loss*0.05}")
        if (epoch % 50 == 0 and epoch > 0):
            resnet_model.eval()
            with torch.no_grad():
                generate_feature(resnet_model, feature_data_path, img_path, device=device, img_size=num_demo*70)
                Training_data = door_demo_playback("door-v0", demo_data, feature_data_path, num_demo)
                train_koopman(Training_data, num_hand, num_obj, koopman_save_path)
                cont_koopman_operator = np.load(koopman_save_path) # matrix_file
                if (epoch % 50 == 0):
                    success_rate = koopman_policy_control("door-v0", Controller, Koopman, cont_koopman_operator, Testing_data, False, num_hand, num_obj, "Drafted", resnet_model=resnet_model, device=device) #use_resnet=True, resnet_model=resnet_model
                cont_koopman_operator = torch.from_numpy(cont_koopman_operator).to(device)
            resnet_model.train()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', default=40, help='dynamic prediction length')
    args = parser.parse_args()
    print(f" N length {args.N}")
    main(args)


