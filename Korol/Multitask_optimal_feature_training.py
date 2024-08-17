from cProfile import label
from glob import escape
from attr import asdict
import torch
import click 
import json
import os
import numpy as np
import gym
import gc
import pickle
from utils.gym_env import GymEnv
from utils.Observables import *
from utils.Koopman_evaluation import train_koopman
from utils.Koopman_multitask_evaluation import koopman_door_multi, koopman_hammer_multi, koopman_relocate_multi, koopman_reori_multi
from utils.Controller import *
from utils.quatmath import quat2euler, euler2quat
from utils.coord_trans import ori_transform, ori_transform_inverse
from utils.resnet_multitask import ResNet34, ResNet34_freq, ResNet18_freq, ResNet50_freq, ResNet50, ResNet18
from utils.demo_loader import door_demo_playback, hammer_demo_playback, relocate_demo_playback, reorientation_demo_playback
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

class RGBDDataset(Dataset):
    def __init__(self, img_path, demo_feature_data_path):
        self.img_path = img_path
        img_data = os.listdir(img_path)
        data = [img for img in img_data if img[-3:] == "npy"]
        self.length = len(data)
        if (demo_feature_data_path is not None):
            with open(demo_feature_data_path, 'rb') as f:
                self.demo_feature = pickle.load(f)
        else:
            self.demo_feature = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        path = os.path.join(self.img_path, "rgbd_"+str(index)+".npy") 
        rgbd = np.load(path)
        rgbd = np.transpose(rgbd, (2, 0, 1))
        desire_posori = np.array([0.,0,0])
        if (self.demo_feature is not None):
            dict = self.demo_feature[index]
            if ('desired_pos' in dict):
                desire_posori = dict['desired_pos']
                #print(f"desire_posori {desire_posori}")
                #print(f"desire_posori {desire_posori}")
            elif ('desired_ori' in dict):
                desire_posori = dict['desired_ori']
                #print(f"desired_ori {desire_posori}")
        return rgbd, index, desire_posori

# demo_path = "/home/hongyic/3D_Learning/Kodex/Door/Data/door_full_objpos_feature.pickle"
# img_path = "/home/hongyic/3D_Learning/Kodex/Door/Data"
def generate_feature(model, feature_path, img_path, device="cuda", desire_posori_features = None):
    batch_size = 32
    Training_data = []
    model.eval()
    train_dataset = RGBDDataset(img_path, desire_posori_features)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    error = 0
    with torch.no_grad():
        for batch_num, (rgbds, indexs, desire_posoris) in enumerate(train_dataloader):
            rgbds = rgbds.float().to(device)
            desire_posoris = desire_posoris.float().to(device)
            outputs_feat = model(rgbds, desire_posoris)
            outputs = outputs_feat.detach().cpu().numpy() #outputs_pos
            # if (batch_num % 400 == 0):
            #     print(outputs_feat)
            for index, output, rgbd in zip(indexs, outputs, rgbds):
                Training_data.append({index:output})
    with open(feature_path, 'wb') as handle:
        pickle.dump(Training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

class DynamicsPredictionDataset(Dataset):
    def __init__(self, demos, data_lengths, img_paths, length = 20):
        self.datas = demos
        self.img_paths = img_paths
        self.length = length
        self.data_lengths = data_lengths #np.cumsum(data_lengths)

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        i = 0
        for i in range(len(self.data_lengths)):
            if (index < self.data_lengths[i]):
                break
        path = self.datas[index]
        current_idx = random.randint(0, len(path)-(self.length+2))
        robot_current = path[current_idx]['handpos']
        count = path[current_idx]['count'] #+ self.init_counts[i]
        # img_path = "./Door/Distinct_Data/rgbd_"+str(count)+".npy"
        img_path = self.img_paths[i] + "/rgbd_" + str(count) + ".npy"
        rgbd = np.load(img_path) # NAZA , mmap_mode="r"
        rgbd = np.transpose(rgbd, (2, 0, 1))
        desire_pos_ori = np.array([0.,0,0])
        if (i == 2):
            desire_pos_ori = path[current_idx]['desired_pos']
            #print(f"desire_pos_ori {desire_pos_ori}")
        elif (i == 3):
            desire_pos_ori = path[current_idx]['desired_ori']
        robot_next = [path[current_idx+i]['handpos'] for i in range(1,self.length+1)]
        return rgbd, robot_current, desire_pos_ori, robot_next


def main(args):
    learningRate = 1e-4 
    num_demo = 200
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    folder_name = os.getcwd()
    num_hand = 30 
    num_obj = 8 

    NN_Input_size = 2 * 28
    NN_Output_size = 28
    NN_size = [NN_Input_size, 2 * NN_Input_size, 2 * NN_Input_size, NN_Input_size, NN_Output_size] 
    Controller_loc_door = 'Door/Results/Controller/NN_controller_best.pt'
    Controller_door = FCNetwork(NN_size, nonlinearity='relu')
    Controller_door.load_state_dict(torch.load(Controller_loc_door))
    Controller_door.eval() # eval mode
    
    NN_Input_size = 2 * 26
    NN_Output_size = 26
    NN_size = [NN_Input_size, 2 * NN_Input_size, 2 * NN_Input_size, NN_Input_size, NN_Output_size]  
    Controller_loc_tool = 'Tool/Results/Controller/NN_controller_best.pt'
    Controller_tool = FCNetwork(NN_size, nonlinearity='relu')
    Controller_tool.load_state_dict(torch.load(Controller_loc_tool))
    Controller_tool.eval() # eval mode
    
    NN_Input_size = 2 * 30
    NN_Output_size = 30
    NN_size = [NN_Input_size, 2 * NN_Input_size, 2 * NN_Input_size, NN_Input_size, NN_Output_size]  
    Controller_loc_relocate = 'Relocation/Results/Controller/NN_controller_best.pt'
    Controller_relocate = FCNetwork(NN_size, nonlinearity='relu')
    Controller_relocate.load_state_dict(torch.load(Controller_loc_relocate))
    Controller_relocate.eval() # eval mode
    
    NN_Input_size = 2 * 24
    NN_Output_size = 24
    NN_size = [NN_Input_size, 2 * NN_Input_size, 2 * NN_Input_size, NN_Input_size, NN_Output_size]  
    Controller_loc_reorientate = 'Reorientation/Results/Controller/NN_controller_best.pt'
    Controller_reorientate = FCNetwork(NN_size, nonlinearity='relu')
    Controller_reorientate.load_state_dict(torch.load(Controller_loc_reorientate))
    Controller_reorientate.eval() # eval mode

    # training parameters
    # if not args.use_freq:
    #     resnet_model = ResNet34(feat_dim = 8)
    # else:
    #     resnet_model = ResNet34_freq(feat_dim = 8)
    resnet_model = ResNet18(feat_dim = 8) # ResNet34_freq
    resnet_model = resnet_model.float().to(device)
    resnet_model.eval()
    
    door_env = GymEnv("door-v0")
    door_env.reset()
    Testing_data_door = door_env.generate_unseen_data_door(200)#[200:] 
    tool_env = GymEnv("hammer-v0")
    tool_env.reset()
    Testing_data_hammer = tool_env.generate_unseen_data_hammer(200)#[200:] 
    relocate_env = GymEnv("relocate-v0")
    relocate_env.reset()
    Testing_data_relocate = relocate_env.generate_unseen_data_relocate(200)#[200:]
    reori_env = GymEnv("pen-v0")
    reori_env.reset()
    Testing_data_reori = reori_env.generate_unseen_data_reorientate(200)#[200:]
    #pdb.set_trace()
    door_img_path = "./Door/Data"
    tool_img_path = "./Tool/Data"
    relocate_img_path = "./Relocation/Data"
    reori_img_path = "./Reorientation/Data"
    img_paths = [door_img_path, tool_img_path, relocate_img_path, reori_img_path]
    koopman_save_path = os.path.join(folder_name, "Door/multi_koopmanMatrix.npy")
    door_demo_data = pickle.load(open("./Door/Data/Testing.pickle", 'rb'))
    tool_demo_data = pickle.load(open("./Tool/Data/Hammer_task.pickle", 'rb'))
    relocate_demo_data = pickle.load(open("./Relocation/Data/Relocate_task.pickle", 'rb'))
    reori_demo_data = pickle.load(open("./Reorientation/Data/Pen_task.pickle", 'rb'))
    door_feature_data_path = './Door/feature_door_demo_full.pickle'
    tool_feature_data_path = './Tool/feature_tool_demo_full.pickle'
    relocate_feature_data_path = './Relocation/feature_relocate_demo_full.pickle'
    reori_feature_data_path = './Reorientation/feature_reori_demo_full.pickle'
    Koopman = DraftedObservable(num_hand, num_obj)
    
    generate_feature(resnet_model, door_feature_data_path, door_img_path, device=device)
    Training_data_door = door_demo_playback("door-v0", door_demo_data, door_feature_data_path, num_demo, multi_task=True)
    generate_feature(resnet_model, tool_feature_data_path, tool_img_path, device=device)
    Training_data_tool = hammer_demo_playback("hammer-v0", tool_demo_data, tool_feature_data_path, num_demo, multi_task=True)
    Relocate_desired_posori_path = f'./Relocation/Data/full_features.pickle'
    generate_feature(resnet_model, relocate_feature_data_path, relocate_img_path, device=device, desire_posori_features = Relocate_desired_posori_path)
    Training_data_relocate = relocate_demo_playback("relocate-v0", relocate_demo_data, relocate_feature_data_path, num_demo, multi_task=True)
    
    Reori_desired_posori_path = f'./Reorientation/Data/full_features.pickle'
    generate_feature(resnet_model, reori_feature_data_path, reori_img_path, device=device, desire_posori_features = Reori_desired_posori_path)
    Training_data_reori = reorientation_demo_playback("pen-v0", reori_demo_data, reori_feature_data_path, num_demo, multi_task=True)
    

    Training_data = Training_data_door + Training_data_tool + Training_data_relocate + Training_data_reori
    data_lengths = [len(Training_data_door), len(Training_data_door)+len(Training_data_tool), 600, 800]

    # Resnet Dynamics Training dataset
    dynamics_batch_num = 8
    dynamics_train_dataset = DynamicsPredictionDataset(Training_data, data_lengths, img_paths, length = int(args.N)) 
    dynamics_train_dataloader = DataLoader(dynamics_train_dataset, batch_size=dynamics_batch_num, shuffle=True, num_workers=0)

    train_koopman(Training_data, num_hand, num_obj, koopman_save_path)
    cont_koopman_operator = np.load(koopman_save_path) # matrix_file
    #koopman_policy_control("door-v0", Controller, Koopman, cont_koopman_operator, Testing_data, False, num_hand, num_obj, "Drafted", resnet_model=resnet_model, device=device) #use_resnet=True, resnet_model=resnet_model
    cont_koopman_operator = torch.from_numpy(cont_koopman_operator).to(device)
    

    loss = torch.nn.L1Loss()
    results = []
    # Train 
    for epoch in range(2001):
        #print(epoch)
        resnet_model.train()
        if epoch != 0 and epoch % 10 == 0:
            learningRate *= 0.9
        optimizer_feature = torch.optim.Adam(resnet_model.parameters(), lr=learningRate)
        ErrorInOriginalRobot = 0
        total_loss = 0
        for batch_num, (rgbds, robot_currents, desire_pos_oris, robot_nexts) in enumerate(dynamics_train_dataloader):
            #pdb.set_trace()
            robot_currents = robot_currents.float().to(device)
            rgbds = rgbds.float().to(device)
            desire_pos_oris = desire_pos_oris.float().to(device)
            pred_feats = resnet_model(rgbds, desire_pos_oris)
            batch_size = len(robot_currents)
            for i in range(batch_size):
                hand_OriState = robot_currents[i]
                obj_OriState = pred_feats[i]
                z_t_computed = Koopman.z_torch(hand_OriState, obj_OriState).to(device)
                for t in range(len(robot_nexts)):
                    robot_next = robot_nexts[t][i].float().to(device)
                    #pdb.set_trace()
                    z_t_1_computed = torch.matmul(cont_koopman_operator.float(), z_t_computed.float()) 
                    ErrorInOriginalRobot += loss(z_t_1_computed[:num_hand], robot_next)
                    z_t_computed = z_t_1_computed
            ErrorInOriginalRobot *= 0.05 # weights
            total_loss += ErrorInOriginalRobot.item()
            ErrorInOriginalRobot.backward()
            optimizer_feature.step()
            ErrorInOriginalRobot = 0
            optimizer_feature.zero_grad()
            del rgbds
            del robot_currents
            del robot_nexts
            gc.collect()

        print(f"epoch {epoch}, loss {total_loss/len(dynamics_train_dataloader)}")
        total_loss = 0
        if (epoch % 50 == 0 and epoch >= 50):
            resnet_model.eval()
            with torch.no_grad():
                generate_feature(resnet_model, door_feature_data_path, door_img_path, device=device)
                Training_data_door = door_demo_playback("door-v0", door_demo_data, door_feature_data_path, num_demo, multi_task=True)
                generate_feature(resnet_model, tool_feature_data_path, tool_img_path, device=device)
                Training_data_tool = hammer_demo_playback("hammer-v0", tool_demo_data, tool_feature_data_path, num_demo, multi_task=True)
                Relocate_desired_posori_path = f'./Relocation/Data/full_features.pickle'
                generate_feature(resnet_model, relocate_feature_data_path, relocate_img_path, device=device, desire_posori_features = Relocate_desired_posori_path)
                Training_data_relocate = relocate_demo_playback("relocate-v0", relocate_demo_data, relocate_feature_data_path, num_demo, multi_task=True)
                Reori_desired_posori_path = f'./Reorientation/Data/full_features.pickle'
                generate_feature(resnet_model, reori_feature_data_path, reori_img_path, device=device, desire_posori_features = Reori_desired_posori_path)
                Training_data_reori = reorientation_demo_playback("pen-v0", reori_demo_data, reori_feature_data_path, num_demo, multi_task=True)
                Training_data = Training_data_door + Training_data_tool + Training_data_relocate + Training_data_reori
                train_koopman(Training_data, num_hand, num_obj, koopman_save_path)
                cont_koopman_operator = np.load(koopman_save_path) # matrix_file
                
                koopman_reori_multi("pen-v0", Controller_reorientate, Koopman, cont_koopman_operator, Testing_data_reori, False, num_hand, num_obj, "Drafted", resnet_model=resnet_model, device=device) 
                koopman_relocate_multi("relocate-v0", Controller_relocate, Koopman, cont_koopman_operator, Testing_data_relocate, False, num_hand, num_obj, "Drafted", resnet_model=resnet_model, device=device)
                koopman_door_multi("door-v0", Controller_door, Koopman, cont_koopman_operator, Testing_data_door, False, num_hand, num_obj, "Drafted", resnet_model=resnet_model, device=device) 
                koopman_hammer_multi("hammer-v0", Controller_tool, Koopman, cont_koopman_operator, Testing_data_hammer, False, num_hand, num_obj, "Drafted", resnet_model=resnet_model, device=device) 
                
                cont_koopman_operator = torch.from_numpy(cont_koopman_operator).to(device)
                #torch.save({'model_state_dict': resnet_model.state_dict()}, "./model/door_full_resnet")
            resnet_model.train()

    print(f"results {results}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', default=40, help='dynamic prediction length')
    parser.add_argument('--use_freq', action='store_true', help='use frequency feature')
    args = parser.parse_args()
    print(f" N length {args.N}")
    main(args)


