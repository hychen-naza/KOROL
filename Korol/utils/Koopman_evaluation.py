"""
Define the functions that are used to test the learned koopman matrix
"""
from glob import escape
from attr import asdict
import numpy as np
import time
from tqdm import tqdm
from utils.gym_env import GymEnv
from utils.quatmath import euler2quat
from utils.Observables import *
from utils.coord_trans import ori_transform, ori_transform_inverse
from utils.resnet import *
import torch
import pdb
import torch.nn as nn  
import os
import scipy
from torch.utils.data import Dataset, DataLoader
import pickle 
from PIL import Image
import matplotlib.pyplot as plt
import gc

class RGBDDataset(Dataset):
    def __init__(self, img_path, img_size=None):
        self.img_path = img_path
        img_data = os.listdir(img_path)
        data = [img for img in img_data if img[-3:] == "npy"]
        self.length = len(data) if img_size is None else img_size

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        path = os.path.join(self.img_path, "rgbd_"+str(index)+".npy") 
        rgbd = np.load(path)
        rgbd = np.transpose(rgbd, (2, 0, 1))
        return rgbd, index


def generate_feature(model, feature_path, img_path, device="cuda", epoch=0, img_size=None):
    batch_size = 32
    Training_data = []
    model.eval()
    train_dataset = RGBDDataset(img_path, img_size=img_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    error = 0
    count = 0
    vis_imgs = []
    rgbd_imgs = []
    with torch.no_grad():
        for batch_num, (rgbds, indexs) in enumerate(train_dataloader):
            rgbds = rgbds.float().to(device)
            outputs_pos, outputs_feat = model(rgbds)
            outputs = outputs_feat.detach().cpu().numpy() #outputs_pos
            for index, output, rgbd in zip(indexs, outputs, rgbds):
                Training_data.append({index:output})
            count += batch_size
            del rgbds
            gc.collect()
        
    with open(feature_path, 'wb') as handle:
        pickle.dump(Training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_env_rgbd(e):
    img_obs, depth_obs = e.env.mj_render()
    depth_obs = depth_obs[...,np.newaxis]
    img_obs = (img_obs.astype(np.uint8) - 128.0) / 128
    rgbd = np.concatenate((img_obs, depth_obs),axis=2)
    rgbd = np.transpose(rgbd, (2, 0, 1))
    rgbd = rgbd[np.newaxis, ...]
    return rgbd

def train_koopman(Training_data, num_hand, num_obj, koopman_save_path):
    Koopman = DraftedObservable(num_hand, num_obj)
    num_obs = Koopman.compute_observable(num_hand, num_obj)

    A = np.zeros((num_obs, num_obs))  
    G = np.zeros((num_obs, num_obs))
    ## loop to collect data
    print("Drafted koopman training starts!\n")
    for k in tqdm(range(len(Training_data))):
        hand_OriState = Training_data[k][0]['handpos']
        obj_OriState = Training_data[k][0]['rgbd_feature'] 
        assert len(obj_OriState) == num_obj
        assert len(hand_OriState) == num_hand

        z_t = Koopman.z(hand_OriState, obj_OriState)  # initial states in lifted space
        for t in range(len(Training_data[k]) - 1):
            hand_OriState = Training_data[k][t + 1]['handpos']
            obj_OriState = Training_data[k][t+1]['rgbd_feature']
            z_t_1 = Koopman.z(hand_OriState, obj_OriState) 
            A += np.outer(z_t_1, z_t)
            G += np.outer(z_t, z_t)
            z_t = z_t_1
    M = len(Training_data) * (len(Training_data[0]) - 1)
    A /= M
    G /= M
    koopman_operator = np.dot(A, scipy.linalg.pinv(G)) 
    cont_koopman_operator = koopman_operator
    np.save(koopman_save_path, cont_koopman_operator)
    print(f"Koopman matrix is saved! {koopman_save_path}\n")

def koopman_policy_control(env_name, controller, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, koopmanoption, resnet_model=None, device='cuda:1'):

    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_list_sim = []
    success_rate = str()
    horizon = 70
    print(f"len(Test_data) {len(Test_data)}")

    for k in tqdm(range(len(Test_data))):
        gif_frames = []
        num_handpos = len(Test_data[k][0]['handpos'])
        hand_OriState = Test_data[k][0]['handpos']

        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['door_body_pos'] = Test_data[k][0]['init']['door_body_pos'] 
        e.set_env_state(init_state_dict)

        rgb, depth = e.env.mj_render()
        rgb = (rgb.astype(np.uint8) - 128.0) / 128
        depth = depth[...,np.newaxis]
        rgbd = np.concatenate((rgb,depth),axis=2)
        rgbd = np.transpose(rgbd, (2, 0, 1))
        rgbd = rgbd[np.newaxis, ...]

        _, implict_objpos = resnet_model(torch.from_numpy(rgbd).float().to(device)) 
        obj_OriState = implict_objpos[0].cpu().detach().numpy()
        z_t = koopman_object.z(hand_OriState, obj_OriState)  
        for t in range(horizon - 1): 
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
            obj_pos_world = x_t_1_computed[56:59]
            hand_pos_desired = x_t_1_computed[:num_handpos]  
            current = z_t[:num_hand] #e.get_env_state()['qpos'][:28] # current state
            z_t = z_t_1_computed
            obs_dict = e.env.get_obs_dict(e.env.sim)
            
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()
            e.step(NN_output)  
            obs_dict = e.env.get_obs_dict(e.env.sim)
            current_hinge_pos = obs_dict['door_pos']
        if current_hinge_pos > 1.35:
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    return len(success_list_sim) / len(Test_data)

def koopman_policy_control_hammer(env, controller, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, koopmanoption, resnet_model=None, device='cuda:1'):
    print("Begin to compute the simulation errors!")
    #e = GymEnv(env_name)
    env.reset()
    init_state_dict = dict()
    success_list_sim = []
    success_rate = str()
    horizon = 51
    for k in tqdm(range(len(Test_data))): #len(Test_data)
        gif_frames = []
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['board_pos'] = Test_data[k][0]['init']['board_pos']  # fixed for each piece of demo data
        env.set_env_state(init_state_dict)

        num_handpos = len(Test_data[k][0]['handpos'])
        assert num_handpos == num_hand
        hand_OriState = Test_data[k][0]['handpos']

        rgb, depth = env.env.mj_render()
        rgb = (rgb.astype(np.uint8) - 128.0) / 128
        depth = depth[...,np.newaxis]
        rgbd = np.concatenate((rgb,depth),axis=2)
        rgbd = np.transpose(rgbd, (2, 0, 1))
        rgbd = rgbd[np.newaxis, ...]
        _, implict_objpos = resnet_model(torch.from_numpy(rgbd).float().to(device)) 
        implict_objpos = implict_objpos[0].cpu().detach().numpy()
        obj_OriState = implict_objpos
        
        if koopmanoption == 'Drafted': 
            z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        for t in range(horizon - 1): 
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
            hand_pos = x_t_1_computed[:num_handpos]
            hand_pos_desired = hand_pos
            
            current = z_t[:num_hand] #env.get_env_state()['qpos'][:26] #[:num_handpos] # current state
            z_t = z_t_1_computed
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            env.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            #gt_hand_pos = Test_data[k][t + 1]['handpos']
            obs_dict = env.env.get_obs_dict(env.env.sim)
            obj_obs = env.get_obs()
            current_nail_pos = obs_dict['target_pos']#obs_dict['obj_pos']#obj_obs[42:45]
            goal_nail_pos = obs_dict['goal_pos']#obj_obs[46:49] #Test_data[k][t]['nail_goal'] #obj_obs[46:49]
            dist = np.linalg.norm(current_nail_pos - goal_nail_pos)
        if dist < 0.01:
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    return len(success_list_sim) / len(Test_data)


def koopman_policy_control_reorientation_single_task(env_name, controller, koopman_object, koopman_matrix, Test_data, num_hand, num_obj, resnet_model, device):
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_koopman = []
    success_list_RL = []
    fall_list_sim = []
    fall_list_koopman = []
    fall_list_RL = []
    success_rate = str()
    horizon = 100
    # e.set_env_state(path['init_state_dict'])
    for k in tqdm(range(len(Test_data))): #tqdm()
        gif = []
        success_count_sim = np.zeros(horizon)
        num_handpos = len(Test_data[k][0]['handpos'])
        hand_OriState = Test_data[k][0]['handpos']
        init_state_dict['qpos'] = np.append(Test_data[k][0]['handpos'], np.zeros(6))
        init_state_dict['qvel'] = np.zeros(30)#np.append(Test_data[k][0]['handvel'], np.zeros(6))
        init_state_dict['desired_orien'] = euler2quat(Test_data[k][0]['pen_desired_orien'])
        e.set_env_state(Test_data[k][0]['init_state_dict']) #init_state_dict

        rgb, depth = e.env.mj_render()
        rgb = (rgb.astype(np.uint8) - 128.0) / 128
        depth = depth[...,np.newaxis]
        rgbd = np.concatenate((rgb,depth),axis=2)
        rgbd = np.transpose(rgbd, (2, 0, 1))
        rgbd = rgbd[np.newaxis, ...]
        rgbd = torch.from_numpy(rgbd).float().to(device)
        desired_ori = Test_data[k][0]['pen_desired_orien']
        desired_ori = desired_ori[np.newaxis, ...]
        desired_ori = torch.from_numpy(desired_ori).float().to(device)
        implict_objpos = resnet_model(rgbd, desired_ori) 
        obj_OriState = implict_objpos[0].cpu().detach().numpy()

        z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        for t in range(horizon):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
            # obj_ori_world = ori_transform_inverse(x_t_1_computed[27:30], e.get_obs()[36:39]) # [27:30] obj_orientation
            # obj_pos_world = x_t_1_computed[24:27]
            hand_pos_desired = x_t_1_computed[:num_handpos]  # desired hand joint state
            current = z_t[:num_hand] #e.get_env_state()['qpos'][:num_handpos] # current state
            z_t = z_t_1_computed
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation
            obs_dict = e.env.get_obs_dict(e.env.sim)
            obj_vel = obs_dict['obj_vel']       
            orien_similarity_sim = np.dot(obs_dict['obj_rot'], obs_dict['obj_des_rot'])
            success_count_sim[t] = 1 if (orien_similarity_sim > 0.90) else 0
        if np.abs(obs_dict['obj_err_pos'])[2] > 0.15:
            fall_list_sim.append(1)
        else:
            if sum(success_count_sim) > success_threshold and np.mean(np.abs(obj_vel)) < 1.: 
                success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f\n" % (len(success_list_sim) / len(Test_data))  



def koopman_policy_control_relocate(env_name, controller, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, resnet_model, device):
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_koopman = []
    success_list_demo = []
    success_rate = str()
    horizon = 100
    for k in tqdm(range(len(Test_data))): 
        success_count_sim = np.zeros(horizon)
        num_handpos = len(Test_data[k][0]['handpos'])
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['obj_pos'] = Test_data[k][0]['init']['obj_pos']
        init_state_dict['target_pos'] = Test_data[k][0]['init']['target_pos']
        e.set_env_state(init_state_dict)
        hand_OriState = Test_data[k][0]['handpos']

        rgb, depth = e.env.mj_render()
        rgb = (rgb.astype(np.uint8) - 128.0) / 128
        depth = depth[...,np.newaxis]
        rgbd = np.concatenate((rgb,depth),axis=2)
        rgbd = np.transpose(rgbd, (2, 0, 1))
        rgbd = rgbd[np.newaxis, ...]
        rgbd = torch.from_numpy(rgbd).float().to(device)
        desired_pos = Test_data[k][0]['init']['target_pos']
        desired_pos = desired_pos[np.newaxis, ...]
        desired_pos = torch.from_numpy(desired_pos).float().to(device)
        implict_objpos = resnet_model(rgbd, desired_pos) 
        obj_OriState = implict_objpos[0].cpu().detach().numpy()
        
        z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        for t in range(horizon):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
            hand_pos = x_t_1_computed[:num_handpos] 
            hand_pos_desired = hand_pos
            current = z_t[:num_hand] #e.get_env_state()['hand_qpos'] # current state z_t[:num_hand] #
            z_t = z_t_1_computed
            
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)  
            err = e.env.get_obs_dict(e.env.sim)['obj_tar_err']
            obj_pos = e.env.get_obs_dict(e.env.sim)['obj_pos']
            if np.linalg.norm(err) < 0.1:
                success_count_sim[t] = 1
        if sum(success_count_sim) > success_threshold:
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))) 