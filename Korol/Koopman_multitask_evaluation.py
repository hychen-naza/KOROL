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

def get_env_rgbd(e):
    img_obs, depth_obs = e.env.mj_render()
    depth_obs = depth_obs[...,np.newaxis]
    img_obs = (img_obs.astype(np.uint8) - 128.0) / 128
    rgbd = np.concatenate((img_obs, depth_obs),axis=2)
    rgbd = np.transpose(rgbd, (2, 0, 1))
    rgbd = rgbd[np.newaxis, ...]
    return rgbd

def koopman_door_multi(env_name, controller, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, koopmanoption, resnet_model=None, device='cuda:1'):

    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_list_sim = []
    success_rate = str()
    horizon = 70
    print(f"len(Test_data) {len(Test_data)}")
    assert num_hand == 30
    for k in tqdm(range(len(Test_data))):
        # pad_handpos = np.zeros(30)
        # pad_handpos[2:] = Test_data[k][0]['handpos']
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        hand_OriState = init_state_dict['qpos'] #pad_handpos
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['door_body_pos'] = Test_data[k][0]['init']['door_body_pos'] 
        e.set_env_state(init_state_dict)

        rgb, depth = e.env.mj_render()
        rgb = (rgb.astype(np.uint8) - 128.0) / 128
        depth = depth[...,np.newaxis]
        rgbd = np.concatenate((rgb,depth),axis=2)
        rgbd = np.transpose(rgbd, (2, 0, 1))
        rgbd = rgbd[np.newaxis, ...]
        desired_posori = torch.from_numpy(np.array([0.,0,0])).unsqueeze(0).float().to(device)
        implict_objpos = resnet_model(torch.from_numpy(rgbd).float().to(device), desired_posori) 
        obj_OriState = implict_objpos[0].cpu().detach().numpy()
        z_t = koopman_object.z(hand_OriState, obj_OriState)  
        for t in range(horizon - 1): 
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])

            hand_pos_desired = x_t_1_computed[:num_hand]  
            current = z_t[:num_hand] 
            z_t = z_t_1_computed
            obs_dict = e.env.get_obs_dict(e.env.sim)
            set_goal = hand_pos_desired.copy() # next state
            #pdb.set_trace()
            # To unpad the robot state
            NN_input = torch.from_numpy(np.append(current[2:], set_goal[2:]))
            NN_output = controller(NN_input).detach().numpy()
            e.step(NN_output)  
            obs_dict = e.env.get_obs_dict(e.env.sim)
            current_hinge_pos = obs_dict['door_pos']#obj_obs[28:29] # door opening angle
        if current_hinge_pos > 1.35:
            success_list_sim.append(1)
    print("Door Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    return len(success_list_sim) / len(Test_data)

def koopman_hammer_multi(env_name, controller, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, koopmanoption, resnet_model=None, device='cuda:1'):
    print("Begin to compute the simulation errors!")
    #e = GymEnv(env_name)
    env = GymEnv(env_name)
    env.reset()
    init_state_dict = dict()
    success_list_sim = []
    success_rate = str()
    horizon = 71
    for k in tqdm(range(len(Test_data))): #len(Test_data)
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['board_pos'] = Test_data[k][0]['init']['board_pos']  # fixed for each piece of demo data
        env.set_env_state(init_state_dict)

        num_handpos = 30 
        assert num_handpos == num_hand
        # PAD robot state
        # pad_handpos = np.zeros(30)
        # pad_handpos[3:5] = Test_data[k][0]['handpos'][:2]
        # pad_handpos[6:] = Test_data[k][0]['handpos'][2:]
        hand_OriState = init_state_dict['qpos']
        rgb, depth = env.env.mj_render()
        rgb = (rgb.astype(np.uint8) - 128.0) / 128
        depth = depth[...,np.newaxis]
        rgbd = np.concatenate((rgb,depth),axis=2)
        rgbd = np.transpose(rgbd, (2, 0, 1))
        rgbd = rgbd[np.newaxis, ...]
        
        desired_posori = torch.from_numpy(np.array([0.,0,0])).unsqueeze(0).float().to(device)
        implict_objpos = resnet_model(torch.from_numpy(rgbd).float().to(device), desired_posori) 
        implict_objpos = implict_objpos[0].cpu().detach().numpy()
        # Test_data[k][0]['rgbd_feature'] #
        obj_OriState = implict_objpos
        
        if koopmanoption == 'Drafted': 
            z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        for t in range(horizon - 1): 
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
            #hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            hand_pos = x_t_1_computed[:num_handpos]
            # NAZA
            # hand_pos = np.concatenate((hand_pos[3:5], hand_pos[6:]))
            # hand_pos_desired = hand_pos + Test_data[k][0]['final_handpos']
            hand_pos_desired = hand_pos
            
            current = z_t[:num_hand] #env.get_env_state()['qpos'][:26] #[:num_handpos] # current state
            z_t = z_t_1_computed
            set_goal = hand_pos_desired.copy() # next state
            # to unpad the robot state 
            current = np.append(current[3:5], current[6:])
            set_goal = np.append(set_goal[3:5], set_goal[6:])
            NN_input = torch.from_numpy(np.append(current, set_goal))
            # NN_input = torch.from_numpy(np.append(current[:-6], set_goal[:-6]))
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
    print("Hammer Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    return len(success_list_sim) / len(Test_data)

def koopman_relocate_multi(env_name, controller, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, koopmanoption, resnet_model=None, device='cuda:1'):
    print("Begin to compute the simulation errors!")
    #e = GymEnv(env_name)
    env = GymEnv(env_name)
    env.reset()
    init_state_dict = dict()
    success_list_sim = []
    success_rate = str()
    horizon = 100
    success_threshold = 10
    for k in tqdm(range(len(Test_data))): #len(Test_data)
        success_count_sim = np.zeros(horizon)
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['obj_pos'] = Test_data[k][0]['init']['obj_pos']
        init_state_dict['target_pos'] = Test_data[k][0]['init']['target_pos']
        env.set_env_state(init_state_dict)
        hand_OriState = Test_data[k][0]['handpos']

        num_handpos = 30 #len(Test_data[k][0]['handpos'])

        rgb, depth = env.env.mj_render()
        rgb = (rgb.astype(np.uint8) - 128.0) / 128
        depth = depth[...,np.newaxis]
        rgbd = np.concatenate((rgb,depth),axis=2)
        rgbd = np.transpose(rgbd, (2, 0, 1))
        rgbd = rgbd[np.newaxis, ...]
        desired_pos = Test_data[k][0]['init']['target_pos']
        desired_pos = desired_pos[np.newaxis, ...]
        desired_pos = torch.from_numpy(desired_pos).float().to(device)
        implict_objpos = resnet_model(torch.from_numpy(rgbd).float().to(device), desired_pos) 
        implict_objpos = implict_objpos[0].cpu().detach().numpy()
        obj_OriState = implict_objpos
        
        if koopmanoption == 'Drafted': 
            z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        for t in range(horizon - 1): 
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
            #hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            hand_pos = x_t_1_computed[:num_handpos]
            # NAZA
            # hand_pos = np.concatenate((hand_pos[3:5], hand_pos[6:]))
            # hand_pos_desired = hand_pos + Test_data[k][0]['final_handpos']
            hand_pos_desired = hand_pos
            
            current = z_t[:num_hand] #env.get_env_state()['qpos'][:26] #[:num_handpos] # current state
            z_t = z_t_1_computed
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            env.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            err = env.env.get_obs_dict(env.env.sim)['obj_tar_err']
            obj_pos = env.env.get_obs_dict(env.env.sim)['obj_pos']
            if np.linalg.norm(err) < 0.1:
                success_count_sim[t] = 1
        if sum(success_count_sim) > success_threshold:
            success_list_sim.append(1)
    print("Relocation Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    return len(success_list_sim) / len(Test_data)


def koopman_reori_multi(env_name, controller, koopman_object, koopman_matrix, Test_data, Velocity, num_hand, num_obj, koopmanoption, resnet_model=None, device='cuda:1'):
    print("Begin to compute the simulation errors!")
    #e = GymEnv(env_name)
    env = GymEnv(env_name)
    env.reset()
    init_state_dict = dict()
    success_list_sim = []
    fall_list_sim = []
    success_rate = str()
    horizon = 100
    success_threshold = 10
    for k in tqdm(range(len(Test_data))): #len(Test_data)
        success_count_sim = np.zeros(horizon)
        init_state_dict['qpos'] = np.append(Test_data[k][0]['handpos'], np.zeros(6))
        init_state_dict['qvel'] = np.zeros(30)#np.append(Test_data[k][0]['handvel'], np.zeros(6))
        init_state_dict['desired_orien'] = euler2quat(Test_data[k][0]['pen_desired_orien'])
        env.set_env_state(Test_data[k][0]['init_state_dict']) #init_state_dict
 
        pad_handpos = np.zeros(30)
        pad_handpos[6:] = Test_data[k][0]['handpos']
        hand_OriState = pad_handpos

        num_handpos = 30 #len(Test_data[k][0]['handpos'])

        rgb, depth = env.env.mj_render()
        rgb = (rgb.astype(np.uint8) - 128.0) / 128
        depth = depth[...,np.newaxis]
        rgbd = np.concatenate((rgb,depth),axis=2)
        rgbd = np.transpose(rgbd, (2, 0, 1))
        rgbd = rgbd[np.newaxis, ...]
        desired_ori = Test_data[k][0]['pen_desired_orien']
        desired_ori = desired_ori[np.newaxis, ...]
        desired_ori = torch.from_numpy(desired_ori).float().to(device)
        implict_objpos = resnet_model(torch.from_numpy(rgbd).float().to(device), desired_ori) 
        implict_objpos = implict_objpos[0].cpu().detach().numpy()
        obj_OriState = implict_objpos
        
        if koopmanoption == 'Drafted': 
            z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        for t in range(horizon - 1): 
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
            #hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            hand_pos = x_t_1_computed[:num_handpos]
            # NAZA
            # hand_pos = np.concatenate((hand_pos[3:5], hand_pos[6:]))
            # hand_pos_desired = hand_pos + Test_data[k][0]['final_handpos']
            hand_pos_desired = hand_pos
            
            current = z_t[:num_hand] #env.get_env_state()['qpos'][:26] #[:num_handpos] # current state
            z_t = z_t_1_computed
            set_goal = hand_pos_desired.copy() # next state
            current = current[6:]
            set_goal = set_goal[6:]
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            env.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            obs_dict = env.env.get_obs_dict(env.env.sim)
            obj_vel = obs_dict['obj_vel']       
            orien_similarity_sim = np.dot(obs_dict['obj_rot'], obs_dict['obj_des_rot'])
            success_count_sim[t] = 1 if (orien_similarity_sim > 0.90) else 0
        if np.abs(obs_dict['obj_err_pos'])[2] > 0.15:
            fall_list_sim.append(1)
        else:
            if sum(success_count_sim) > success_threshold and np.mean(np.abs(obj_vel)) < 1.: 
                success_list_sim.append(1)
    print("Reori Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    return len(success_list_sim) / len(Test_data)
