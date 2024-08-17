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

def BC_policy_control_door(env_name, controller, BC_agent, Test_data, num_hand, device, resnet_model, use_gt = True):
    print("Begin to compute the simulation errors!")
    Velocity = False
    koopmanoption = 'Drafted'
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    # for this task, we are only able to report the success rate for the simulation results
    # also, for this task, no failure cases.
    success_list_sim = []
    success_rate = str()
    horizon = 70
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        hand_OriState = Test_data[k][0]['handpos']
        #obj_OriState = Test_data[k][0]['rgbd_feature']
        
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['door_body_pos'] = Test_data[k][0]['init']['door_body_pos']  # fixed for each piece of demo data
        e.set_env_state(init_state_dict)
        
        if use_gt:
            #np.append(Test_data[k][0]['objpos'],
            obj_OriState = np.append(Test_data[k][0]['objpos'],Test_data[k][0]['handle_init'])
            #print(obj_OriState)
            #print(obj_OriState)
        else:
            rgb, depth = e.env.mj_render()
            rgb = (rgb.astype(np.uint8) - 128.0) / 128
            depth = depth[...,np.newaxis]
            rgbd = np.concatenate((rgb,depth),axis=2)
            rgbd = np.transpose(rgbd, (2, 0, 1))
            rgbd = rgbd[np.newaxis, ...]
            
            _, implict_objpos = resnet_model(torch.from_numpy(rgbd).float().to(device)) #Test_data[k][0]['rgbd_feature']
            obj_OriState = implict_objpos[0].detach().cpu().numpy()
        
        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)
        for t in range(horizon - 1):  # this loop is for system evolution, open loop control, no feedback
            #pdb.set_trace()
            z_t_1_computed = BC_agent(z_t)
            x_t_1_computed = z_t_1_computed.detach().numpy()
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            # hand_pos_desired = hand_pos + Test_data[k][0]['final_handpos']
            hand_pos_desired = hand_pos  # control frequency
            current = z_t[:num_handpos] #e.get_env_state()['qpos'][:28]
            z_t = z_t_1_computed
            # current state
            #pdb.set_trace()
            set_goal = hand_pos_desired.copy() # next state
            #pdb.set_trace()
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)  
            obs_dict = e.env.get_obs_dict(e.env.sim)
            current_hinge_pos = obs_dict['door_pos']#obj_obs[28:29] # door opening angle
        if (current_hinge_pos > 1.35):
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    print("Success rate (RL) = %f" % (1))


def BC_policy_control_tool(env_name, controller, BC_agent, Test_data, num_hand, device, resnet_model, use_gt = True):
    print("Begin to compute the simulation errors!")
    Velocity = False
    koopmanoption = 'Drafted'
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    # for this task, we are only able to report the success rate for the simulation results
    # also, for this task, no failure cases.
    success_list_sim = []
    success_rate = str()
    horizon = 70
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        hand_OriState = Test_data[k][0]['handpos']

        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['board_pos'] = Test_data[k][0]['init']['board_pos']  # fixed for each piece of demo data
        e.set_env_state(init_state_dict)
        
        if use_gt:
            obs_dict = e.env.get_obs_dict(e.env.sim) 
            obj_OriState = np.append(obs_dict['target_pos'], obs_dict['goal_pos'])
        else:
            rgb, depth = e.env.mj_render()
            rgb = (rgb.astype(np.uint8) - 128.0) / 128
            depth = depth[...,np.newaxis]
            rgbd = np.concatenate((rgb,depth),axis=2)
            rgbd = np.transpose(rgbd, (2, 0, 1))
            rgbd = rgbd[np.newaxis, ...]
            _, implict_objpos = resnet_model(torch.from_numpy(rgbd).float().to(device)) #Test_data[k][0]['rgbd_feature']
            obj_OriState = implict_objpos[0].detach().cpu().numpy()

        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)
        
        for t in range(horizon - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = BC_agent(z_t)
            #x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
            hand_pos = z_t_1_computed[:num_hand].detach().numpy() #x_t_1_computed[:num_handpos]  # desired hand joint states
            hand_pos_desired = hand_pos
            current = z_t[:num_handpos] # 
            z_t = z_t_1_computed
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            e.step(NN_output)
            obs_dict = e.env.get_obs_dict(e.env.sim)
            current_nail_pos = obs_dict['target_pos']
            goal_nail_pos = obs_dict['goal_pos']
            dist = np.linalg.norm(current_nail_pos - goal_nail_pos)
        if dist < 0.01:
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))


def BC_policy_control_relocate(env_name, controller, BC_agent, Test_data, num_hand, device, resnet_model, use_gt = True):
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    success_list_koopman = []
    success_list_demo = []
    success_rate = str()
    horizon = 100
    koopmanoption = 'Drafted'
    for k in tqdm(range(len(Test_data))): 
        success_count_sim = np.zeros(horizon)
        num_handpos = len(Test_data[k][0]['handpos'])
        
        #np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['obj_pos'] = Test_data[k][0]['init']['obj_pos']
        init_state_dict['target_pos'] = Test_data[k][0]['init']['target_pos']
        e.set_env_state(init_state_dict)
        hand_OriState = Test_data[k][0]['handpos']
        
        if use_gt:
            obj_OriState = np.append(e.env.get_obs_dict(e.env.sim)['obj_pos'], Test_data[k][0]['init']['target_pos'])
        else:
            rgb, depth = e.env.mj_render()
            rgb = (rgb.astype(np.uint8) - 128.0) / 128
            depth = depth[...,np.newaxis]
            rgbd = np.concatenate((rgb,depth),axis=2)
            rgbd = np.transpose(rgbd, (2, 0, 1))
            rgbd = rgbd[np.newaxis, ...]
            implict_objpos = resnet_model(torch.from_numpy(rgbd).float().to(device)) #Test_data[k][0]['rgbd_feature']
            obj_OriState = implict_objpos[0].detach().cpu().numpy()

        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)

        for t in range(horizon):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = BC_agent(z_t)
            #x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
            hand_pos = z_t_1_computed[:num_hand].detach().numpy() #x_t_1_computed[:num_handpos]  # desired hand joint states
            hand_pos_desired = hand_pos
            current = z_t[:num_hand] # e.get_env_state()['qpos'][:num_handpos] #
            z_t = z_t_1_computed
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            e.step(NN_output)
            err = e.env.get_obs_dict(e.env.sim)['obj_tar_err']
            if (t % 5 == 0 and k % 20 == 0):
                rgb, depth = e.env.mj_render()
                from PIL import Image 
                img_obs = Image.fromarray(rgb)
                img_obs.save(f"door_opening_{k}_{t}.png")
            if np.linalg.norm(err) < 0.1:
                success_count_sim[t] = 1
        if sum(success_count_sim) > success_threshold:
            print(f"success in {k}")
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))) 


def BC_policy_control_reorientation(env_name, controller, BC_agent, Test_data, num_hand, device, resnet_model, use_gt = True):
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_threshold = 10
    success_list_sim = []
    fall_list_sim = []
    success_list_koopman = []
    success_list_demo = []
    success_rate = str()
    horizon = 100
    koopmanoption = 'Drafted'
    for k in tqdm(range(len(Test_data))): 
        success_count_sim = np.zeros(horizon)
        num_handpos = len(Test_data[k][0]['handpos'])
        
        success_count_sim = np.zeros(horizon)
        num_handpos = len(Test_data[k][0]['handpos'])
        hand_OriState = Test_data[k][0]['handpos']
        init_state_dict['qpos'] = np.append(Test_data[k][0]['handpos'], np.zeros(6))
        init_state_dict['qvel'] = np.zeros(30)#np.append(Test_data[k][0]['handvel'], np.zeros(6))
        init_state_dict['desired_orien'] = euler2quat(Test_data[k][0]['pen_desired_orien'])
        e.set_env_state(Test_data[k][0]['init_state_dict']) #init_state_dict
        
        if use_gt:
            obj_OriState = np.append(e.env.get_obs_dict(e.env.sim)['obj_rot'], Test_data[k][0]['pen_desired_orien'])
        else:
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
            obj_OriState = implict_objpos[0].detach().cpu().numpy()

        if koopmanoption == 'Drafted': 
            tmp_input = np.append(hand_OriState, obj_OriState)
            z_t = torch.from_numpy(tmp_input)

        for t in range(horizon):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = BC_agent(z_t)
            #x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
            hand_pos = z_t_1_computed[:num_hand].detach().numpy() #x_t_1_computed[:num_handpos]  # desired hand joint states
            hand_pos_desired = hand_pos
            current = z_t[:num_hand] # e.get_env_state()['qpos'][:num_handpos] #
            z_t = z_t_1_computed
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            e.step(NN_output)
            obs_dict = e.env.get_obs_dict(e.env.sim)
            obj_vel = obs_dict['obj_vel']       
            orien_similarity_sim = np.dot(obs_dict['obj_rot'], obs_dict['obj_des_rot'])
            success_count_sim[t] = 1 if (orien_similarity_sim > 0.90) else 0
        if np.abs(obs_dict['obj_err_pos'])[2] > 0.15:
            fall_list_sim.append(1)
        else:
            if sum(success_count_sim) > success_threshold and np.mean(np.abs(obj_vel)) < 1.: 
                #print(k)
                success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))) 