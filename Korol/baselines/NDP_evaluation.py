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


def NDP_policy_control_door(env_name, controller, NDP_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, resnet_model, device):
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    # for this task, we are only able to report the success rate for the simulation results
    # also, for this task, no failure cases.
    success_list_sim = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        hand_OriState = Test_data[k][0]['handpos']
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['door_body_pos'] = Test_data[k][0]['init']['door_body_pos']  # fixed for each piece of demo data
        e.set_env_state(init_state_dict)

        if (resnet_model is not None):
            rgb, depth = e.env.mj_render()
            rgb = (rgb.astype(np.uint8) - 128.0) / 128
            depth = depth[...,np.newaxis]
            rgbd = np.concatenate((rgb,depth),axis=2)
            rgbd = np.transpose(rgbd, (2, 0, 1))
            rgbd = rgbd[np.newaxis, ...]
            
            _, implict_objpos = resnet_model(torch.from_numpy(rgbd).float().to(device))
            obj_OriState = implict_objpos[0].detach().cpu().numpy()
        else:
            obj_OriState = np.append(Test_data[k][0]['objpos'], Test_data[k][0]['handle_init'])
        z_t = torch.from_numpy(np.append(hand_OriState, obj_OriState))
        traj = NDP_agent.execute(z_t, z_t) # traj -> [T, dim]
        traj_len = len(traj)
        for t in range(traj_len - 1):  # this loop is for system evolution, open loop control, no feedback
            x_t_1_computed = traj[t+1]
            obj_pos_world = x_t_1_computed[28:31]  # handle pos
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            hand_pos_desired = hand_pos  # control frequency
            current = traj[t][:num_handpos] #e.get_env_state()['qpos'][:28] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()
            e.step(NN_output)     
            obj_obs = e.get_obs()
            current_hinge_pos = obj_obs[28:29] 
        if current_hinge_pos > 1.35:
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    return len(success_list_sim) / len(Test_data)

def NDP_policy_control_tool(env_name, controller, NDP_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, resnet_model, device):
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    # for this task, we are only able to report the success rate for the simulation results
    # also, for this task, no failure cases.
    success_list_sim = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        hand_OriState = Test_data[k][0]['handpos']
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['board_pos'] = Test_data[k][0]['init']['board_pos']  # fixed for each piece of demo data
        e.set_env_state(init_state_dict)

        if (resnet_model is not None):
            rgb, depth = e.env.mj_render()
            rgb = (rgb.astype(np.uint8) - 128.0) / 128
            depth = depth[...,np.newaxis]
            rgbd = np.concatenate((rgb,depth),axis=2)
            rgbd = np.transpose(rgbd, (2, 0, 1))
            rgbd = rgbd[np.newaxis, ...]
            
            _, implict_objpos = resnet_model(torch.from_numpy(rgbd).float().to(device))
            obj_OriState = implict_objpos[0].detach().cpu().numpy()
        else:
            obs_dict = e.env.get_obs_dict(e.env.sim) 
            obj_OriState = np.append(obs_dict['target_pos'], obs_dict['goal_pos'])
        z_t = torch.from_numpy(np.append(hand_OriState, obj_OriState))
        traj = NDP_agent.execute(z_t, z_t) # traj -> [T, dim]
        traj_len = len(traj)
        horizon = 51
        for t in range(horizon - 1):  # this loop is for system evolution, open loop control, no feedback
            x_t_1_computed = traj[t+1]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            hand_pos_desired = hand_pos  # control frequency
            current = traj[t][:num_handpos] #e.get_env_state()['qpos'][:26] # traj[t][:num_handpos] #
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()
            e.step(NN_output)     
            obs_dict = e.env.get_obs_dict(e.env.sim)
            current_nail_pos = obs_dict['target_pos']
            goal_nail_pos = obs_dict['goal_pos']
            dist = np.linalg.norm(current_nail_pos - goal_nail_pos)
            # if (t % 5 == 0 and k % 20 == 0):
            #     rgb, depth = e.env.mj_render()
            #     from PIL import Image 
            #     img_obs = Image.fromarray(rgb)
            #     img_obs.save(f"door_opening_{k}_{t}.png")
        if dist < 0.01:
            #print(k)
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    return len(success_list_sim) / len(Test_data)

def NDP_policy_control_relocate(env_name, controller, NDP_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, resnet_model, device):
    e = GymEnv(env_name)
    e.reset()
    success_threshold = 10
    init_state_dict = dict()
    # for this task, we are only able to report the success rate for the simulation results
    # also, for this task, no failure cases.
    success_list_sim = []
    success_rate = str()
    horizon = 70
    for k in tqdm(range(len(Test_data))):
        success_count_sim = np.zeros(horizon)
        num_handpos = len(Test_data[k][0]['handpos'])
        hand_OriState = Test_data[k][0]['handpos']
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['obj_pos'] = Test_data[k][0]['init']['obj_pos']
        init_state_dict['target_pos'] = Test_data[k][0]['init']['target_pos']
        e.set_env_state(init_state_dict)

        if (resnet_model is not None):
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
            obj_OriState = implict_objpos[0].detach().cpu().numpy()
        else:
            obs_dict = e.env.get_obs_dict(e.env.sim) 
            obj_OriState = np.append(obs_dict['obj_pos'], Test_data[k][0]['init']['target_pos'])

        z_t = torch.from_numpy(np.append(hand_OriState, obj_OriState))
        traj = NDP_agent.execute(z_t, z_t) # traj -> [T, dim]
        traj_len = len(traj)
        horizon = 70
        for t in range(horizon - 1):  # this loop is for system evolution, open loop control, no feedback
            x_t_1_computed = traj[t+1]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            hand_pos_desired = hand_pos  # control frequency
            current = traj[t][:num_handpos] #e.get_env_state()['qpos'][:26] # traj[t][:num_handpos] #
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()
            e.step(NN_output)     
            obs_dict = e.env.get_obs_dict(e.env.sim)
            err = e.env.get_obs_dict(e.env.sim)['obj_tar_err']
            obj_pos = e.env.get_obs_dict(e.env.sim)['obj_pos']
            # if (k % 10 == 0 and t % 5 ==0):
            #     print(f"desired_pos {desired_pos}, obj_pos {obj_pos}, err {err}")
            if np.linalg.norm(err) < 0.1:
                success_count_sim[t] = 1
        if sum(success_count_sim) > success_threshold:
            #print(f"success in {k}")
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    return len(success_list_sim) / len(Test_data)

def NDP_policy_control_reorientation(env_name, controller, NDP_agent, Test_data, Velocity, num_hand, num_obj, koopmanoption, resnet_model, device):
    e = GymEnv(env_name)
    e.reset()
    success_threshold = 10
    init_state_dict = dict()
    # for this task, we are only able to report the success rate for the simulation results
    # also, for this task, no failure cases.
    success_list_sim = []
    fall_list_sim = []
    success_rate = str()
    horizon = 70
    for k in tqdm(range(len(Test_data))):
        success_count_sim = np.zeros(horizon)
        num_handpos = len(Test_data[k][0]['handpos'])
        hand_OriState = Test_data[k][0]['handpos']
        init_state_dict['qpos'] = np.append(Test_data[k][0]['handpos'], np.zeros(6))
        init_state_dict['qvel'] = np.zeros(30)#np.append(Test_data[k][0]['handvel'], np.zeros(6))
        init_state_dict['desired_orien'] = euler2quat(Test_data[k][0]['pen_desired_orien'])
        e.set_env_state(Test_data[k][0]['init_state_dict']) #init_state_dict

        if (resnet_model is not None):
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
        else:
            obj_OriState = np.append(e.env.get_obs_dict(e.env.sim)['obj_rot'], Test_data[k][0]['pen_desired_orien'])


        z_t = torch.from_numpy(np.append(hand_OriState, obj_OriState))
        traj = NDP_agent.execute(z_t, z_t) # traj -> [T, dim]
        traj_len = len(traj)
        for t in range(horizon - 1):  # this loop is for system evolution, open loop control, no feedback
            x_t_1_computed = traj[t+1]
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint state
            hand_pos_desired = hand_pos  # control frequency
            current = traj[t][:num_handpos] #e.get_env_state()['qpos'][:26] # traj[t][:num_handpos] #
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
    success_rate += "Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))
    return len(success_list_sim) / len(Test_data)
