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
import gc
import os
import scipy
from torch.utils.data import Dataset, DataLoader
import pickle 
from PIL import Image
import collections

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


def Diffusion_policy_control_door(env_name, controller, ema_nets, Test_data, normalize_stats, device, obs_horizon, pred_horizon, action_horizon, num_diffusion_iters, noise_scheduler, use_gt = True):
    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_list_sim = []
    horizon = 70
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        num_obj = 6
        action_dim = num_handpos + num_obj
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['door_body_pos'] = Test_data[k][0]['init']['door_body_pos']  # fixed for each piece of demo data
        e.set_env_state(init_state_dict)
        
        robot_state = e.get_env_state()['qpos'][:num_handpos]
        obj_OriState = np.append(Test_data[k][0]['objpos'],Test_data[k][0]['handle_init'])
        
        obs = {'robot_state':np.append(robot_state, obj_OriState)}
        obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
        
        B = 1
        t = 0
        while t < horizon - 1:  # this loop is for system evolution, 

            agent_poses = np.stack([x['robot_state'] for x in obs_deque])

            # normalize observation
            # pdb.set_trace()
            nagent_poses = normalize_data(agent_poses, stats=normalize_stats['robot_state'])
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
            # (2,2)
            # infer action
            with torch.no_grad():
                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = nagent_poses.unsqueeze(0).flatten(start_dim=1)

                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (B, pred_horizon, action_dim), device=device)
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = ema_nets(
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=normalize_stats['action'])

            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end,:]
            for i in range(len(action)):
                # stepping env
                current = e.get_env_state()['qpos'][:num_handpos]
                set_goal = action[i][:num_handpos]
                NN_input = torch.from_numpy(np.append(current, set_goal))
                NN_output = controller(NN_input).detach().numpy()
                e.step(NN_output)  
                robot_state = e.get_env_state()['qpos'][:num_handpos]
                obs_dict = e.env.get_obs_dict(e.env.sim)
                obj_OriState = action[i][num_handpos:]
                #np.append(obs_dict['handle_pos'],Test_data[k][0]['handle_init']) 
                #action[i][num_handpos:]
                #np.append(obs_dict['handle_pos'],Test_data[k][0]['handle_init']) 
                #print(f"obj_OriState {obj_OriState}")
                obs = {'robot_state':np.append(robot_state, obj_OriState)}
                obs_deque.append(obs)
                t += 1

            obs_dict = e.env.get_obs_dict(e.env.sim)
            current_hinge_pos = obs_dict['door_pos']#obj_obs[28:29] # door opening angle
            if (current_hinge_pos > 1.35):
                success_list_sim.append(1)
                break
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))


def Diffusion_policy_control_tool(env_name, controller, ema_nets, Test_data, normalize_stats, device, obs_horizon, pred_horizon, action_horizon, num_diffusion_iters, noise_scheduler, use_gt = True):
    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_list_sim = []
    horizon = 70
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        num_obj = 6
        action_dim = num_handpos + num_obj
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['board_pos'] = Test_data[k][0]['init']['board_pos']  # fixed for each piece of demo data
        e.set_env_state(init_state_dict)
        
        robot_state = e.get_env_state()['qpos'][:num_handpos]
        obs_dict = e.env.get_obs_dict(e.env.sim) 
        obj_OriState = np.append(obs_dict['target_pos'], obs_dict['goal_pos'])
        
        obs = {'robot_state':np.append(robot_state, obj_OriState)}
        obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)

        B = 1
        t = 0
        while t < horizon - 1:  # this loop is for system evolution, 

            agent_poses = np.stack([x['robot_state'] for x in obs_deque])

            # normalize observation
            # pdb.set_trace()
            nagent_poses = normalize_data(agent_poses, stats=normalize_stats['robot_state'])
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
            # (2,2)
            # infer action
            with torch.no_grad():
                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = nagent_poses.unsqueeze(0).flatten(start_dim=1)

                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (B, pred_horizon, action_dim), device=device)
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = ema_nets(
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=normalize_stats['action'])

            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end,:]
            for i in range(len(action)):
                # stepping env
                current = e.get_env_state()['qpos'][:num_handpos]
                set_goal = action[i][:num_handpos]
                NN_input = torch.from_numpy(np.append(current, set_goal))
                NN_output = controller(NN_input).detach().numpy()
                e.step(NN_output)  
                robot_state = e.get_env_state()['qpos'][:num_handpos]
                obs_dict = e.env.get_obs_dict(e.env.sim)
                obj_OriState = action[i][num_handpos:] 
                #action[i][num_handpos:]
                #np.append(obs_dict['handle_pos'],Test_data[k][0]['handle_init']) 
                #print(f"obj_OriState {obj_OriState}")
                obs = {'robot_state':np.append(robot_state, obj_OriState)}
                obs_deque.append(obs)
                t += 1

            obs_dict = e.env.get_obs_dict(e.env.sim)
            current_nail_pos = obs_dict['target_pos']
            goal_nail_pos = obs_dict['goal_pos']
            dist = np.linalg.norm(current_nail_pos - goal_nail_pos)
            if dist < 0.01:
                success_list_sim.append(1)
                break

    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))


def Diffusion_policy_control_relocate(env_name, controller, ema_nets, Test_data, normalize_stats, device, obs_horizon, pred_horizon, action_horizon, num_diffusion_iters, noise_scheduler, use_gt = True):
    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_list_sim = []
    horizon = 100
    success_threshold = 10
    for k in tqdm(range(len(Test_data))):
        success_count_sim = np.zeros(horizon+action_horizon)
        num_handpos = len(Test_data[k][0]['handpos'])
        num_obj = 6
        action_dim = num_handpos + num_obj
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['obj_pos'] = Test_data[k][0]['init']['obj_pos']
        init_state_dict['target_pos'] = Test_data[k][0]['init']['target_pos']
        e.set_env_state(init_state_dict)

        robot_state = e.get_env_state()['qpos'][:num_handpos]
        obs_dict = e.env.get_obs_dict(e.env.sim) 
        obj_OriState = np.append(obs_dict['obj_pos'], Test_data[k][0]['init']['target_pos'])
        
        obs = {'robot_state':np.append(robot_state, obj_OriState)}
        obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)

        B = 1
        t = 0
        while t < horizon - 1:  # this loop is for system evolution, 
            agent_poses = np.stack([x['robot_state'] for x in obs_deque])
            # normalize observation
            # pdb.set_trace()
            nagent_poses = normalize_data(agent_poses, stats=normalize_stats['robot_state'])
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
            # (2,2)
            # infer action
            with torch.no_grad():
                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = nagent_poses.unsqueeze(0).flatten(start_dim=1)

                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (B, pred_horizon, action_dim), device=device)
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = ema_nets(
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
                        # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=normalize_stats['action'])

            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end,:]
            for i in range(len(action)):
                # stepping env
                current = e.get_env_state()['qpos'][:num_handpos]
                set_goal = action[i][:num_handpos]
                NN_input = torch.from_numpy(np.append(current, set_goal))
                NN_output = controller(NN_input).detach().numpy()
                e.step(NN_output)  
                robot_state = e.get_env_state()['qpos'][:num_handpos]
                obs_dict = e.env.get_obs_dict(e.env.sim)
                obj_OriState = action[i][num_handpos:] 
                #action[i][num_handpos:]
                #np.append(obs_dict['handle_pos'],Test_data[k][0]['handle_init']) 
                #print(f"obj_OriState {obj_OriState}")
                obs = {'robot_state':np.append(robot_state, obj_OriState)}
                obs_deque.append(obs)
                t += 1
                err = e.env.get_obs_dict(e.env.sim)['obj_tar_err']
                if np.linalg.norm(err) < 0.1:
                    success_count_sim[t] = 1
        if sum(success_count_sim) > success_threshold:
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))


def Diffusion_policy_control_reorientation(env_name, controller, ema_nets, Test_data, normalize_stats, device, obs_horizon, pred_horizon, action_horizon, num_diffusion_iters, noise_scheduler, use_gt = True):
    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    e.reset()
    init_state_dict = dict()
    success_list_sim = []
    horizon = 100
    success_threshold = 10
    for k in tqdm(range(len(Test_data))):
        success_count_sim = np.zeros(horizon+action_horizon)
        num_handpos = len(Test_data[k][0]['handpos'])
        num_obj = 6
        action_dim = num_handpos + num_obj

        init_state_dict['qpos'] = np.append(Test_data[k][0]['handpos'], np.zeros(6))
        init_state_dict['qvel'] = np.zeros(30) #np.append(Test_data[k][0]['handvel'], np.zeros(6))
        init_state_dict['desired_orien'] = euler2quat(Test_data[k][0]['pen_desired_orien'])
        e.set_env_state(Test_data[k][0]['init_state_dict']) #init_state_dict

        robot_state = e.get_env_state()['qpos'][:num_handpos]
        obs_dict = e.env.get_obs_dict(e.env.sim) 
        obj_OriState = np.append(obs_dict['obj_rot'], Test_data[k][0]['pen_desired_orien'])
        
        obs = {'robot_state':np.append(robot_state, obj_OriState)}
        obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)

        B = 1
        t = 0
        while t < horizon - 1:  # this loop is for system evolution, 

            agent_poses = np.stack([x['robot_state'] for x in obs_deque])
            # normalize observation
            # pdb.set_trace()
            nagent_poses = normalize_data(agent_poses, stats=normalize_stats['robot_state'])
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
            # (2,2)
            # infer action
            with torch.no_grad():
                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = nagent_poses.unsqueeze(0).flatten(start_dim=1)

                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (B, pred_horizon, action_dim), device=device)
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = ema_nets(
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
                        # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=normalize_stats['action'])

            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end,:]
            for i in range(len(action)):
                # stepping env
                current = e.get_env_state()['qpos'][:num_handpos]
                set_goal = action[i]
                NN_input = torch.from_numpy(np.append(current, set_goal[:num_handpos]))
                NN_output = controller(NN_input).detach().numpy()
                e.step(NN_output)  
                robot_state = e.get_env_state()['qpos'][:num_handpos]
                obs_dict = e.env.get_obs_dict(e.env.sim)
                obj_OriState = action[i][num_handpos:] 
                #action[i][num_handpos:]
                #np.append(obs_dict['handle_pos'],Test_data[k][0]['handle_init']) 
                #print(f"obj_OriState {obj_OriState}")
                obs = {'robot_state':np.append(robot_state, obj_OriState)}
                obs_deque.append(obs)
                t += 1
                obj_vel = obs_dict['obj_vel']       
                orien_similarity_sim = np.dot(obs_dict['obj_rot'], obs_dict['obj_des_rot'])
                success_count_sim[t] = 1 if (orien_similarity_sim > 0.90) else 0
        if np.abs(obs_dict['obj_err_pos'])[2] > 0.15:
            pass
        else:
            if sum(success_count_sim) > success_threshold and np.mean(np.abs(obj_vel)) < 1.: 
                #print(k)
                success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))

# def Diffusion_policy_control_reorientation(env_name, controller, Diffusion_agent, Test_data, num_hand, device, resnet_model, use_gt = True):
#     e = GymEnv(env_name)
#     e.reset()
#     init_state_dict = dict()
#     success_threshold = 10
#     success_list_sim = []
#     fall_list_sim = []

#     horizon = 100
#     koopmanoption = 'Drafted'
#     for k in tqdm(range(len(Test_data))): 
#         success_count_sim = np.zeros(horizon)
#         num_handpos = len(Test_data[k][0]['handpos'])
        
#         success_count_sim = np.zeros(horizon)
#         num_handpos = len(Test_data[k][0]['handpos'])
#         hand_OriState = Test_data[k][0]['handpos']
#         init_state_dict['qpos'] = np.append(Test_data[k][0]['handpos'], np.zeros(6))
#         init_state_dict['qvel'] = np.zeros(30)#np.append(Test_data[k][0]['handvel'], np.zeros(6))
#         init_state_dict['desired_orien'] = euler2quat(Test_data[k][0]['pen_desired_orien'])
#         e.set_env_state(Test_data[k][0]['init_state_dict']) #init_state_dict
        
#         if use_gt:
#             obj_OriState = np.append(e.env.get_obs_dict(e.env.sim)['obj_rot'], Test_data[k][0]['pen_desired_orien'])
#         else:
#             rgb, depth = e.env.mj_render()
#             rgb = (rgb.astype(np.uint8) - 128.0) / 128
#             depth = depth[...,np.newaxis]
#             rgbd = np.concatenate((rgb,depth),axis=2)
#             rgbd = np.transpose(rgbd, (2, 0, 1))
#             rgbd = rgbd[np.newaxis, ...]
#             rgbd = torch.from_numpy(rgbd).float().to(device)

#             desired_ori = Test_data[k][0]['pen_desired_orien']
#             desired_ori = desired_ori[np.newaxis, ...]
#             desired_ori = torch.from_numpy(desired_ori).float().to(device)
            
#             implict_objpos = resnet_model(rgbd, desired_ori)
#             obj_OriState = implict_objpos[0].detach().cpu().numpy()

#         if koopmanoption == 'Drafted': 
#             tmp_input = np.append(hand_OriState, obj_OriState)
#             z_t = torch.from_numpy(tmp_input)

#         for t in range(horizon):  # this loop is for system evolution, open loop control, no feedback
#             z_t_1_computed = Diffusion_agent(z_t)
#             #x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
#             hand_pos = z_t_1_computed[:num_hand].detach().numpy() #x_t_1_computed[:num_handpos]  # desired hand joint states
#             hand_pos_desired = hand_pos
#             current = z_t[:num_hand] # e.get_env_state()['qpos'][:num_handpos] #
#             z_t = z_t_1_computed
#             set_goal = hand_pos_desired.copy() # next state
#             NN_input = torch.from_numpy(np.append(current, set_goal))
#             NN_output = controller(NN_input).detach().numpy()   
#             e.step(NN_output)
#             obs_dict = e.env.get_obs_dict(e.env.sim)
#             obj_vel = obs_dict['obj_vel']       
#             orien_similarity_sim = np.dot(obs_dict['obj_rot'], obs_dict['obj_des_rot'])
#             success_count_sim[t] = 1 if (orien_similarity_sim > 0.90) else 0
#         if np.abs(obs_dict['obj_err_pos'])[2] > 0.15:
#             fall_list_sim.append(1)
#         else:
#             if sum(success_count_sim) > success_threshold and np.mean(np.abs(obj_vel)) < 1.: 
#                 #print(k)
#                 success_list_sim.append(1)
#     print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))) 