import pickle 
import numpy as np
from tqdm import tqdm
from utils.gym_env import GymEnv
import pdb 
from utils.quatmath import quat2euler, euler2quat

def door_demo_playback(env_name, demo_paths, feature_paths, num_demo, multi_task = False):
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
    success_list_sim = []
    min_action_values = [100] * len(demo_paths[0]['actions'][0])
    max_action_values = [-100] * len(demo_paths[0]['actions'][0])
    for i in tqdm(sample_index):
        path = demo_paths[i]
        path_data = []
        # if multi_task:
        #     path['init_state_dict']['qpos'] = np.zeros(32)
        #     path['init_state_dict']['qvel'] = np.zeros(32)
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
            if (multi_task):
                pad_handpos = np.zeros(30)
                pad_handpos[2:] = handpos
                #tmp['handpos'] = e.env.get_full_obs_visualization()[:30]
                tmp['handpos'] = pad_handpos
            else:
                tmp['handpos'] = handpos
            
            tmp['handvel'] = obs_visual[30:58]
            tmp['objpos'] = obs[32:35]
            
            tmp['objvel'] = obs_visual[58:59]
            tmp['handle_init'] = path['init_state_dict']['door_body_pos'] 
            tmp['observation'] = obs[35:38]
            tmp['action'] = actions[tt]
            min_action_values = np.minimum(min_action_values, actions[tt])
            max_action_values = np.maximum(max_action_values, actions[tt])
            dict_value = feature_data[count].values()
            predict = list(dict_value)[0]
            tmp['rgbd_feature'] = predict
            tmp['count'] = count
            count += 1
            path_data.append(tmp)
            # if (tt == 0 and i < 10):
            #     print(f"i {i}, path['init_state_dict']['door_body_pos']  {path['init_state_dict']['door_body_pos'] }")
        Training_data.append(path_data)
    #     obs_dict = e.env.get_obs_dict(e.env.sim)
    #     current_hinge_pos = obs_dict['door_pos']#obj_obs[28:29] # door opening angle
    #     if (current_hinge_pos > 1.35):
    #         print(i)
    #         success_list_sim.append(1)
    # #print(f"len( {len(success_list_sim)})")
    # pdb.set_trace()
    # with open('multi_task_demo/door_demo.pickle', 'wb') as handle:
    #     pickle.dump(Training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return Training_data, min_action_values, max_action_values


def hammer_demo_playback(env_name, demo_paths, feature_paths, num_demo, multi_task = False):
    with open(feature_paths, 'rb') as handle:
        feature_data = pickle.load(handle)
    e = GymEnv(env_name)
    e.reset()
    Training_data = []
    print("Begin loading demo data!")
    print(num_demo)
    sample_index = np.arange(num_demo)
    # sample_index = range(num_demo)
    success_list_sim = []
    count = 0
    for i in tqdm(sample_index):
        path = demo_paths[i]
        path_data = []
        # if multi_task:
        #     path['init_state_dict']['qpos'] = np.zeros(37)
        #     path['init_state_dict']['qvel'] = np.zeros(37)
        e.set_env_state(path['init_state_dict'])
        actions = path['actions']
        observations = path['observations'] 
        observations_visualize = path['observations_visualization']
        handVelocity = path['handVelocity'] 
        Selected_timesteps = len(actions)
        for t in range(Selected_timesteps): 
            tmp = dict()
            if t == 0:
                tmp['init'] = path['init_state_dict']
            obs = observations[t]
            obs_visual = observations_visualize[t]
            allvel = handVelocity[t] 
            handpos = obs[:26]
            if (multi_task):
                #tmp['handpos'] = obs[:30]
                pad_handpos = np.zeros(30)
                pad_handpos[3:5] = handpos[:2]
                pad_handpos[6:] = handpos[2:]
                #observations[t] = e.env.get_full_obs_visualization()[:30]
                tmp['handpos'] = pad_handpos #e.env.get_full_obs_visualization()[:30]
            else:
                tmp['handpos'] = handpos
            tmp['handvel'] = allvel[:26]
            tmp['objpos'] = obs[42:45]  
            tmp['toolpos'] = obs[49:52]
            tmp['objorient'] = obs[39:42] 
            tmp['objvel'] = obs[27:33] 
            tmp['nail_goal'] = obs[46:49] 
            tmp['observation'] = obs[49:52]
            action = actions[t]
            #e.step(actions[t])
            tmp['action'] = action #pad_action
            dict_value = feature_data[count].values()
            predict = list(dict_value)[0]
            tmp['rgbd_feature'] = predict
            tmp['count'] = count
            count += 1
            
            path_data.append(tmp)
        Training_data.append(path_data)
        # obs_dict = e.env.get_obs_dict(e.env.sim)
        # current_nail_pos = obs_dict['target_pos']
        # goal_nail_pos = obs_dict['goal_pos']
        # dist = np.linalg.norm(current_nail_pos - goal_nail_pos)
        # if dist < 0.01:
        #     print(i)
        #     success_list_sim.append(1)
    # print(len(success_list_sim))
    # pdb.set_trace()
    # with open('multi_task_demo/door_demo.pickle', 'wb') as handle:
    #     pickle.dump(Training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return Training_data


def relocate_demo_playback(env_name, demo_paths, feature_path, num_demo, multi_task=False): 
    e = GymEnv(env_name)
    e.reset()
    Training_data = []
    feature_data = []
    print("Begin loading demo data!")
    print(num_demo)
    sample_index = np.arange(num_demo)
    with open(feature_path, 'rb') as handle:
        feature_data = pickle.load(handle)
    count = 0
    for i in tqdm(sample_index):
        #pdb.set_trace()
        path = demo_paths[i]
        path_data = []
        e.set_env_state(path['init_state_dict'])
        actions = path['actions']
        observations = path['observations'] 
        observations_visualize = path['observations_visualization']
        handVelocity = path['handVelocity'] 
        for t in range(len(actions)):
            tmp = dict()
            if t == 0:
                tmp['init'] = path['init_state_dict']
            obs = observations[t]
            obs_visual = observations_visualize[t]
            handpos = obs[:30]  
            tmp['handpos'] = handpos
            allvel = handVelocity[t]
            tmp['handvel'] = allvel[:30]
            objpos = obs[39:42] 
            tmp['desired_pos'] = obs[45:48] 
            tmp['objpos'] = objpos # - obs[45:48] 
            tmp['objorient'] = obs_visual[33:36]
            tmp['obj_pos_visual'] = obs_visual[30:33]
            tmp['objvel'] = allvel[30:]
            tmp['observation'] = obs[36:39]  
            tmp['action'] = actions[t]
            dict_value = feature_data[count].values()
            feature = list(dict_value)[0]
            tmp['rgbd_feature'] = feature
            tmp['count'] = count
            count += 1
            path_data.append(tmp)
        Training_data.append(path_data)
    return Training_data

def reorientation_demo_playback(env_name, demo_paths, feature_path, num_demo, multi_task=False):
    e = GymEnv(env_name)
    e.reset()
    Training_data = []
    state_dict = {}
    with open(feature_path, 'rb') as handle:
        feature_data = pickle.load(handle)
    print("Begin loading demo data!")
    print(num_demo)
    count = 0
    demos = demo_paths
    sample_index = np.arange(num_demo)
    for t in tqdm(sample_index):
        path = demos[t]
        path_data = []
        e.set_env_state(path['init_state_dict'])
        #pdb.set_trace()
        #print(f"t {t}, path['init_state_dict'] {path['init_state_dict']}")
        actions = path['actions']
        observations = path['observations']  
        observations_visualization = path['observations_visualization']
        handVelocity = path['handVelocity']  
        for t in range(len(actions)):
            tmp = dict()
            obs = observations[t] 
            obs_visual = observations_visualization[t] 
            state_dict['desired_orien'] = quat2euler(path['init_state_dict']['desired_orien'])
            state_dict['qpos'] = obs_visual[:30]
            state_dict['qvel'] = obs_visual[30:]
            handpos = obs[:24]
            if (multi_task):
                tmp['handpos'] = obs[:30]
                pad_handpos = np.zeros(30)
                pad_handpos[6:] = handpos
                tmp['handpos'] = pad_handpos
            else:
                tmp['handpos'] = handpos
            handvel = handVelocity[t]
            tmp['handvel'] = handvel
            objpos = obs[24:27]
            tmp['objpos'] = objpos
            objvel = obs[27:33] 
            tmp['objvel'] = objvel
            tmp['desired_ori'] = obs[36:39] 
            objorient = obs[33:36] 
            tmp['objorient'] = objorient #ori_transform(objorient, tmp['desired_ori']) 
            tmp['observation'] = obs[42:45]  
            tmp['action'] = actions[t]
            tmp['pen_desired_orien'] = path['desired_orien']
            #pdb.set_trace()
            dict_value = feature_data[count].values()
            feature = list(dict_value)[0]
            tmp['rgbd_feature'] = feature
            tmp['count'] = count
            count += 1
            tmp['init_state_dict'] = path['init_state_dict']
            if False:
                e.env.mj_render()  # render the visualizer
            path_data.append(tmp)
        Training_data.append(path_data)
    return Training_data
