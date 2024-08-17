from cProfile import label
from glob import escape
from attr import asdict
import torch
#import mj_envs
import click 
import json
import os
import numpy as np
import gym
import pickle
from tqdm import tqdm
from utils.gym_env import GymEnv
from utils.Observables import *
from utils.Koopman_evaluation import *
from utils.Controller import *
from utils.quatmath import quat2euler, euler2quat
from utils.coord_trans import ori_transform, ori_transform_inverse
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.linalg import logm
import scipy
import sys
import os
import random
import time 
import shutil
import robohive
from PIL import Image
import pdb 

DESC = '''
Helper script to visualize demonstrations.\n
USAGE:\n
    Visualizes demonstrations on the env\n
    $ python utils/visualize_demos --env_name relocate-v0\n
'''

# MAIN =========================================================
@click.command(help=DESC)
# @click.option('--env_name', type=str, help='environment to load', required= True)
# @click.option('--demo_file', type=str, help='demo file to load', default=None)
# @click.option('--num_demo', type=str, help='define the number of demo', default='0') 
@click.option('--koopmanoption', type=str, help='Indicate the koopman choice (Drafted, MLP, GNN)', default=None)   
# @click.option('--velocity', type=str, help='If using hand velocity', default=None)
# @click.option('--control', type=str, help='apply with a controller', default='')
@click.option('--error_type', type=str, help='define how to calculate the errors', default='demo') # two options: demo; goal
# @click.option('--visualize', type=str, help='define whether or not to visualze the manipulation results', default='') # two options: demo; goal

def main(koopmanoption, error_type):
    
    
    controller = True
    Visualize = True
    num_demo = 200
    folder_name = os.getcwd()
    resnet_model = ClassificationNetwork18(feat_dim = 8)
    # checkpoint18 = torch.load("./Door/door_full_resnet", map_location=torch.device('cpu'))
    # resnet_model.load_state_dict(checkpoint18['model_state_dict'])
    #resnet_model = torch.load(os.path.join(folder_name, "Door/door_resnet.pt"))
    resnet_model.eval()
    # resnet_init_model = torch.load(os.path.join(folder_name, "Door/door_init_resnet.pt"))
    # resnet_init_model.eval()

    # loading demo datas
    # env = "Relocation"
    # obj_embedder = torch.load(os.path.join(folder_name, env, "obj_embedder.pt"))
    # obj_embedder.eval()
    # demos = pickle.load(open("./Relocation/Data/Relocate_task.pickle", 'rb'))
    # Training_data_relocate = relocate_demo_playback("relocate-v0", demos, num_demo, obj_embedder) 

    # pdb.set_trace()

    # env = "Reorientation"
    # obj_embedder = torch.load(os.path.join(folder_name, env, "obj_embedder.pt"))
    # obj_embedder.eval()
    # demos = pickle.load(open("./Reorientation/Data/Pen_task.pickle", 'rb'))
    # Training_data_reorientation = reorientation_demo_playback("pen-v0", demos, num_demo, obj_embedder) #obj_embedder

    # env = "Door"
    # obj_embedder = torch.load(os.path.join(folder_name, env, "obj_embedder.pt"))
    # obj_embedder.eval()
    # demos = pickle.load(open("./Door/Data/Testing.pickle", 'rb'))
    # Training_data_door = door_demo_playback("door-v0", demos, num_demo, obj_embedder)
    
    env = "Tool"
    obj_embedder = torch.load(os.path.join(folder_name, env, "obj_embedder.pt"))
    obj_embedder.eval()
    demos = pickle.load(open("./Tool/Data/Hammer_task.pickle", 'rb'))
    Training_data_tool = hammer_demo_playback("hammer-v0", demos, num_demo, obj_embedder)
    pdb.set_trace()
    Koopman_matrix_path = os.path.join(folder_name, env, "koopmanMatrix.npy")

    Training_data = Training_data_tool #Training_data_door + Training_data_tool + Training_data_relocate + Training_data_reorientation

    num_hand = len(Training_data[0][0]['handpos'])
    num_obj = len(Training_data[0][0]['rgbd_feature'])
    
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

    print(f"Trained drafted koopman matrix loaded!, path {Koopman_matrix_path}")
    cont_koopman_operator = np.load(Koopman_matrix_path) # matrix_file
    Koopman = DraftedObservable(num_hand, num_obj)
    print(f"Trained drafted koopman matrix loaded! num_hand {num_hand}, num_obj {num_obj}")
    if controller:

        errors_simu = koopman_policy_control_door("door-v0", Controller_door, Koopman, cont_koopman_operator, Training_data[:200], False, num_hand, num_obj, koopmanoption, error_type, Visualize, obj_embedder, folder_name)
        
        errors_simu = koopman_policy_control_hammer("hammer-v0", Controller_tool, Koopman, cont_koopman_operator, Training_data[200:400], num_hand, num_obj, koopmanoption, error_type)

        errors_simu = koopman_policy_control_relocate("relocate-v0", Controller_relocate, Koopman, cont_koopman_operator, Training_data[400:600], False, num_hand, num_obj, koopmanoption, error_type)

        errors_simu = koopman_policy_control_reorientation("pen-v0", Controller_reorientate, Koopman, cont_koopman_operator, Training_data[600:], False, num_hand, num_obj, koopmanoption, error_type)

    print("Finish the evaluation!")


    
def hammer_demo_playback(env_name, demo_paths, num_demo, obj_embedder):
    e = GymEnv(env_name)
    e.reset()
    Training_data = []
    print("Begin loading demo data!")
    print(num_demo)
    sample_index = np.arange(num_demo)
    count = 0
    # sample_index = range(num_demo)
    for i in tqdm(sample_index):
        path = demo_paths[i]
        path_data = []
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
            pad_handpos = np.zeros(30)
            pad_handpos[3:5] = handpos[:2]
            pad_handpos[6:] = handpos[2:]
            tmp['handpos'] = pad_handpos
            tmp['handvel'] = allvel[:26]
            objpos = obs[49:52] + obs[42:45] 
            tmp['objpos'] = objpos 
            tmp['objorient'] = obs[39:42] 
            tmp['objvel'] = obs[27:33] 
            tmp['nail_goal'] = obs[46:49] 
            #print(f"t {t}, objpos {objpos}, objorient {tmp['objorient']} nail_goal {obs[46:49]}")
            tmp['observation'] = obs[49:52]
            action = actions[t]
            # pad_action = np.zeros(28)
            # pad_action[1:3] = action[:2]
            # pad_action[4:] = action[2:]
            tmp['action'] = action #pad_action
            tmp['rgbd_feature'] = np.concatenate((tmp['objpos'], tmp['objorient'], tmp['nail_goal']), axis=0)
            tmp['rgbd_feature'] = obj_embedder(torch.from_numpy(tmp['rgbd_feature'])).detach().numpy()
            tmp['count'] = count
            #if (t == 0):
            rgb, depth = e.env.mj_render()
            img_obs = Image.fromarray(copy.deepcopy(rgb))
            img_obs.save(f"reorientation_{i}_{t}.png")
            pdb.set_trace()
            rgb = (rgb.astype(np.uint8) - 128.0) / 128
            depth = depth[...,np.newaxis]
            rgbd = np.concatenate((rgb,depth),axis=2)
            np.save(f'./Tool/Data/rgbd_{count}.npy', rgbd) #./Door/Partial_Distinct_Data/rgbd_{count}.npy
            count += 1
            e.step(actions[t])
            path_data.append(tmp)
        Training_data.append(path_data)
    #pdb.set_trace()
    return Training_data


def door_demo_playback(env_name, demo_paths, num_demo, obj_embedder):
    # feature_door_demo_full
    # with open('./Door/feature_door_demo_full.pickle', 'rb') as handle:
    #     feature_data = pickle.load(handle)
    # return Training_data
    e = GymEnv(env_name)
    e.reset()
    Training_data = []
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
        #handle_init_feature = feature_data[i][0]['rgbd_feature']
        # rgb, depth = e.env.mj_render()
        # # from PIL import Image 
        # # img_obs = Image.fromarray(copy.deepcopy(rgb))
        # # img_obs.save(f"door_opening_{i}_{tt}.png")
        # depth = depth[...,np.newaxis]
        # rgbd = np.concatenate((rgb,depth),axis=2)
        # rgbd = np.transpose(rgbd, (2, 0, 1))
        # rgbd = rgbd[np.newaxis, ...]
        #handle_init_feature = feature_data[i][0]['handle_init_feature'] 
        #resnet_init_model(torch.from_numpy(rgbd)).detach().numpy()[0]
        rgbds = []
        for tt in range(len(actions)):
            tmp = dict()
            if tt == 0:
                tmp['init'] = path['init_state_dict']
            obs = observations[tt] 
            obs_visual = observations_visualize[tt]
            handpos = obs_visual[:28] 
            pad_handpos = np.zeros(30)
            pad_handpos[2:] = handpos
            tmp['handpos'] = pad_handpos
            tmp['handvel'] = obs_visual[30:58]
            tmp['objpos'] = obs[32:35]
            
            tmp['objvel'] = obs_visual[58:59]
            tmp['handle_init'] = path['init_state_dict']['door_body_pos'] 

            #tmp['handle_init_feature'] = handle_init_feature
            tmp['observation'] = obs[35:38]
            tmp['action'] = actions[tt]

            # dict_value = feature_data[count].values()
            # dict_value = list(dict_value)[0]
            # predict = obj_embedder(torch.from_numpy(dict_value)).detach().numpy()
            gt_predict = obj_embedder(torch.from_numpy(tmp['objpos'])).detach().numpy()
            #if (tt == 0):
            #    print(f"i {i}, count {count} tmp['objpos'] {tmp['objpos']} dict_value {dict_value}")
            tmp['rgbd_feature'] = gt_predict #predict
            
            count += 1
            path_data.append(tmp)
        Training_data.append(path_data)
    
    # with open('Door/feature_door_demo_full.pickle', 'wb') as handle:
    #     pickle.dump(Training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return Training_data

def relocate_demo_playback(env_name, demo_paths, num_demo, obj_embedder): #, obj_embedder
    e = GymEnv(env_name)
    e.reset()
    Training_data = []
    feature_data = []
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
        handVelocity = path['handVelocity'] 
        for t in range(len(actions)):
            tmp = dict()
            feature = dict()
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
            tmp['objpos'] = objpos - obs[45:48] 
            tmp['objorient'] = obs_visual[33:36]
            tmp['obj_pos_visual'] = obs_visual[30:33]
            tmp['objvel'] = allvel[30:]
            tmp['observation'] = obs[36:39]  
            tmp['action'] = actions[t]
            tmp['rgbd_feature'] = np.concatenate((tmp['objpos'], tmp['objorient']), axis=0)
            tmp['rgbd_feature'] = obj_embedder(torch.from_numpy(tmp['rgbd_feature'])).detach().numpy()
            feature['desired_pos'] = tmp['desired_pos']
            feature['rgbd_feature'] = tmp['rgbd_feature']
            # rgb, depth = e.env.mj_render()
            # # img_obs = Image.fromarray(copy.deepcopy(rgb))
            # # img_obs.save(f"reorientation_{i}_{t}.png")
            # rgb = (rgb.astype(np.uint8) - 128.0) / 128
            # depth = depth[...,np.newaxis]
            # rgbd = np.concatenate((rgb,depth),axis=2)
            # np.save(f'./Relocation/Data/rgbd_{count}.npy', rgbd) #./Door/Partial_Distinct_Data/rgbd_{count}.npy
            tmp['count'] = count
            count += 1
            e.step(actions[t])
            if False:
                e.env.mj_render()  # render the visualizer
            path_data.append(tmp)
            feature_data.append(feature)
        Training_data.append(path_data)
    with open('./Relocation/Data/full_features.pickle', 'wb') as handle: #Partial_Distinct_Data/
        pickle.dump(feature_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return Training_data

def reorientation_demo_playback(env_name, demo_paths, num_demo, obj_embedder):
    e = GymEnv(env_name)
    e.reset()
    Training_data = []
    state_dict = {}
    print("Begin loading demo data!")
    print(num_demo)
    count = 0
    sample_index = np.arange(num_demo)
    feature_data = []
    for i in tqdm(sample_index):
        path = demo_paths[i]
        path_data = []
        e.set_env_state(path['init_state_dict'])
        actions = path['actions']
        observations = path['observations']  
        observations_visualization = path['observations_visualization']
        handVelocity = path['handVelocity']  
        # skip = False
        # is_distinct = True
        # for idx, pos in enumerate(obj_pos_list):
        #     if np.linalg.norm(objpos_init - pos) < 0.05:
        #         is_distinct = False
        #         if obj_pos_dict[idx] >= 3:
        #             skip = True
        #             #print(f"skip i {i}, objpos {objpos_init}, {len(obj_pos_list)}")
        #             break
        #         else:
        #             obj_pos_dict[idx] += 1
        #             #print(f"update i {i}, objpos {objpos_init}, {len(obj_pos_list)}")
        #             break

        # if skip:
        #     continue
        # elif is_distinct:
        #     obj_pos_dict[distinct_count] = 1
        #     distinct_count += 1
        #     obj_pos_list.append(objpos_init)
        #     demo_idx.append(i)
        # else:
        #     demo_idx.append(i)

        for t in range(len(actions)):
            
            tmp = dict()
            feature = dict()
            obs = observations[t] 
            obs_visual = observations_visualization[t] 
            state_dict['desired_orien'] = quat2euler(path['init_state_dict']['desired_orien'])
            state_dict['qpos'] = obs_visual[:30]
            state_dict['qvel'] = obs_visual[30:]
            handpos = obs[:24]
            pad_handpos = np.zeros(30)
            pad_handpos[6:] = handpos
            tmp['handpos'] = pad_handpos
            handvel = handVelocity[t]
            tmp['handvel'] = handvel
            objpos = obs[24:27]
            tmp['objpos'] = objpos
            objvel = obs[27:33] 
            tmp['objvel'] = objvel
            tmp['desired_ori'] = obs[36:39] 
            # if (t == 0):
            #     print(f"i {i}, t {t}, tmp['desired_ori'] {tmp['desired_ori']}")
            objorient = obs[33:36] 
            tmp['objorient'] = ori_transform(objorient, tmp['desired_ori']) 
            tmp['observation'] = obs[42:45]  
            tmp['action'] = actions[t]
            tmp['pen_desired_orien'] = path['desired_orien']
            tmp['rgbd_feature'] = np.concatenate((tmp['objpos'], tmp['objorient']), axis=0)
            tmp['rgbd_feature'] = obj_embedder(torch.from_numpy(tmp['rgbd_feature'])).detach().numpy()
            tmp['init_state_dict'] = path['init_state_dict']

            feature['desired_ori'] = obs[36:39] 
            feature['rgbd_feature'] = tmp['rgbd_feature']

            # rgb, depth = e.env.mj_render()
            # # img_obs = Image.fromarray(copy.deepcopy(rgb))
            # # img_obs.save(f"reorientation_{i}_{t}.png")
            # rgb = (rgb.astype(np.uint8) - 128.0) / 128
            # depth = depth[...,np.newaxis]
            # rgbd = np.concatenate((rgb,depth),axis=2)
            # np.save(f'./Reorientation/Data/rgbd_{count}.npy', rgbd) #./Door/Partial_Distinct_Data/rgbd_{count}.npy
            tmp['count'] = count
            count += 1
            e.step(actions[t])
            if False:
                e.env.mj_render()  # render the visualizer
            path_data.append(tmp)
            feature_data.append(feature)
        Training_data.append(path_data)
    with open('./Reorientation/Data/full_features.pickle', 'wb') as handle: #Partial_Distinct_Data/
        pickle.dump(feature_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return Training_data



if __name__ == '__main__':
    main()

