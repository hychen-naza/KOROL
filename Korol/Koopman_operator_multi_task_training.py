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
from tqdm import tqdm
from utils.gym_env import GymEnv
from utils.Observables import *
from utils.Koopman_evaluation import *
from utils.Controller import *
from utils.quatmath import quat2euler, euler2quat
from utils.coord_trans import ori_transform, ori_transform_inverse
# from utils.resnet import *
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
import pdb 
        
DESC = '''
Helper script to visualize demonstrations.\n
USAGE:\n
    Visualizes demonstrations on the env\n
    $ python utils/visualize_demos --env_name relocate-v0\n
'''

# MAIN =========================================================
# @click.command(help=DESC)
# @click.option('--env_name', type=str, help='environment to load', required= True)
# @click.option('--demo_file', type=str, help='demo file to load', default=None)
# @click.option('--num_demo', type=str, help='define the number of demo', default='0')  


def main():
    
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
    env = "Reorientation"
    # obj_embedder = FCNetwork([6, 8], nonlinearity='relu')
    # torch.save(obj_embedder, os.path.join(folder_name, env, "obj_embedder.pt"))
    obj_embedder = torch.load(os.path.join(folder_name, env, "obj_embedder.pt"))
    obj_embedder.eval()
    demos = pickle.load(open("./Reorientation/Data/Pen_task.pickle", 'rb'))
    Training_data_reorientation = reorientation_demo_playback("pen-v0", demos, num_demo, obj_embedder) #obj_embedder

    env = "Relocation"
    # obj_embedder = FCNetwork([6, 8], nonlinearity='relu')
    # torch.save(obj_embedder, os.path.join(folder_name, env, "obj_embedder.pt"))
    obj_embedder = torch.load(os.path.join(folder_name, env, "obj_embedder.pt"))
    obj_embedder.eval()
    demos = pickle.load(open("./Relocation/Data/Relocate_task.pickle", 'rb'))
    Training_data_relocate = relocate_demo_playback("relocate-v0", demos, num_demo, obj_embedder) #obj_embedder

    env = "Door"
    obj_embedder = torch.load(os.path.join(folder_name, env, "obj_embedder.pt"))
    obj_embedder.eval()
    demos = pickle.load(open("./Door/Data/Testing.pickle", 'rb'))
    Training_data_door = door_demo_playback("door-v0", demos, num_demo, obj_embedder)

    env = "Tool"
    obj_embedder = torch.load(os.path.join(folder_name, env, "obj_embedder.pt"))
    obj_embedder.eval()
    demos = pickle.load(open("./Tool/Data/Hammer_task.pickle", 'rb'))
    Training_data_tool = hammer_demo_playback("hammer-v0", demos, num_demo, obj_embedder)
    Training_data = Training_data_door + Training_data_tool + Training_data_relocate + Training_data_reorientation

    num_hand = len(Training_data[0][0]['handpos'])
    num_obj = len(Training_data[0][0]['rgbd_feature']) #+ num_objvel + implicit_obj_dim #num_objpos + num_objvel + num_init_pos # 

    '''
    Train the koopman dynamics from demo data
    '''

    if not os.path.exists(os.path.join(folder_name, "koopman")):
        os.mkdir(os.path.join(folder_name, "koopman"))
    else:
        shutil.rmtree(os.path.join(folder_name, "koopman"))   # Removes all the subdirectories!
        os.mkdir(os.path.join(folder_name, "koopman"))
    
    Koopman = DraftedObservable(num_hand, num_obj)
    print(f"Trained matrix num_hand {num_hand}, num_obj {num_obj}")
    num_obs = Koopman.compute_observable(num_hand, num_obj)
    print("number of observables:", num_obs)
    A = np.zeros((num_obs, num_obs))  
    G = np.zeros((num_obs, num_obs))
    ## loop to collect data
    print("Drafted koopman training starts!\n")
    for k in tqdm(range(len(Training_data))):
        hand_OriState = Training_data[k][0]['handpos']
        # NAZA, embed the training data again
        obj_OriState = Training_data[k][0]['rgbd_feature'] 
        #np.append(np.append(Training_data[k][0]['objpos'], np.append(Training_data[k][0]['objorient'], Training_data[k][0]['objvel'])), Training_data[k][0]['nail_goal'])
        #Training_data[k][0]['rgbd_feature'] 
        z_t = Koopman.z(hand_OriState, obj_OriState)  # initial states in lifted space
        for t in range(len(Training_data[k]) - 1):
            # if k==0:
            #     print(f"t: {t}, obj_OriState {obj_OriState}, num_hand {num_hand}, num_obj {num_obj}")
            hand_OriState = Training_data[k][t + 1]['handpos']
            # NAZA, embed the training data again
            obj_OriState = Training_data[k][t+1]['rgbd_feature']
            #np.append(np.append(Training_data[k][t + 1]['objpos'], np.append(Training_data[k][t + 1]['objorient'], Training_data[k][t + 1]['objvel'])), Training_data[k][t + 1]['nail_goal'])
            #Training_data[k][t+1]['rgbd_feature']
            z_t_1 = Koopman.z(hand_OriState, obj_OriState) # states in lifted space at next time step
            # A and G are cumulated through all the demo data
            A += np.outer(z_t_1, z_t)
            G += np.outer(z_t, z_t)
            z_t = z_t_1
    M = len(Training_data) * (len(Training_data[0]) - 1)
    A /= M
    G /= M
    koopman_operator = np.dot(A, scipy.linalg.pinv(G)) # do not use np.linalg.pinv, it may include all large singular values
    # cont_koopman_operator = logm(koopman_operator) / dt
    cont_koopman_operator = koopman_operator
    # generate another matrix with similar matrix magnitude to verify the correctness of the learnt koopman matrix
    # we want to see that each element in the matrix does matter
    koopman_mean = np.mean(cont_koopman_operator)
    print("Koopman mean:%f"%(koopman_mean))
    koopman_std = np.std(cont_koopman_operator)
    print("Koopman std:%f"%(koopman_std))
    Test_matrix = np.random.normal(loc = koopman_mean, scale = koopman_std, size = cont_koopman_operator.shape)
    print("Fake matrix mean:%f"%(np.mean(Test_matrix)))
    print("Fake matrix std:%f"%(np.std(Test_matrix)))
    print("Drafted koopman training ends!\n")
    # print("The drafted koopman matrix is: ", cont_koopman_operator)
    # print("The drafted koopman matrix shape is: ", koopman_operator.shape)
            # save the trained koopman matrix
    save_path = os.path.join(folder_name, env, "koopmanMatrix.npy")
    np.save(save_path, cont_koopman_operator)
    print(f"Koopman matrix is saved in {save_path}!\n")

    print("Koopman final testing starts!\n")
    ErrorInLifted, ErrorInOriginal, ErrorInOriginalObj, ErrorInOriginalRobot = koopman_evaluation(Koopman, cont_koopman_operator, Training_data, num_hand, num_obj, obj_embedder)
    #Fake_ErrorInLifted, Fake_ErrorInOriginal, Fake_ErrorInOriginalObj, Fake_ErrorInOriginalRobot = koopman_evaluation(Koopman, Test_matrix, Training_data, Velocity, num_hand, num_obj, obj_embedder)
    print("Koopman final testing ends!\n")

    print("sum error")
    print("The final test accuracy in lifted space is: %f, and the accuracy in original space is: %f, and the accuracy in original obj pos is: %f., and the accuracy in original robot pos is: %f."
          %(np.sum(ErrorInLifted), np.sum(ErrorInOriginal), np.sum(ErrorInOriginalObj), np.sum(ErrorInOriginalRobot)))
    # print("The fake test accuracy in lifted space is: %f, and the fake accuracy in original space is: %f, and the accuracy in original obj pos is: %f, "
    #       %(np.median(Fake_ErrorInLifted), np.median(Fake_ErrorInOriginal), np.median(Fake_ErrorInOriginalObj), np.me))

    print("mean error")
    print("The final test accuracy in lifted space is: %f, and the accuracy in original space is: %f, and the accuracy in original obj pos is: %f, and the accuracy in original robot pos is: %f."
          %(np.mean(ErrorInLifted), np.mean(ErrorInOriginal), np.mean(ErrorInOriginalObj), np.mean(ErrorInOriginalRobot)))
    # print("The fake test accuracy in lifted space is: %f, and the fake accuracy in original space is: %f, and the accuracy in original obj pos is: %f."%(np.mean(Fake_ErrorInLifted), np.mean(Fake_ErrorInOriginal), np.mean(Fake_ErrorInOriginalObj)))


def hammer_demo_playback(env_name, demo_paths, num_demo, obj_embedder):
    e = GymEnv(env_name)
    e.reset()
    Training_data = []
    print("Begin loading demo data!")
    print(num_demo)
    sample_index = np.arange(num_demo)
    # sample_index = range(num_demo)
    for t in tqdm(sample_index):
        path = demo_paths[t]
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
            pad_action = np.zeros(28)
            pad_action[1:3] = action[:2]
            pad_action[4:] = action[2:]
            tmp['action'] = pad_action
            tmp['rgbd_feature'] = np.concatenate((tmp['objpos'], tmp['objorient'], tmp['nail_goal']), axis=0)
            tmp['rgbd_feature'] = obj_embedder(torch.from_numpy(tmp['rgbd_feature'])).detach().numpy()
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
    print("Begin loading demo data!")
    print(num_demo)
    sample_index = np.arange(num_demo)
    for t in tqdm(sample_index):
        path = demo_paths[t]
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
            tmp['objpos'] = objpos - obs[45:48] 
            tmp['objorient'] = obs_visual[33:36]
            tmp['obj_pos_visual'] = obs_visual[30:33]
            tmp['objvel'] = allvel[30:]
            tmp['observation'] = obs[36:39]  
            tmp['action'] = actions[t]
            tmp['rgbd_feature'] = np.concatenate((tmp['objpos'], tmp['objorient']), axis=0)
            tmp['rgbd_feature'] = obj_embedder(torch.from_numpy(tmp['rgbd_feature'])).detach().numpy()
            if False:
                e.env.mj_render()  # render the visualizer
            path_data.append(tmp)
        Training_data.append(path_data)
    return Training_data


def reorientation_demo_playback(env_name, demo_paths, num_demo, obj_embedder):
    e = GymEnv(env_name)
    e.reset()
    Training_data = []
    state_dict = {}
    print("Begin loading demo data!")
    print(num_demo)
    sample_index = np.arange(num_demo)
    for t in tqdm(sample_index):
        path = demo_paths[t]
        path_data = []
        e.set_env_state(path['init_state_dict'])
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
            objorient = obs[33:36] 
            tmp['objorient'] = ori_transform(objorient, tmp['desired_ori']) 
            tmp['observation'] = obs[42:45]  
            tmp['action'] = actions[t]
            tmp['pen_desired_orien'] = path['desired_orien']
            tmp['rgbd_feature'] = np.concatenate((tmp['objpos'], tmp['objorient']), axis=0)
            tmp['rgbd_feature'] = obj_embedder(torch.from_numpy(tmp['rgbd_feature'])).detach().numpy()
            if False:
                e.env.mj_render()  # render the visualizer
            path_data.append(tmp)
        Training_data.append(path_data)
    return Training_data


if __name__ == '__main__':
    main()