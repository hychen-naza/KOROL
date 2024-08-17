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
from utils.demo_loader import *
        


def door_main():
    # get current directory path
    folder_name = os.getcwd()

    demo_data = pickle.load(open("./Door/Data/Testing.pickle", 'rb'))
    feature_data_path = './Door/feature_door_demo_full.pickle'
    num_demo = 200 #200
    horizon = 70

    Training_data = door_demo_playback("door-v0", demo_data, feature_data_path, num_demo)
    num_handpos = len(Training_data[0][0]['handpos'])
    num_handvel = len(Training_data[0][0]['handvel'])
    num_objpos = len(Training_data[0][0]['objpos'])
    num_objvel = len(Training_data[0][0]['objvel'])
    num_init_pos = len(Training_data[0][0]['handle_init'])
    dt = 1  # simulation time for each step, assuming it should be 1
    num_obj = num_objpos + num_objvel + num_init_pos #+ num_objvel + implicit_obj_dim #num_objpos + num_objvel + num_init_pos # 
    num_hand = num_handpos
    #NN_size = [obj_input_size, 32, 64, 32, obj_output_size]  # define a neural network to learn the PID mapping
    
    '''
    Train the koopman dynamics from demo data
    '''
    Koopman = DraftedObservable(num_hand, num_obj)
    num_obs = Koopman.compute_observable(num_hand, num_obj)
    print("number of observables:", num_obs)
    A = np.zeros((num_obs, num_obs))  
    G = np.zeros((num_obs, num_obs))
    ## loop to collect data
    print("Drafted koopman training starts!\n")
    for k in tqdm(range(len(Training_data))):
        hand_OriState = Training_data[k][0]['handpos']
        obj_OriState = np.append(Training_data[k][0]['objpos'], np.append(Training_data[k][0]['objvel'], Training_data[k][0]['handle_init']))
        #pdb.set_trace()
        z_t = Koopman.z(hand_OriState, obj_OriState)  # initial states in lifted space
        for t in range(len(Training_data[k]) - 1):
            hand_OriState = Training_data[k][t + 1]['handpos']
            obj_OriState = np.append(Training_data[k][t+1]['objpos'],np.append(Training_data[k][t+1]['objvel'], Training_data[k][t+1]['handle_init']))
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
    np.save(os.path.join(folder_name, "gt_koopmanMatrix.npy"), cont_koopman_operator)

    # Evaluate the learned koopman operator 
    error = 0
    gt_pred = []
    for k in tqdm(range(len(Training_data))):
        hand_OriState = Training_data[k][0]['handpos']
        obj_OriState = np.append(Training_data[k][0]['objpos'], np.append(Training_data[k][0]['objvel'], Training_data[k][0]['handle_init']))
        z_t = Koopman.z(hand_OriState, obj_OriState)  # initial states in lifted space
        for t in range(horizon - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = np.dot(koopman_operator, z_t)
            z_t = z_t_1_computed
            error += np.mean(np.abs(Training_data[k][t + 1]['handpos'], z_t_1_computed[:num_handpos]))
            #pdb.set_trace()
            gt_pred.append(z_t_1_computed[:num_handpos])
    np.save("gt_pred_robot_state.npy", np.array(gt_pred))
    print(f"avg error {error/len(Training_data)}")

def tool_main():
    Velocity = False
    number_sample = 200  # num of unseen samples used for test
    # get current directory path
    folder_name = os.getcwd()

    demo_data = pickle.load(open("./Tool/Data/Hammer_task.pickle", 'rb'))
    feature_data_path = './Tool/feature_Tool_demo_full.pickle'
    num_demo = 200

    Training_data = hammer_demo_playback("hammer-v0", demo_data, feature_data_path, num_demo)
    dt = 1  # simulation time for each step, assuming it should be 1
    num_obj = 6
    num_hand = 26

    '''
    Train the koopman dynamics from demo data
    '''
    Koopman = DraftedObservable(num_hand, num_obj)
    num_obs = Koopman.compute_observable(num_hand, num_obj)
    print("number of observables:", num_obs)
    A = np.zeros((num_obs, num_obs))  
    G = np.zeros((num_obs, num_obs))
    ## loop to collect data
    print("Drafted koopman training starts!\n")
    for k in tqdm(range(len(Training_data))):
        hand_OriState = Training_data[k][0]['handpos']
        obj_OriState = np.append(Training_data[k][0]['objpos'],  Training_data[k][0]['nail_goal'])
        z_t = Koopman.z(hand_OriState, obj_OriState)  # initial states in lifted space
        for t in range(len(Training_data[k]) - 1):
            hand_OriState = Training_data[k][t + 1]['handpos']
            obj_OriState = np.append(Training_data[k][t+1]['objpos'],  Training_data[k][t+1]['nail_goal'])
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
    np.save(os.path.join(folder_name, "koopmanMatrix.npy"), cont_koopman_operator)

def relocate_main():
    Velocity = False
    number_sample = 200  # num of unseen samples used for test
    # get current directory path
    folder_name = os.getcwd()

    demo_data = pickle.load(open("./Relocation/Data/Relocate_task.pickle", 'rb'))
    feature_data_path = './Relocation/Data/generate_feature.pickle'
    num_demo = 10 #200

    Training_data = relocate_demo_playback("relocate-v0", demo_data, feature_data_path, num_demo)
    dt = 1  # simulation time for each step, assuming it should be 1
    num_obj = 6 #+ num_objvel + implicit_obj_dim #num_objpos + num_objvel + num_init_pos # 
    num_hand = 30

    '''
    Train the koopman dynamics from demo data
    '''
    Koopman = DraftedObservable(num_hand, num_obj)
    num_obs = Koopman.compute_observable(num_hand, num_obj)
    print("number of observables:", num_obs)
    A = np.zeros((num_obs, num_obs))  
    G = np.zeros((num_obs, num_obs))
    ## loop to collect data
    print("Drafted koopman training starts!\n")
    for k in tqdm(range(len(Training_data))):
        hand_OriState = Training_data[k][0]['handpos']
        obj_OriState = np.append(Training_data[k][0]['objpos'], Training_data[k][0]['desired_pos'])
        z_t = Koopman.z(hand_OriState, obj_OriState)  # initial states in lifted space
        for t in range(len(Training_data[k]) - 1):
            hand_OriState = Training_data[k][t + 1]['handpos']
            obj_OriState = np.append(Training_data[k][t+1]['objpos'], Training_data[k][t+1]['desired_pos'])
            #print(obj_OriState)
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
    np.save(os.path.join(folder_name, "koopmanMatrix.npy"), cont_koopman_operator)

    error = 0
    gt_pred = []
    for k in tqdm(range(10)):
        hand_OriState = Training_data[k][0]['handpos']
        obj_OriState = np.append(Training_data[k][0]['objpos'], Training_data[k][0]['desired_pos'])
        z_t = Koopman.z(hand_OriState, obj_OriState)  # initial states in lifted space
        for t in range(100 - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = np.dot(koopman_operator, z_t)
            z_t = z_t_1_computed
            error += np.mean(np.abs(Training_data[k][t + 1]['handpos'] - z_t_1_computed[:num_hand]))
    print(f"avg error {error/len(Training_data)}")        

def reori_main():
    Velocity = False
    number_sample = 200  # num of unseen samples used for test
    # get current directory path
    folder_name = os.getcwd()

    demo_data = pickle.load(open("./Reorientation/Data/Pen_task.pickle", 'rb'))
    feature_data_path = './Reorientation/Data/generate_feature.pickle'
    num_demo = 10 #200

    Training_data = reorientation_demo_playback("pen-v0", demo_data, feature_data_path, num_demo)
    dt = 1  # simulation time for each step, assuming it should be 1
    num_obj = 6 #+ num_objvel + implicit_obj_dim #num_objpos + num_objvel + num_init_pos # 
    num_hand = 24

    '''
    Train the koopman dynamics from demo data
    '''
    Koopman = DraftedObservable(num_hand, num_obj)
    num_obs = Koopman.compute_observable(num_hand, num_obj)
    print("number of observables:", num_obs)
    A = np.zeros((num_obs, num_obs))  
    G = np.zeros((num_obs, num_obs))
    ## loop to collect data
    print("Drafted koopman training starts!\n")
    for k in tqdm(range(len(Training_data))):
        hand_OriState = Training_data[k][0]['handpos']
        obj_OriState = np.append(Training_data[k][0]['objorient'], Training_data[k][0]['desired_ori'])
        z_t = Koopman.z(hand_OriState, obj_OriState)  # initial states in lifted space
        for t in range(len(Training_data[k]) - 1):
            hand_OriState = Training_data[k][t + 1]['handpos']
            obj_OriState = np.append(Training_data[k][t+1]['objorient'], Training_data[k][t+1]['desired_ori'])
            #print(obj_OriState)
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
    np.save(os.path.join(folder_name, "koopmanMatrix.npy"), cont_koopman_operator)

if __name__ == '__main__':
    env = "relocate"
    if env == "door":
        door_main()
    elif env == "tool":
        tool_main()
    elif env == "relocate":
        relocate_main()
    elif env == "reori":
        reori_main()