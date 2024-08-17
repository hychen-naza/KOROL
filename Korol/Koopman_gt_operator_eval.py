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
from utils.demo_loader import *
from utils.Koopman_gt_evaluation import * 

env_name = "Tool"

def main():
    num_demo = 200 # if num_demo != 0 -> we manually define the num of demo for testing the sample efficiency
    folder_name = os.getcwd()

    if (env_name == "Door"):
        e = GymEnv("door-v0")
        e.reset()
        Testing_data = e.generate_unseen_data_door(1000) 
        num_obj = 7 #implicit_obj_dim # + num_objvel + implicit_obj_dim
        num_hand = 28
        Controller_loc = './Door/controller/NN_controller_best.pt'
    elif (env_name == "Tool"):
        e = GymEnv("hammer-v0")
        e.reset()
        Testing_data = e.generate_unseen_data_hammer(1000) 
        num_obj = 6 #implicit_obj_dim # + num_objvel + implicit_obj_dim
        num_hand = 26
        Controller_loc = './Tool/controller/NN_controller_best.pt'
    elif (env_name == "Relocate"):
        e = GymEnv("relocate-v0")
        e.reset()
        Testing_data = e.generate_unseen_data_relocate(1000) 
        num_obj = 6 #implicit_obj_dim # + num_objvel + implicit_obj_dim
        num_hand = 30
        Controller_loc = './Relocation/controller/NN_controller_best.pt'
    elif (env_name == "Reorientation"):
        e = GymEnv("pen-v0")
        e.reset()
        Testing_data = e.generate_unseen_data_reorientate(1000) 
        num_obj = 6 #implicit_obj_dim # + num_objvel + implicit_obj_dim
        num_hand = 24
        Controller_loc = './Reorientation/controller/NN_controller_best.pt'

    # define the input and outputs of the neural network
    NN_Input_size = 2 * num_hand
    NN_Output_size = num_hand
    NN_size = [NN_Input_size, 2 * NN_Input_size, 2 * NN_Input_size, NN_Input_size, NN_Output_size]  # define a neural network to learn the PID mapping
    Controller = FCNetwork(NN_size, nonlinearity='relu')
    Controller.load_state_dict(torch.load(Controller_loc))
    Controller.eval() # eval mode

    print("Trained drafted koopman matrix loaded!")
    cont_koopman_operator = np.load(os.path.join(folder_name, "koopmanMatrix.npy")) # matrix_file
    Koopman = DraftedObservable(num_hand, num_obj)
    
    if (env_name == "Door"):
        door_koopman_policy_control_unseenTest("door-v0", Controller, Koopman, cont_koopman_operator, Testing_data, num_hand, num_obj)
    elif (env_name == "Tool"):
        tool_koopman_policy_control_unseenTest("hammer-v0", Controller, Koopman, cont_koopman_operator, Testing_data, num_hand, num_obj)
    elif (env_name == "Relocate"):
        relocate_koopman_policy_control_unseenTest("relocate-v0", Controller, Koopman, cont_koopman_operator, Testing_data, num_hand, num_obj)
    elif (env_name == "Reorientation"):
        reori_koopman_policy_control_unseenTest("pen-v0", Controller, Koopman, cont_koopman_operator, Testing_data, num_hand, num_obj)

if __name__ == '__main__':
    main()

