from glob import escape
from attr import asdict
import numpy as np
import time
from tqdm import tqdm
from utils.gym_env import GymEnv
from utils.quatmath import euler2quat
from utils.coord_trans import ori_transform, ori_transform_inverse
import torch
import pdb 

def door_koopman_policy_control_unseenTest(env_name, controller, koopman_object, koopman_matrix, Test_data, num_hand, num_obj):
    print("Testing the learned koopman dynamcis!")
    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    horizon = 70 # selected time steps that is enough to finish the task with good performance
    e.reset()
    init_state_dict = dict()
    # e.set_env_state(path['init_state_dict'])
    success_list_sim = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        hand_OriState = Test_data[k][0]['handpos']
        obj_OriState = np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objvel'], Test_data[k][0]['handle_init']))
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['door_body_pos'] = Test_data[k][0]['init']['door_body_pos']  # fixed for each piece of demo data
        e.set_env_state(init_state_dict)
        z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        
        for t in range(horizon - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
            obj_pos_world = x_t_1_computed[28:31]  # handle pos
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint states
            hand_pos_desired = hand_pos
            z_t = z_t_1_computed
            current = e.get_env_state()['qpos'][:28] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)
            obs_dict = e.env.get_obs_dict(e.env.sim)
            current_hinge_pos = obs_dict['door_pos']#obj_obs[28:29] # door opening angle
        if current_hinge_pos > 1.35:
            print(k)
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))


def tool_koopman_policy_control_unseenTest(env_name, controller, koopman_object, koopman_matrix, Test_data, num_hand, num_obj):
    print("Testing the learned koopman dynamcis!")
    print("Begin to compute the simulation errors!")
    e = GymEnv(env_name)
    horizon = 51 # selected time steps that is enough to finish the task with good performance
    e.reset()
    init_state_dict = dict()
    success_list_sim = []
    success_rate = str()
    for k in tqdm(range(len(Test_data))):
        num_handpos = len(Test_data[k][0]['handpos'])
        hand_OriState = Test_data[k][0]['handpos']
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['board_pos'] = Test_data[k][0]['init']['board_pos']  # fixed for each piece of demo data
        e.set_env_state(init_state_dict)
        obs_dict = e.env.get_obs_dict(e.env.sim) 
        obj_OriState = np.append(obs_dict['target_pos'], obs_dict['goal_pos'])
        # noise = (np.random.rand(obj_OriState.shape[0]) - 0.5)* 0.02
        # obj_OriState += noise
        #pdb.set_trace()
        z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        
        for t in range(horizon - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
            hand_pos = x_t_1_computed[:num_handpos]  # desired hand joint states
            hand_pos_desired = hand_pos
            z_t = z_t_1_computed
            current = e.get_env_state()['qpos'][:num_handpos] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            e.step(NN_output)
            obs_dict = e.env.get_obs_dict(e.env.sim)
            current_nail_pos = obs_dict['target_pos']
            goal_nail_pos = obs_dict['goal_pos']
            dist = np.linalg.norm(current_nail_pos - goal_nail_pos)
        if dist < 0.01:
            print(k)
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    

def relocate_koopman_policy_control_unseenTest(env_name, controller, koopman_object, koopman_matrix, Test_data, num_hand, num_obj):
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
        
        #np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
        init_state_dict['qpos'] = Test_data[k][0]['init']['qpos']
        init_state_dict['qvel'] = Test_data[k][0]['init']['qvel']
        init_state_dict['obj_pos'] = Test_data[k][0]['init']['obj_pos']
        init_state_dict['target_pos'] = Test_data[k][0]['init']['target_pos']
        e.set_env_state(init_state_dict)
        hand_OriState = Test_data[k][0]['handpos']
        obj_OriState = np.append(e.env.get_obs_dict(e.env.sim)['obj_pos'], Test_data[k][0]['init']['target_pos'])
        
        z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        for t in range(horizon):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
            hand_pos = x_t_1_computed[:num_handpos] 
            hand_pos_desired = hand_pos
            z_t = z_t_1_computed
            current = e.get_env_state()['hand_qpos'] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation)  
            err = e.env.get_obs_dict(e.env.sim)['obj_tar_err']
            # if (k % 10 == 0 and t % 5 ==0):
            #     print(f"desired_pos {desired_pos}, obj_pos {obj_pos}, err {err}")
            if np.linalg.norm(err) < 0.1:
                success_count_sim[t] = 1
        if sum(success_count_sim) > success_threshold:
            print(f"success in {k}")
            success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data))) 


def reori_koopman_policy_control_unseenTest(env_name, controller, koopman_object, koopman_matrix, Test_data, num_hand, num_obj):
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
        
        #np.append(Test_data[k][0]['objpos'], np.append(Test_data[k][0]['objorient'], Test_data[k][0]['objvel']))
        # init_state_dict['qpos'] = np.append(Test_data[k][0]['handpos'] + Test_data[k][0]['final_handpos'], np.zeros(6))
        init_state_dict['qpos'] = np.append(Test_data[k][0]['handpos'], np.zeros(6))
        init_state_dict['qvel'] = np.zeros(30)#np.append(Test_data[k][0]['handvel'], np.zeros(6))
        init_state_dict['desired_orien'] = euler2quat(Test_data[k][0]['pen_desired_orien'])

        e.set_env_state(Test_data[k][0]['init_state_dict']) #init_state_dict
        obj_OriState = np.append(e.env.get_obs_dict(e.env.sim)['obj_rot'], Test_data[k][0]['pen_desired_orien'])

        z_t = koopman_object.z(hand_OriState, obj_OriState)  # initial states in lifted space
        for t in range(horizon):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
            hand_pos_desired = x_t_1_computed[:num_handpos]  # desired hand joint state
            z_t = z_t_1_computed
            current = e.get_env_state()['qpos'][:num_handpos] # current state
            set_goal = hand_pos_desired.copy() # next state
            NN_input = torch.from_numpy(np.append(current, set_goal))
            NN_output = controller(NN_input).detach().numpy()   
            #pdb.set_trace()
            e.step(NN_output)  # Visualize the demo using the actions (more like a simulation
            obs_dict = e.env.get_obs_dict(e.env.sim)
            obj_vel = obs_dict['obj_vel']       
            orien_similarity_sim = np.dot(obs_dict['obj_rot'], obs_dict['obj_des_rot'])
            success_count_sim[t] = 1 if (orien_similarity_sim > 0.90) else 0
        if np.abs(obs_dict['obj_err_pos'])[2] > 0.15:
            fall_list_sim.append(1)
        else:
            if sum(success_count_sim) > success_threshold and np.mean(np.abs(obj_vel)) < 1.: 
                print(k)
                success_list_sim.append(1)
    print("Success rate (sim) = %f" % (len(success_list_sim) / len(Test_data)))
    success_rate += "Success rate (sim) = %f\n" % (len(success_list_sim) / len(Test_data))  

