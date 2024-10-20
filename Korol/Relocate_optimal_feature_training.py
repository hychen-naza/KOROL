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
from utils.resnet_relocation import *
#from utils.resnet_relocation_vis import *
from utils.demo_loader import *
from scipy.linalg import logm
import scipy
import sys
import os
import random
import time 
import shutil
import robohive
import pdb 
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class RGBDDataset(Dataset):
    def __init__(self, demo_feature_path, img_path, img_size = None):
        # self.data is useless
        with open(demo_feature_path, 'rb') as f:
            self.demo_feature = pickle.load(f)
        self.img_path = img_path
        self.length = len(self.demo_feature) if img_size is None else img_size

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        path = os.path.join(self.img_path, "rgbd_"+str(index)+".npy") 
        rgbd = np.load(path)
        rgbd = np.transpose(rgbd, (2, 0, 1))
        return rgbd, self.demo_feature[index]['desired_pos'], self.demo_feature[index]['rgbd_feature'], index



def generate_feature(model, generate_feature_data_path, demo_feature_data_path, img_path, device="cuda", img_size=None):
    batch_size = 32
    generate_feature_data = []
    model.eval()
    train_dataset = RGBDDataset(demo_feature_data_path, img_path, img_size = img_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    error = 0
    count = 0
    with torch.no_grad():
        for batch_num, (rgbds, desired_pos, _, indexs) in enumerate(train_dataloader):
            rgbds = rgbds.float().to(device)
            desired_pos = desired_pos.float().to(device)
            outputs_feat = model(rgbds, desired_pos)
            outputs = outputs_feat.detach().cpu().numpy() #outputs_pos
            for index, output, rgbd in zip(indexs, outputs, rgbds):
                generate_feature_data.append({index:output})
            del rgbds
            del desired_pos
            del indexs
    with open(generate_feature_data_path, 'wb') as handle:
        pickle.dump(generate_feature_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


class DynamicsPredictionDataset(Dataset):
    def __init__(self, demo, img_path, length = 20):
        #with open(demo_path, 'rb') as f:
        self.demo = demo
        self.use_object_embedder = False
        self.img_path = img_path
        self.length = length

    def __len__(self):
        return len(self.demo)

    def __getitem__(self, index):
        path = self.demo[index]
        current_idx = random.randint(0, len(path)-(self.length+2))
        robot_current = path[current_idx]['handpos']
        desired_pos_current = path[current_idx]['desired_pos']
        feat_current = path[current_idx]['rgbd_feature']
        count = path[current_idx]['count']
        img_path = os.path.join(self.img_path, "rgbd_"+str(count)+".npy") 
        rgbd = np.load(img_path)
        rgbd = np.transpose(rgbd, (2, 0, 1))

        robot_next = [path[current_idx+i]['handpos'] for i in range(1,self.length+1)]
        feat_next = [path[current_idx+i]['rgbd_feature'] for i in range(1,self.length+1)]
        desired_ori_next = [path[current_idx+i]['desired_pos'] for i in range(1,self.length+1)]

        return rgbd, robot_current, feat_current, desired_pos_current, robot_next, feat_next

def train_koopman(Training_data, num_hand, num_obj, koopman_save_path):
    Koopman = DraftedObservable(num_hand, num_obj)
    num_obs = Koopman.compute_observable(num_hand, num_obj)

    A = np.zeros((num_obs, num_obs))  
    G = np.zeros((num_obs, num_obs))
    ## loop to collect data
    print("Drafted koopman training starts!\n")
    for k in tqdm(range(len(Training_data))):
        hand_OriState = Training_data[k][0]['handpos']
        obj_OriState = Training_data[k][0]['rgbd_feature'] 
        assert len(obj_OriState) == num_obj
        assert len(hand_OriState) == num_hand

        z_t = Koopman.z(hand_OriState, obj_OriState)  # initial states in lifted space
        for t in range(len(Training_data[k]) - 1):
            hand_OriState = Training_data[k][t + 1]['handpos']
            obj_OriState = Training_data[k][t+1]['rgbd_feature']
            z_t_1 = Koopman.z(hand_OriState, obj_OriState) 
            A += np.outer(z_t_1, z_t)
            G += np.outer(z_t, z_t)
            z_t = z_t_1
    M = len(Training_data) * (len(Training_data[0]) - 1)
    A /= M
    G /= M
    koopman_operator = np.dot(A, scipy.linalg.pinv(G)) 
    cont_koopman_operator = koopman_operator

    np.save(koopman_save_path, cont_koopman_operator)
    print(f"Koopman matrix is saved!\n")


def main():
    learningRate = 1e-4 
    num_demo = 200
    device = "cuda" if torch.cuda.is_available() else "cpu"
    folder_name = os.getcwd()
    num_hand = 30
    num_obj = 8 
    dynamics_batch_num = 8
    env_folder = 'Relocation'
    env_name = "relocate-v0"

    demo_data_path = f"./{env_folder}/Data/Relocate_task.pickle" #pickle.load(open(f"./{env_folder}/Data/Pen_task.pickle", 'rb'))
    demo_feature_data_path = f'./{env_folder}/Data/full_features.pickle'
    generate_feature_data_path = f'./{env_folder}/Data/generate_feature.pickle'
    img_path = f"./{env_folder}/Data"
    Controller_loc = f'./{env_folder}/controller/NN_controller_best.pt'
    koopman_save_path = f"./{env_folder}/koopmanMatrix.npy"
    demo_data = pickle.load(open(demo_data_path, 'rb'))

    NN_Input_size = 2 * num_hand
    NN_Output_size = num_hand
    NN_size = [NN_Input_size, 2 * NN_Input_size, 2 * NN_Input_size, NN_Input_size, NN_Output_size]  # define a neural network to learn the PID mapping
    Controller = FCNetwork(NN_size, nonlinearity='relu')
    Controller.load_state_dict(torch.load(Controller_loc))
    Controller.eval() # eval mode

    # training parameters
    resnet_model = ClassificationNetwork18_woDCT(feat_dim = 8)
    resnet_model = resnet_model.float()
    resnet_model.eval()
    resnet_model = resnet_model.to(device)

    e = GymEnv(env_name)
    e.reset()
    Testing_data = e.generate_unseen_data_relocate(200) 
    has_lower_loss = False
    ErrorInOriginalRobot = 0
    ErrorInOriginal = 0
    min_ErrorInOriginal = 1.70
    min_loss = 100

    loss = torch.nn.L1Loss()
    optimizer_feature = torch.optim.Adam(resnet_model.parameters(), lr=learningRate)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_feature, milestones=[4,8,12,16,20,24,28,32], gamma=0.2)

    generate_feature(resnet_model, generate_feature_data_path, demo_feature_data_path, img_path, device=device, img_size=num_demo*100)
    Training_data = relocate_demo_playback(env_name, demo_data, generate_feature_data_path, num_demo)

    # Resnet Dynamics Training dataset
    dynamics_train_dataset = DynamicsPredictionDataset(Training_data, img_path, length = int(args.N)) 
    dynamics_train_dataloader = DataLoader(dynamics_train_dataset, batch_size=dynamics_batch_num, shuffle=True, num_workers=4)

    Koopman = DraftedObservable(num_hand, num_obj)
    train_koopman(Training_data, num_hand, num_obj, koopman_save_path)
    cont_koopman_operator = np.load(koopman_save_path) # matrix_file
    #koopman_policy_control_relocate(env_name, Controller, Koopman, cont_koopman_operator, Testing_data, False, num_hand, num_obj, resnet_model, device)
    cont_koopman_operator = torch.from_numpy(cont_koopman_operator).to(device)
    
    # Train 
    for epoch in range(501):
        resnet_model.train()
        # Load the new Koopman operator
        if epoch != 0 and epoch % 10 == 0:
            learningRate *= 0.9
        optimizer_feature = torch.optim.Adam(resnet_model.parameters(), lr=learningRate)
        ErrorInOriginalRobot = 0
        epoch_loss = 0
        optimizer_feature.zero_grad()
        for batch_num, (rgbds, robot_currents, feat_currents, desired_pos_current, robot_nexts, feat_nexts) in enumerate(dynamics_train_dataloader):
            robot_currents = robot_currents.float().to(device)
            rgbds = rgbds.float().to(device)
            feat_currents = feat_currents.float().to(device)
            desired_pos_current = desired_pos_current.float().to(device)
            pred_feats = resnet_model(rgbds, desired_pos_current)
            batch_size = len(robot_currents)
            for i in range(batch_size):
                hand_OriState = robot_currents[i]
                obj_OriState = pred_feats[i]
                z_t_computed = Koopman.z_torch(hand_OriState, obj_OriState).to(device)
                for t in range(len(robot_nexts)):
                    robot_next = robot_nexts[t][i].float().to(device)
                    z_t_1_computed = torch.matmul(cont_koopman_operator.float(), z_t_computed.float()) 
                    ErrorInOriginalRobot += loss(z_t_1_computed[:num_hand], robot_next)
                    z_t_computed = z_t_1_computed
                    epoch_loss += loss(z_t_1_computed[:num_hand], robot_next).item()
            ErrorInOriginalRobot *= 0.05 # weights
            ErrorInOriginalRobot.backward()
            optimizer_feature.step()
            ErrorInOriginalRobot = 0
            optimizer_feature.zero_grad()

            del rgbds
            gc.collect()

        if (epoch >= 50 and epoch % 50 == 0):
            resnet_model.eval()
            with torch.no_grad():
                generate_feature(resnet_model, generate_feature_data_path, demo_feature_data_path, img_path, device=device, img_size=num_demo*100)
                Training_data = relocate_demo_playback(env_name, demo_data, generate_feature_data_path, num_demo)
                train_koopman(Training_data, num_hand, num_obj, koopman_save_path)
                cont_koopman_operator = np.load(koopman_save_path) # matrix_file
                if (epoch >= 50):
                    koopman_policy_control_relocate(env_name, Controller, Koopman, cont_koopman_operator, Testing_data, False, num_hand, num_obj, resnet_model, device)
                cont_koopman_operator = torch.from_numpy(cont_koopman_operator).to(device)
                

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', default=40, help='dynamic prediction length')
    args = parser.parse_args()
    print(f" N length {args.N}")
    main()


"""
Visualization Code
# if (batch_num in [0,1,2,3,4]):
            #     if (batch_num % 10 == 0):
            #         print(f"Processing {batch_num} batch")

            #     if (batch_num == 0):
            #         vis_imgs = []
            #         rgbd_imgs = []
            #     for i in range(batch_size):
            #         activation_map, activation = model.activation_vis(rgbds[i:i+1], desired_pos[i:i+1])
            #         activation_map[activation_map < 0] = 0
            #         rgbd = rgbds[i:i+1,:3].cpu()
            #         rgb = (rgbd * 128.0) + 128.
            #         rgb = rgb.to(torch.uint8)
            #         rgb = np.transpose(rgb.squeeze().numpy(), (1, 2, 0))
            #         #pdb.set_trace()
            #         # rgbd_imgs.append(Image.fromarray(rgb))

            #         fig, ax = plt.subplots()
            #         # Hide axes
            #         ax.axis('off')
            #         # Display the RGB image
            #         ax.imshow(rgb)
            #         # Overlay the heatmap
            #         ax.imshow(activation_map, cmap='jet', alpha=0.5)
            #         import io
            #         # Save the figure to a bytes buffer
            #         buf = io.BytesIO()
            #         plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            #         buf.seek(0)
            #         # Load the image from the buffer into PIL
            #         # Close the buffer and figure to free resources
            #         img_pil = Image.open(buf)
            #         #pdb.set_trace()
            #         vis_imgs.append(img_pil.resize((256,256)))
            #     if (batch_num == 4):
            #         #rgbd_imgs[0].save(f'./door_vis_gif/rgb_{epoch}_{batch_num}.gif', save_all=True,optimize=False, append_images=rgbd_imgs[1:], loop=0)
            #         vis_imgs[0].save(f'./tool_vis_gif/relocate_vis_{batch_num}.gif', save_all=True,optimize=False, append_images=vis_imgs[1:], loop=0)
            #         buf.close()
            #         plt.close(fig)
            # count += batch_size
            # if (count < 100):
            #     for i in range(batch_size):
            #         activation_map, activation = model.activation_vis(rgbds[i:i+1], desired_pos[i:i+1])
            #         activation_map[activation_map < 0] = 0
            #         rgbd = rgbds[i:i+1,:3].cpu()
            #         rgb = (rgbd * 128.0) + 128.
            #         rgb = rgb.to(torch.uint8)
            #         #print(f"batch {batch_num}, i {i}")
            #         #pdb.set_trace()
            #         plt.figure(figsize=(12, 8))
            #         plt.subplot(1, 2, 1)
            #         plt.imshow(np.transpose(rgb.squeeze().numpy(), (1, 2, 0)))
            #         #plt.title('Original Image')
            #         plt.axis('off')
            #         plt.subplot(1, 2, 2)
            #         plt.imshow(np.transpose(rgb.squeeze().numpy(), (1, 2, 0)))
            #         plt.imshow(activation_map, cmap='jet', alpha=0.5)
            #         #plt.title('Activation Map')
            #         plt.axis('off')
            #         #pdb.set_trace()
            #         plt.savefig(f'./relocate_vis/activation_map_overlay_{batch_num}_{i}.png')
"""