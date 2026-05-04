import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torchvision.io as rv

class DeepVIODataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the sequence subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        # List all sequence folders (e.g., 'seq_0', 'seq_1', etc.)
        self.sequences = [os.path.join(root_dir, d) for d in os.listdir(root_dir) 
                          if os.path.isdir(os.path.join(root_dir, d))]
        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_path = self.sequences[idx]
        
        # 1. Load the Trajectory Metadata (IMU data)
        imu_data = np.load(os.path.join(seq_path, 'imu_data.npy'))

        gt_data = np.load(os.path.join(seq_path, 'pos_data.npy'))
            
        # 2. Load Video
        video_path = os.path.join(seq_path, "trajectory_video.mp4")
            
        # 3. Extract IMU data (Accel and Gyro) 
        # Assumes trajectory_data is a list of dicts with 'imu' key
        imu_tensor = torch.tensor(imu_data, dtype=torch.float32) # [Seq_Len*10, 6]

        gt_tensor = torch.tensor(gt_data, dtype=torch.float32) # [Seq_Len, 7]

        return video_path, imu_tensor, gt_tensor

class DeepVIORandomDataset(Dataset):
    def __init__(self, root_dir, dataset_type):
        """
        Args:
            root_dir (string): Directory with all the sequence subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        if dataset_type == 'train':
            self.len = 8
        else:
            self.len = 6
        

        # List all sequence folders (e.g., 'seq_0', 'seq_1', etc.)
        self.sequence_dict = {}
        self.sequence_len = []
        for d in os.listdir(root_dir):
            sub_d = os.path.join(root_dir, d)
            if os.path.isdir(sub_d):
                self.sequence_dict[d] = [os.path.join(sub_d, traj_d) for traj_d in os.listdir(sub_d) 
                                            if os.path.isdir(os.path.join(sub_d, traj_d))]
                if d == "TrajectoriesLines": 
                    if self.dataset_type == 'train': [self.sequence_len.append(d+'.'+str(i)) for i in range(3)]
                    else: [self.sequence_len.append(d+'.'+str(i)) for i in range(2)]
                elif d == "TrajectoriesCirclesHeight" and self.dataset_type == 'train': [self.sequence_len.append(d+'.'+str(i)) for i in range(2)]
                else: self.sequence_len.append(d)
            
        self.sequence_dict
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        seq_type = self.sequence_len[idx]

        if "TrajectoriesLines" in seq_type and self.dataset_type == 'train':
            total_lines = len(self.sequence_dict["TrajectoriesLines"])
            line_split = int(np.ceil(total_lines/3))
            i_set = int(seq_type.split('.')[-1])
            traj_max = (i_set+1)*line_split if (i_set+1)*line_split<total_lines else None
            seq_trajs = self.sequence_dict["TrajectoriesLines"][i_set*line_split:traj_max]
        elif "TrajectoriesLines" in seq_type:
            total_lines = len(self.sequence_dict["TrajectoriesLines"])
            line_split = int(np.ceil(total_lines/2))
            i_set = int(seq_type.split('.')[-1])
            traj_max = (i_set+1)*line_split if (i_set+1)*line_split<total_lines else None
            seq_trajs = self.sequence_dict["TrajectoriesLines"][i_set*line_split:traj_max]
        elif "TrajectoriesCirclesHeight" in seq_type and self.dataset_type == 'train':
            total_lines = len(self.sequence_dict["TrajectoriesCirclesHeight"])
            line_split = int(np.ceil(total_lines/2))
            i_set = int(seq_type.split('.')[-1])
            traj_max = (i_set+1)*line_split if (i_set+1)*line_split<total_lines else None
            seq_trajs = self.sequence_dict["TrajectoriesCirclesHeight"][i_set*line_split:traj_max]
        else:
            seq_trajs = self.sequence_dict[seq_type]
        
        seq_idx = np.random.randint(len(seq_trajs))

        seq_path = seq_trajs[seq_idx]
        
        # 1. Load the Trajectory Metadata (IMU data)
        imu_data = np.load(os.path.join(seq_path, 'imu_data.npy'))

        gt_data = np.load(os.path.join(seq_path, 'pos_data.npy'))
            
        # 2. Load Video
        video_path = os.path.join(seq_path, "trajectory_video.mp4")
            
        # 3. Extract IMU data (Accel and Gyro) 
        # Assumes trajectory_data is a list of dicts with 'imu' key
        imu_tensor = torch.tensor(imu_data, dtype=torch.float32) # [Seq_Len*10, 6]

        gt_tensor = torch.tensor(gt_data, dtype=torch.float32) # [Seq_Len, 7]

        start_t = 0 if "TrajectoriesLines" in seq_type else np.random.randint(600) 

        return video_path, imu_tensor[start_t*10:(start_t+400)*10], gt_tensor[start_t:start_t+400], start_t