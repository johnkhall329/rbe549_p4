import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torchvision.io as rv
from torchcodec.decoders import VideoDecoder

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
