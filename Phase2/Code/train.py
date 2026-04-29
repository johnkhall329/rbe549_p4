import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from dataloader import DeepVIODataset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchcodec.decoders import VideoDecoder

from Network import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 30

def train(args):
    epochs = args.epochs
    traj_set = args.traj_set
    
    # Initialize Data
    # Initialize the dataset

    # Define basic image transformations
    data_transforms = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=0.5, std=0.5),
        transforms.Resize((520, 960)),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = DeepVIODataset(root_dir="Phase2/Data/Trajectories", transform=data_transforms)

    # Initialize the DataLoader
    # batch_first=True is standard for your VINet LSTM training 
    dataloader = DataLoader(dataset, batch_size=args.traj_set, shuffle=True, num_workers=2)

    # Initialize Model
    model = DeepVIO(model_type=args.model_type)
    model.to(device)
    abs_traj = None

    optimizer = torch.optim.AdamW(model.parameters(), args.l_rate)

    for epoch_i in tqdm(range(epochs)):
        model.train()

        for i, (video_paths, imu, gt) in enumerate(dataloader):
            
            # images shape: [Batch, Seq_Len, C, H, W]
            # imu shape: [Batch, Seq_Len*10, 6]
            # gt shape: [Batch, Seq_Len, 7]

            decoders = [VideoDecoder(path) for path in video_paths]

            imu = imu.to(device)
            gt = gt.to(device)
            print(f"Batch {i} - Images: {data_transforms(decoders[0][0]).shape}, IMU: {imu.shape}, GT: {gt.shape}")

            start_pos = gt[:,[0]]
            traj_pos = start_pos
            for j in range(decoders[0].metadata.num_frames - 1):
                curr_img_pairs = torch.stack([data_transforms(decoders[d][j:j+2]) for d in range(len(decoders))])
                curr_imu_data = imu[:, j*10:(j+1)*10]
                gt_data = gt[:, j:j+2] - traj_pos
                curr_img_pairs = curr_img_pairs.to(device)
                out = model(curr_img_pairs, curr_imu_data, traj_pos)
                # convert se3 to SE3 for loss and loop input ...




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=int, default=2, 
        help='0: VO, 1: IO, 2: VIO.')
    parser.add_argument('--traj_set', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--l_rate', type=float, default=1e-4)
    args = parser.parse_args()

    train(args)