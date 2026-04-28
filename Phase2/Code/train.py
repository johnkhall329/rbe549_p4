import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from dataloader import DeepVIODataset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from Network import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 300

def train(args):
    # Initialize Data
    # Initialize the dataset

    # Define basic image transformations
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = DeepVIODataset(root_dir="Phase2/Data/Trajectories", sequence_length=SEQUENCE_LENGTH, transform=data_transforms)

    # Initialize the DataLoader
    # batch_first=True is standard for your VINet LSTM training 
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    # Initialize Model
    model = DeepVIO(model_type=args.model_type)
    model.to(device)
    abs_traj = None

    epochs = args.epochs
    batch_size = args.batch_size

    optimizer = torch.optim.AdamW(model.parameters(), args.l_rate)

    for epoch_i in tqdm(range(epochs)):
        model.train()

        for i, (images, imu, gt) in enumerate(dataloader):
            # images shape: [Batch, Seq_Len, C, H, W]
            # imu shape: [Batch, Seq_Len*10, 6]
            # gt shape: [Batch, Seq_Len, 7]
            print(f"Batch {i} - Images: {images.shape}, IMU: {imu.shape}, GT: {gt.shape}")

            for j in range(SEQUENCE_LENGTH - 1):
                curr_image_pair = images[:, j:j+2]
                curr_imu_data = imu[:, j*10:(j+1)*10]

                gt_data = gt[:, j:j+2]




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=int, default=2, 
        help='0: VO, 1: IO, 2: VIO.')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--l_rate', type=float, default=1e-4)
    args = parser.parse_args()

    train(args)