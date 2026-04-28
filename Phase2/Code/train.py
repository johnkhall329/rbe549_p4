import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse

from Network import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):
    model = DeepVIO(model_type=args.model_type)
    model.to(device)
    abs_traj = None

    epochs = args.epochs
    batch_size = args.batch_size

    optimizer = torch.optim.AdamW(model.parameters(), args.l_rate)

    for epoch_i in tqdm(range(epochs)):
        model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=int, default=2, 
        help='0: VO, 1: IO, 2: VIO.')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--l_rate', type=float, default=1e-4)
    args = parser.parse_args()

    train(args)