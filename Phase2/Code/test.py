import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from dataloader import DeepVIODataset, DeepVIORandomDataset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from scipy.spatial.transform import Rotation as Rot

try:
    from torchcodec.decoders import VideoDecoder
except:
    from torchcodec.decoders import SimpleVideoDecoder as VideoDecoder

from Network import *
from transform_utils import process_output, get_twist, relative_start
from traj_plot import plot_traj

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loss(output_twist, output_pose, gt_twist, gt_pose, global_weight, rot_weight=1.0, quat_weight=2.5 ):
    v_loss = F.l1_loss(output_twist[:,:3], gt_twist[:,:3])
    omega_loss = F.l1_loss(output_twist[:,3:], gt_twist[:,3:])
    twist_loss = v_loss + rot_weight*omega_loss
    print(f"twist_loss: {twist_loss} gt twist norm: {gt_twist.norm()} out twist norm: {output_twist.norm()}")

    pos_loss = F.mse_loss(output_pose[:,0,:3], gt_pose[:,0,:3])
    quat_loss = torch.mean(quat_weight*(1 - torch.linalg.vecdot(output_pose[:,0, 3:], gt_pose[:,0,3:])))
    global_loss = pos_loss+quat_loss

    total_loss = (1-global_weight)*twist_loss + global_weight*global_loss

    return total_loss, twist_loss, global_loss


def test(args):    
    # Initialize Data
    # Initialize the dataset

    # Define basic image transformations
    data_transforms = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=0.5, std=0.5),
        transforms.Resize((520, 960)),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = DeepVIORandomDataset(root_dir="Phase2/Data/TestTrajectories", dataset_type="test")
    # dataset = DeepVIODataset(root_dir="Phase2/Data/TestTrajectories/")

    # Initialize the DataLoader
    # batch_first=True is standard for your VINet LSTM training 
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # Initialize Model
    model = DeepVIO(model_type=args.model_type)
    model.load_state_dict(torch.load(args.checkpoint_path+args.model_name, weights_only=True)['model_state_dict'])
    model.to(device)

    model_name = args.model_name.split('.')[0]
    
    model.eval()

    for traj_set_i, (video_paths, imu, gt, start_t) in enumerate(dataloader):
        
        # images shape: [Batch, Seq_Len, C, H, W]
        # imu shape: [Batch, Seq_Len*10, 6]
        # gt shape: [Batch, Seq_Len, 7]
        
        decoders = [VideoDecoder(path) for path in video_paths]

        traj_name = video_paths[0].split('/')[-3] +'_' + video_paths[0].split('/')[-2]

        output_dir = args.output_path+model_name+'/'+traj_name+'/'
        os.makedirs(output_dir, exist_ok=True)

        imu = imu.to(device)
        gt = gt.to(device)
        # print(f"Batch {i} - Images: {data_transforms(decoders[0][0]).shape}, IMU: {imu.shape}, GT: {gt.shape}")

        start_pos = gt[:,[0]]
        traj_pos = relative_start(start_pos,start_pos)

        total_loss = 0
        total_twist_loss = 0
        total_global_loss = 0
        
        output_poses = np.zeros((decoders[0].metadata.num_frames,8))
        gt_poses = np.zeros((decoders[0].metadata.num_frames,8))

        output_poses[0,1:] = traj_pos[0,0,[0,1,2,4,5,6,3]].detach().cpu().numpy() # switch real component to end
        gt_poses[0,1:] = traj_pos[0,0,[0,1,2,4,5,6,3]].detach().cpu().numpy()

        times = np.linspace(0, decoders[0].metadata.num_frames/100, decoders[0].metadata.num_frames, endpoint=False)
        output_poses[:,0] = times
        gt_poses[:,0] = times

        hidden_state = None
        for j in tqdm(range(decoders[0].metadata.num_frames - 1), desc="Sequence"):
            curr_img_pairs = torch.stack([data_transforms(decoders[d][j:j+2]) for d in range(len(decoders))])
            curr_imu_data = imu[:, j*10:(j+1)*10]
            gt_data = relative_start(gt[:, j:j+2], start_pos)
            curr_img_pairs = curr_img_pairs.to(device)

            with torch.no_grad():
                out_twist, hidden_state = model(curr_img_pairs, curr_imu_data, traj_pos, hidden_state)
            # convert se3 to SE3 for loss and loop input ...
            new_pose = process_output(out_twist, traj_pos)
            gt_twist = get_twist(gt_data)
            traj_loss, twist_loss, global_loss  = loss(out_twist, new_pose, gt_twist, gt_data[:, [1], :], 0.5)

            total_loss += traj_loss
            total_twist_loss += twist_loss
            total_global_loss += global_loss

            traj_pos = new_pose.detach()

            np_pose = traj_pos.cpu().numpy()
            output_poses[j+1,1:] = np_pose[0,0,[0,1,2,4,5,6,3]] # switch real component to end
            gt_poses[j+1,1:] = gt_data[0,1,[0,1,2,4,5,6,3]].detach().cpu().numpy()
        

        print(f"Total Loss: {total_loss/decoders[0].metadata.num_frames}")
        print(f"Twist Loss: {total_twist_loss/decoders[0].metadata.num_frames}")
        print(f"Pose Loss: {total_global_loss/decoders[0].metadata.num_frames}")

        np.savetxt(output_dir+'stamped_traj_estimate.txt', output_poses, header="time x y z qx qy qz qw")
        np.savetxt(output_dir+'stamped_groundtruth.txt', gt_poses, header="time x y z qx qy qz qw")

        plot_traj(gt_poses, output_poses, times, model_name+' '+traj_name)

                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=int, default=1, 
        help='0: VO, 1: IO, 2: VIO.')
    parser.add_argument('--log_path',default="./Phase2/Logs/",help="logs path")
    parser.add_argument('--output_path',default="./Phase2/Output/",help="logs path")
    parser.add_argument('--run_name', default="test",help="folder to store images")
    parser.add_argument('--checkpoint_path',default="./Phase2/Checkpoints/",help="checkpoints path")
    parser.add_argument('--model_name',default="batch_data_IOFinal.ckpt",help="checkpoint model name")
    args = parser.parse_args()

    test(args)