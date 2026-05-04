import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from dataloader import DeepVIODataset
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
from train import add_poses, find_delta_poses, trajectory_geodesic_loss, loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    dataset = DeepVIODataset(root_dir="Phase2/Data/Trajectories", transform=data_transforms)

    # Initialize the DataLoader
    # batch_first=True is standard for your VINet LSTM training 
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

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

    for traj_set_i, (video_paths, imu, gt) in enumerate(dataloader):
        
        # images shape: [Batch, Seq_Len, C, H, W]
        # imu shape: [Batch, Seq_Len*10, 6]
        # gt shape: [Batch, Seq_Len, 7]
        
        decoders = [VideoDecoder(path) for path in video_paths]

        traj_name = video_paths[0].split('/')[-2]

        output_dir = args.output_path+model_name+'/'+traj_name+'/'
        os.makedirs(output_dir, exist_ok=True)

        imu = imu.to(device)
        gt = gt.to(device)
        # print(f"Batch {i} - Images: {data_transforms(decoders[0][0]).shape}, IMU: {imu.shape}, GT: {gt.shape}")

        start_pos = gt[:,[0]]
        # traj_pos = relative_start(start_pos,start_pos)
        traj_pos = start_pos # GT_TEST

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
            # gt_data = relative_start(gt[:, j:j+2], start_pos) #GT_TEST
            gt_data = gt[:, j:j+2] 
            curr_img_pairs = curr_img_pairs.to(device)

            with torch.no_grad():
                delta_pose, hidden_state = model(curr_img_pairs, curr_imu_data, traj_pos, hidden_state) # now returns a 7 vector, to be interpreted as pos, quat
            new_pose = add_poses(delta_pose, traj_pos)
            target_delta_pos = find_delta_poses(gt_data)
            traj_loss, twist_loss, global_loss = trajectory_geodesic_loss(delta_pose, new_pose, target_delta_pos, gt_data[:,1], 0.01)
                    # convert se3 to SE3 for loss and loop input ...
            # with torch.no_grad():
            #     out_twist, hidden_state = model(curr_img_pairs, curr_imu_data, traj_pos, hidden_state)
            # # convert se3 to SE3 for loss and loop input ...
            # new_pose = process_output(out_twist, traj_pos)
            # gt_twist = get_twist(gt_data)
            # traj_loss, twist_loss, global_loss  = loss(out_twist, new_pose, gt_twist, gt_data[:, [1], :], 0.5)

            total_loss += traj_loss
            total_twist_loss += twist_loss
            total_global_loss += global_loss

            traj_pos = new_pose.detach().unsqueeze(1)

            np_pose = traj_pos.cpu().numpy()
            output_poses[j+1,1:] = np_pose[0,0,[0,1,2,4,5,6,3]] # switch real component to end
            gt_poses[j+1,1:] = gt_data[0,1,[0,1,2,4,5,6,3]].detach().cpu().numpy()
        

        print(f"Total Loss: {total_loss/decoders[0].metadata.num_frames}")
        print(f"Twist Loss: {total_twist_loss/decoders[0].metadata.num_frames}")
        print(f"Pose Loss: {total_global_loss/decoders[0].metadata.num_frames}")

        np.savetxt(output_dir+'stamped_traj_estimate.txt', output_poses, header="time x y z qx qy qz qw")
        np.savetxt(output_dir+'stamped_groundtruth.txt', gt_poses, header="time x y z qx qy qz qw")

        plot_traj(gt_poses, output_poses, times, model_name+' '+traj_name)

        # fig = plt.figure(figsize=(12, 8))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(gt_poses[:,1], gt_poses[:,2], gt_poses[:,3], label='Ground Truth Trajectory', color='g', linewidth=2)
        # est_points = output_poses[:,1:4].reshape(-1,1,3)
        # segments = np.concatenate([est_points[:-1], est_points[1:]], axis=1)
        # lc = Line3DCollection(segments, cmap='viridis')
        # lc.set_array(times)
        # lc.set_linewidth(2)
        # lc.set_label('Estimated Trajectory')
        # line = ax.add_collection3d(lc)
        # ax.set_title(model_name+' '+traj_name)
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # ax.legend()
        # ax.set_xlim(np.min([gt_poses[:,1],output_poses[:,1]]), np.max([gt_poses[:,1],output_poses[:,1]]))
        # ax.set_ylim(np.min([gt_poses[:,2],output_poses[:,2]]), np.max([gt_poses[:,2],output_poses[:,2]]))
        # ax.set_zlim(np.min([gt_poses[:,3],output_poses[:,3]]), np.max([gt_poses[:,3],output_poses[:,3]]))
        
        # fig2 = plt.figure(figsize=(12, 8))
        # gt_rpy = np.zeros((gt_poses.shape[0],3))
        # for i, quat in enumerate(gt_poses[:,4:]):
        #     rot = Rot.from_quat(quat)
        #     gt_rpy[i] = rot.as_euler('xyz', degrees=True)
        # est_rpy = np.zeros((gt_poses.shape[0],3))
        # for i, quat in enumerate(output_poses[:,4:]):
        #     rot = Rot.from_quat(quat)
        #     est_rpy[i] = rot.as_euler('xyz', degrees=True)
        # plt.plot(times, gt_rpy[:,0], label="GT Roll")
        # plt.plot(times, gt_rpy[:,1], label="GT Pitch")
        # plt.plot(times, gt_rpy[:,2], label="GT Yaw")

        # plt.plot(times, est_rpy[:,0], label="Est Roll")
        # plt.plot(times, est_rpy[:,1], label="Est Pitch")
        # plt.plot(times, est_rpy[:,2], label="Est Yaw")
        # plt.xlabel("Time")
        # plt.ylabel("Angle (Deg)")
        # plt.legend()

        # plt.show()

                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=int, default=2, 
        help='0: VO, 1: IO, 2: VIO.')
    parser.add_argument('--log_path',default="./Phase2/Logs/",help="logs path")
    parser.add_argument('--output_path',default="./Phase2/Output/",help="logs path")
    parser.add_argument('--run_name', default="test",help="folder to store images")
    parser.add_argument('--checkpoint_path',default="./Phase2/Checkpoints/",help="checkpoints path")
    parser.add_argument('--model_name',default="new_loss_test315.ckpt",help="checkpoint model name")
    args = parser.parse_args()

    test(args)