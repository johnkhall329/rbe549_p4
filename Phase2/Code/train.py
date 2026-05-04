import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from dataloader import DeepVIODataset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np

from traj_plot import plot_traj

try:
    from torchcodec.decoders import VideoDecoder
except:
    from torchcodec.decoders import SimpleVideoDecoder as VideoDecoder

from Network import *
from transform_utils import process_output, get_twist, relative_start, standardize_quaternion, quaternion_multiply, quaternion_invert, quaternion_to_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"USING DEVICE: {device}")

def loss(output_twist, output_pose, gt_twist, gt_pose, global_weight, rot_weight=1.0, quat_weight=10.0):
    v_loss = F.l1_loss(output_twist[:,:3], gt_twist[:,:3])
    omega_loss = F.l1_loss(output_twist[:,3:], gt_twist[:,3:])
    twist_loss = v_loss + (rot_weight*omega_loss)

    pos_loss = F.mse_loss(output_pose[:,0,:3], gt_pose[:,0,:3])
    
    out_q = standardize_quaternion(F.normalize(output_pose[:,0, 3:],dim=1))
    gt_q = standardize_quaternion(F.normalize(gt_pose[:,0, 3:],dim=1))
    quat_loss = torch.mean(quat_weight*(1 - torch.linalg.vecdot(out_q, gt_q)))
    global_loss = pos_loss+quat_loss

    total_loss = ((1-global_weight)*twist_loss) + (global_weight*global_loss)

    return total_loss, twist_loss, global_loss

def geodesic_loss(q_pred: torch.Tensor, q_true: torch.Tensor) -> torch.Tensor:
    """
    Computes the geodesic loss between two quaternions.
    Input shapes: (Batch, 4) in (w, x, y, z) format.
    """
    R_s = quaternion_to_matrix(q_pred)
    R_t = quaternion_to_matrix(q_true)
    
    # Calculate R_s @ R_t^T for each item in the batch
    R_error = torch.bmm(R_s, R_t.transpose(1, 2))
    
    # Trace of the error matrix: sum of diagonal elements
    trace = R_error[:, 0, 0] + R_error[:, 1, 1] + R_error[:, 2, 2]
    
    # Clamp input to arccos to avoid NaNs due to floating point precision errors
    argument = (trace - 1.0) / 2.0
    argument = torch.clamp(argument, -1.0 + 1e-7, 1.0 - 1e-7)
    
    # Compute the angle (geodesic distance in radians)
    theta = torch.acos(argument)
    return torch.mean(theta)

def trajectory_geodesic_loss(
    delta_pose, 
    new_pose, 
    target_delta_pose, 
    gt_absolute_pose, 
    global_weight, 
    rot_weight=1.0
):
    """
    Args:
        delta_pose: Predicted changes (Batch, 7) -> [dx, dy, dz, dqw, dqx, dqy, dqz]
        new_pose: Absolute integrated pose (Batch, 7) -> [x, y, z, qw, qx, qy, qz]
        target_delta_pose: GT changes (Batch, 7) -> [dx_gt, dy_gt, dz_gt, dqw_gt, dqx_gt, dqz_gt]
        gt_absolute_pose: GT absolute pose at t=2 (Batch, 7) -> [x_gt, y_gt, z_gt, qw_gt, qx_gt, qz_gt]
        global_weight: Float scaling between delta loss and global loss.
        rot_weight: Scale for orientation errors.
    """
    # -------------------------------------------------------------------------
    # 1. DELTA LOSS (Replaces old twist_loss)
    # -------------------------------------------------------------------------
    # Position Delta Loss (Translation)
    delta_pos_loss = F.mse_loss(delta_pose[:, :3], target_delta_pose[:, :3])
    
    # Rotation Delta Loss (Orientation in radians)
    delta_rot_loss = geodesic_loss(delta_pose[:, 3:], target_delta_pose[:, 3:])
    
    # Total local delta step loss
    delta_loss = delta_pos_loss + (rot_weight * delta_rot_loss)

    # -------------------------------------------------------------------------
    # 2. GLOBAL TRAJECTORY LOSS
    # -------------------------------------------------------------------------
    # Global Position Error
    global_pos_loss = F.mse_loss(new_pose[:, :3], gt_absolute_pose[:, :3])
    
    # Global Rotation Error (Comparing integrated trajectory to ground truth)
    global_rot_loss = geodesic_loss(new_pose[:, 3:], gt_absolute_pose[:, 3:])
    
    # Total global state loss
    global_loss = global_pos_loss + (rot_weight * global_rot_loss)

    # -------------------------------------------------------------------------
    # 3. BALANCED TOTAL LOSS
    # -------------------------------------------------------------------------
    total_loss = ((1.0 - global_weight) * delta_loss) + (global_weight * global_loss)

    return total_loss, delta_loss, global_loss

def add_poses(delta_pose, original_pos):
    delta_xyz = delta_pose[:, :3]  # Shape: (Batch, 3)
    delta_quat = delta_pose[:, 3:] # Shape: (Batch, 4) -> format: (w, x, y, z)

    # 2. Unpack the current/ground-truth trajectory pose
    # traj_pos format: [x, y, z, qw, qx, qy, qz]
    traj_xyz = original_pos[:, -1, :3]  # Shape: (Batch, 3)
    traj_quat = original_pos[:, -1, 3:] # Shape: (Batch, 4) -> format: (w, x, y, z)

    # 3. Compute absolute position via addition
    new_xyz = traj_xyz + delta_xyz

    # 4. Compute absolute rotation via quaternion multiplication
    # Normalizing both quaternions prevents numerical drift over time
    delta_quat_norm = torch.nn.functional.normalize(delta_quat, p=2, dim=-1)
    traj_quat_norm = torch.nn.functional.normalize(traj_quat, p=2, dim=-1)

    # Compose the rotations
    new_quat = quaternion_multiply(traj_quat_norm, delta_quat_norm)

    # 5. Concatenate back into the 7D absolute pose
    new_pose = torch.cat((new_xyz, new_quat), dim=-1)
    return new_pose

def find_delta_poses(sequential_poses):
    pose_1 = sequential_poses[:, 0, :]  # Shape: (Batch, 7) -> [x1, y1, z1, qw1, qx1, qy1, qz1]
    pose_2 = sequential_poses[:, 1, :]  # Shape: (Batch, 7) -> [x2, y2, z2, qw2, qx2, qy2, qz2]

    # Split both poses into position and quaternion components
    pos_1, quat_1 = pose_1[:, :3], pose_1[:, 3:]
    pos_2, quat_2 = pose_2[:, :3], pose_2[:, 3:]

    target_delta_pos = pos_2 - pos_1  # Shape: (Batch, 3)

    quat_1_norm = torch.nn.functional.normalize(quat_1, p=2, dim=-1)
    quat_2_norm = torch.nn.functional.normalize(quat_2, p=2, dim=-1)

    # Invert q1 to get q1_inverse
    quat_1_inv = quaternion_invert(quat_1_norm)

    # Multiply to get the delta
    target_delta_quat = quaternion_multiply(quat_1_inv, quat_2_norm) # Shape: (Batch, 4)
    # Combine into the 7-vector ground truth target
    gt_delta_pose = torch.cat((target_delta_pos, target_delta_quat), dim=-1) # Shape: (Batch, 7)
    return gt_delta_pose


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
    ])

    dataset = DeepVIODataset(root_dir=args.traj_path_train, transform=data_transforms)
    val_dataset = DeepVIODataset(root_dir=args.traj_path_val, transform=data_transforms)

    # Initialize the DataLoader
    # batch_first=True is standard for your VINet LSTM training 
    dataloader = DataLoader(dataset, batch_size=args.traj_set, shuffle=True, num_workers=2, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.traj_set, shuffle=True, num_workers=2, drop_last=False)

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # Initialize Model
    model = DeepVIO(model_type=args.model_type)
    model.to(device)

    writer = SummaryWriter(args.log_path+args.run_name)

    optimizer = torch.optim.Adam(model.parameters(), args.l_rate)
    global_weight_init = 0.5
    global_weight_final = 0.99
    init_x = -np.log(global_weight_init)
    final_x = -np.log(global_weight_final)

    scale_x = (final_x - init_x)/(epochs - 1)

    for epoch_i in tqdm(range(epochs), desc="Epochs"):

        # find the global and f2f weights for this epoch
        global_weight = np.exp(-(init_x + (scale_x*epoch_i)))
        # print(global_weight)

        print(f"Epoch: {epoch_i + 1}")

        # TRAINING
        track_trajectory = True if epoch_i < 5 else False
        epoch_total_loss_train = 0
        epoch_twist_loss_train = 0
        epoch_global_loss_train = 0
        model.train()
        for traj_set_i, (video_paths, imu, gt) in enumerate(dataloader):
            
            # images shape: [Batch, Seq_Len, C, H, W]
            # imu shape: [Batch, Seq_Len*10, 6]
            # gt shape: [Batch, Seq_Len, 7]

            decoders = [VideoDecoder(path) for path in video_paths]
            sequence_length_train = decoders[0].metadata.num_frames

            imu = imu.to(device)
            gt = gt.to(device)
            # print(f"Batch {i} - Images: {data_transforms(decoders[0][0]).shape}, IMU: {imu.shape}, GT: {gt.shape}")

            start_pos = gt[:,[0]]
            traj_pos = start_pos 
            # traj_pos = relative_start(start_pos,start_pos) #GT_TEST

            total_loss = 0
            total_twist_loss = 0
            total_global_loss = 0

            window_total_loss = 0
            window_twist_loss = 0
            window_global_loss = 0

            hidden_state = None

            if epoch_i+1 == args.epochs and args.display:
                output_poses = np.zeros((len(dataloader), decoders[0].metadata.num_frames,8))
                gt_poses = np.zeros((len(dataloader), decoders[0].metadata.num_frames,8))

                output_poses[:,0,1:] = traj_pos[:,0,[0,1,2,4,5,6,3]].detach().cpu().numpy() # switch real component to end
                gt_poses[:,0,1:] = traj_pos[:,0,[0,1,2,4,5,6,3]].detach().cpu().numpy()
                
                times = np.linspace(0, decoders[0].metadata.num_frames/100, decoders[0].metadata.num_frames, endpoint=False)
                output_poses[:, :,0] = times
                gt_poses[:, :,0] = times

            for j in tqdm(range(sequence_length_train - 1), desc="Sequence_Train"):
                curr_img_pairs = torch.stack([data_transforms(decoders[d][j:j+2]) for d in range(len(decoders))])
                curr_imu_data = imu[:, j*10:(j+1)*10]
                # gt_data = relative_start(gt[:, j:j+2], start_pos) #GT_TEST
                gt_data = gt[:, j:j+2] 
                curr_img_pairs = curr_img_pairs.to(device)

                delta_pose, hidden_state = model(curr_img_pairs, curr_imu_data, traj_pos, hidden_state) # now returns a 7 vector, to be interpreted as pos, quat
                new_pose = add_poses(delta_pose, traj_pos)
                target_delta_pos = find_delta_poses(gt_data)
                traj_loss, twist_loss, global_loss = trajectory_geodesic_loss(delta_pose, new_pose, target_delta_pos, gt_data[:,1], global_weight)

                # convert se3 to SE3 for loss and loop input ...
                # new_pose = process_output(out_twist, traj_pos)
                # gt_twist = get_twist(gt_data)
                # traj_loss, twist_loss, global_loss  = loss(out_twist, new_pose, gt_twist, gt_data[:, [1], :], global_weight)

                if epoch_i+1 == args.epochs and args.dipslay:
                    output_poses[traj_set_i*args.traj_set:(traj_set_i+1)*args.traj_set,j+1,1:] = new_pose[:,0,[0,1,2,4,5,6,3]].detach().cpu().numpy() # switch real component to end
                    gt_poses[traj_set_i*args.traj_set:(traj_set_i+1)*args.traj_set,j+1,1:] = gt_data[:,1,[0,1,2,4,5,6,3]].detach().cpu().numpy()

                window_total_loss += traj_loss
                window_twist_loss += twist_loss
                window_global_loss += global_loss

                if (j+1) % args.window_size == 0:
                    (window_total_loss/args.window_size).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    total_loss += window_total_loss.item()
                    total_twist_loss += window_twist_loss.item()
                    total_global_loss += window_global_loss.item()

                    window_total_loss = 0
                    window_twist_loss = 0
                    window_global_loss = 0
                    
                    hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())

                    traj_pos = new_pose.detach().unsqueeze(1)
                else:
                    traj_pos = new_pose.unsqueeze(1)

                if track_trajectory:
                    traj_pos = gt_data[:, [1], :] # GT TEST

            writer.add_scalar("Total_trajectory_loss", total_loss/sequence_length_train, traj_set_i+(epoch_i*len(dataloader)))
            writer.add_scalar("F2F_trajectory_loss", total_twist_loss/sequence_length_train, traj_set_i+(epoch_i*len(dataloader)))
            writer.add_scalar("Global_trajectory_loss", total_global_loss/sequence_length_train, traj_set_i+(epoch_i*len(dataloader)))
            epoch_total_loss_train += total_loss/sequence_length_train
            epoch_global_loss_train += total_global_loss/sequence_length_train
            epoch_twist_loss_train += total_twist_loss/sequence_length_train
            
        
        writer.add_scalar("Total_epoch_train_loss", epoch_total_loss_train/len(dataloader), (epoch_i + 1))
        writer.add_scalar("F2F_epoch_train_loss", epoch_twist_loss_train/len(dataloader), (epoch_i + 1))
        writer.add_scalar("Global_epoch_train_loss", epoch_global_loss_train/len(dataloader), (epoch_i + 1))
        print(f"Total Train Loss: {epoch_total_loss_train/len(dataloader)}")
        print(f"F2F Train Loss: {epoch_twist_loss_train/len(dataloader)}")
        print(f"Global Train Loss: {epoch_global_loss_train/len(dataloader)}")


        # VALIDATION
        epoch_total_loss_val = 0
        epoch_twist_loss_val = 0
        epoch_global_loss_val = 0

        model.eval()
        with torch.no_grad():
            for traj_set_i, (video_paths, imu, gt) in enumerate(val_dataloader):
                
                # images shape: [Batch, Seq_Len, C, H, W]
                # imu shape: [Batch, Seq_Len*10, 6]
                # gt shape: [Batch, Seq_Len, 7]

                decoders = [VideoDecoder(path) for path in video_paths]
                sequence_length_val = decoders[0].metadata.num_frames

                imu = imu.to(device)
                gt = gt.to(device)
                # print(f"Batch {i} - Images: {data_transforms(decoders[0][0]).shape}, IMU: {imu.shape}, GT: {gt.shape}")

                start_pos = gt[:,[0]]
                # traj_pos = relative_start(start_pos,start_pos)
                traj_pos = start_pos # GT_TEST

                total_loss = 0
                total_twist_loss = 0
                total_global_loss = 0

                hidden_state = None
                for j in tqdm(range(sequence_length_val - 1), desc="Sequence_Val"):
                    curr_img_pairs = torch.stack([data_transforms(decoders[d][j:j+2]) for d in range(len(decoders))])
                    curr_imu_data = imu[:, j*10:(j+1)*10]
                    # gt_data = relative_start(gt[:, j:j+2], start_pos) #GT_TEST
                    gt_data = gt[:, j:j+2] 
                    curr_img_pairs = curr_img_pairs.to(device)


                    delta_pose, hidden_state = model(curr_img_pairs, curr_imu_data, traj_pos, hidden_state) # now returns a 7 vector, to be interpreted as pos, quat
                    new_pose = add_poses(delta_pose, traj_pos)
                    target_delta_pos = find_delta_poses(gt_data)
                    traj_loss, twist_loss, global_loss = trajectory_geodesic_loss(delta_pose, new_pose, target_delta_pos, gt_data[:,1], global_weight)
                    # convert se3 to SE3 for loss and loop input ...
                    # new_pose = process_output(out_twist, traj_pos)
                    # gt_twist = get_twist(gt_data)
                    # traj_loss, twist_loss, global_loss = loss(out_twist, new_pose, gt_twist, gt_data[:, [1], :], global_weight)

                    total_loss += traj_loss.item()
                    total_twist_loss += twist_loss.item()
                    total_global_loss += global_loss.item()

                    traj_pos = new_pose.detach().unsqueeze(1) #GT_TEST
                    # traj_pos = gt_data[:, [1], :]
                    hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())

                epoch_total_loss_val += total_loss/sequence_length_val
                epoch_global_loss_val += total_global_loss/sequence_length_val
                epoch_twist_loss_val += total_twist_loss/sequence_length_val
            

            writer.add_scalar("Total_epoch_val_loss", epoch_total_loss_val/len(val_dataloader), (epoch_i + 1))
            writer.add_scalar("Twist_epoch_val_loss", epoch_twist_loss_val/len(val_dataloader), (epoch_i + 1))
            writer.add_scalar("Pose_epoch_val_loss", epoch_global_loss_val/len(val_dataloader), (epoch_i + 1))
            print(f"Total Val Loss: {epoch_total_loss_val/len(val_dataloader)}")
            print(f"F2F Val Loss: {epoch_twist_loss_val/len(val_dataloader)}")
            print(f"Global Val Loss: {epoch_global_loss_val/len(val_dataloader)}")
                
        # SAVE MODEL
        if epoch_i % args.save_ckpt_epoch == 0 and epoch_i > 0:
            SaveName = (
                args.checkpoint_path
                + args.run_name
                + str(epoch_i)
                + ".ckpt"
            )

            torch.save(
                {
                    "epoch": epoch_i,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), 
                }, 
                SaveName, 
            )
            print("\n" + SaveName + " Model Saved...")

    # SAVE FINAL MODEL
    SaveName = (
        args.checkpoint_path
        + args.run_name
        + "Final.ckpt"
    )

    torch.save(
        {
            "epoch": epoch_i,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), 
        }, 
        SaveName, 
    )
    print("\n" + SaveName + " Model Saved...")

    if args.display:
        for k in range(gt_poses.shape[0]):
            plot_traj(gt_poses[k], output_poses[k], times, "test plot")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=int, default=2, 
        help='0: VO, 1: IO, 2: VIO.')
    parser.add_argument('--traj_set', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--l_rate', type=float, default=1e-3)
    parser.add_argument('--log_path',default="./Phase2/Logs/",help="logs path")
    parser.add_argument('--run_name', default="morequatagain",help="folder to store images")
    parser.add_argument('--checkpoint_path',default="./Phase2/Checkpoints/",help="checkpoints path")
    parser.add_argument('--save_ckpt_epoch',default=5,help="num of iteration to save checkpoint")
    parser.add_argument('--display', type=bool, default=False,help="Display final trajectories")
    parser.add_argument('--traj_path_train',default="Phase2/Data/Trajectories")
    parser.add_argument('--traj_path_val',default="Phase2/Data/TrajectoriesVal")
    args = parser.parse_args()

    train(args)