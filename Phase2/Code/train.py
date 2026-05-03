import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from dataloader import DeepVIODataset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np

try:
    from torchcodec.decoders import VideoDecoder
except:
    from torchcodec.decoders import SimpleVideoDecoder as VideoDecoder

from Network import *
from transform_utils import process_output, get_twist, relative_start

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"USING DEVICE: {device}")

def loss(output_twist, output_pose, gt_twist, gt_pose, global_weight, rot_weight=1.0, quat_weight=2.5 ):
    v_loss = F.l1_loss(output_twist[:,:3], gt_twist[:,:3])
    omega_loss = F.l1_loss(output_twist[:,3:], gt_twist[:,3:])
    twist_loss = v_loss + rot_weight*omega_loss

    pos_loss = F.mse_loss(output_pose[:,0,:3], gt_pose[:,0,:3])
    # GEMINI SUGGESTION, EVALUATE FURTHER: Absolute value handles the double-cover property of quaternions properly across the batch
    # PREVIOUSLY WAS: quat_loss = torch.mean(quat_weight * (1 - torch.linalg.vecdot(output_pose[:, 0, 3:], gt_pose[:, 0, 3:])))
    q_pred = F.normalize(output_pose[:, 0, 3:], p=2, dim=-1)
    q_gt = F.normalize(gt_pose[:, 0, 3:], p=2, dim=-1)
    quat_loss = torch.mean(quat_weight * (1 - torch.abs(torch.linalg.vecdot(q_pred, q_gt))))

    global_loss = pos_loss+quat_loss

    total_loss = (1-global_weight)*twist_loss + global_weight*global_loss

    return total_loss, twist_loss, global_loss


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
    dataloader = DataLoader(dataset, batch_size=args.traj_set, shuffle=True, num_workers=2, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.traj_set, shuffle=True, num_workers=2, drop_last=True)

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # Initialize Model
    model = DeepVIO(model_type=args.model_type)
    model.to(device)

    writer = SummaryWriter(args.log_path+args.run_name)

    optimizer = torch.optim.AdamW(model.parameters(), args.l_rate)
    global_weight_init = 0.01
    global_weight_final = 0.9
    init_x = -np.log(global_weight_init)
    final_x = -np.log(global_weight_final)

    scale_x = (final_x - init_x)/(epochs - 1)

    for epoch_i in tqdm(range(epochs), desc="Epochs"):
        model.train()

        # find the global and f2f weights for this epoch
        global_weight = np.exp(-(init_x + (scale_x*epoch_i)))
        # print(global_weight)

        print(f"Epoch: {epoch_i + 1}")

        model.hidden_state = None
        epoch_total_loss_train = 0
        epoch_twist_loss_train = 0
        epoch_global_loss_train = 0
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
            traj_pos = relative_start(start_pos,start_pos)

            total_loss = 0
            total_twist_loss = 0
            total_global_loss = 0

            window_total_loss = 0
            window_twist_loss = 0
            window_global_loss = 0

            for j in tqdm(range(sequence_length_train - 1), desc="Sequence_Train"):
                curr_img_pairs = torch.stack([data_transforms(decoders[d][j:j+2]) for d in range(len(decoders))])
                curr_imu_data = imu[:, j*10:(j+1)*10]
                gt_data = relative_start(gt[:, j:j+2], start_pos)
                curr_img_pairs = curr_img_pairs.to(device)


                out_twist = model(curr_img_pairs, curr_imu_data, traj_pos)
                # convert se3 to SE3 for loss and loop input ...
                new_pose = process_output(out_twist, traj_pos)
                gt_twist = get_twist(gt_data)
                traj_loss, twist_loss, global_loss  = loss(out_twist, new_pose, gt_twist, gt_data[:, [1], :], global_weight)

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

                traj_pos = new_pose.detach()

            writer.add_scalar("Total_trajectory_loss", total_loss/sequence_length_train, traj_set_i+(epoch_i*len(dataloader)))
            writer.add_scalar("Twist_trajectory_loss", twist_loss/sequence_length_train, traj_set_i+(epoch_i*len(dataloader)))
            writer.add_scalar("Pose_trajectory_loss", global_loss/sequence_length_train, traj_set_i+(epoch_i*len(dataloader)))
            epoch_total_loss_train += total_loss/sequence_length_train
            epoch_global_loss_train += global_loss/sequence_length_train
            epoch_twist_loss_train += twist_loss/sequence_length_train
            
        
        writer.add_scalar("Total_epoch_train_loss", epoch_total_loss_train/len(dataloader), (epoch_i + 1))
        writer.add_scalar("Twist_epoch_train_loss", epoch_twist_loss_train/len(dataloader), (epoch_i + 1))
        writer.add_scalar("Pose_epoch_train_loss", epoch_global_loss_train/len(dataloader), (epoch_i + 1))
        print(f"Total Train Loss: {epoch_total_loss_train/len(dataloader)}")
        print(f"Twist Train Loss: {epoch_twist_loss_train/len(dataloader)}")
        print(f"Pose Train Loss: {epoch_global_loss_train/len(dataloader)}")


        # VALIDATION
        epoch_total_loss_val = 0
        epoch_twist_loss_val = 0
        epoch_global_loss_val = 0

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
                traj_pos = relative_start(start_pos,start_pos)

                total_loss = 0
                total_twist_loss = 0
                total_global_loss = 0

                for j in tqdm(range(sequence_length_val - 1), desc="Sequence_Val"):
                    curr_img_pairs = torch.stack([data_transforms(decoders[d][j:j+2]) for d in range(len(decoders))])
                    curr_imu_data = imu[:, j*10:(j+1)*10]
                    gt_data = relative_start(gt[:, j:j+2], start_pos)
                    curr_img_pairs = curr_img_pairs.to(device)


                    out_twist = model(curr_img_pairs, curr_imu_data, traj_pos)
                    # convert se3 to SE3 for loss and loop input ...
                    new_pose = process_output(out_twist, traj_pos)
                    gt_twist = get_twist(gt_data)
                    traj_loss, twist_loss, global_loss = loss(out_twist, new_pose, gt_twist, gt_data[:, [1], :], global_weight)

                    total_loss += traj_loss.item()
                    total_twist_loss += twist_loss.item()
                    total_global_loss += global_loss.item()

                    traj_pos = new_pose.detach()

                epoch_total_loss_val += total_loss/sequence_length_val
                epoch_global_loss_val += global_loss/sequence_length_val
                epoch_twist_loss_val += twist_loss/sequence_length_val
            

            writer.add_scalar("Total_epoch_val_loss", epoch_total_loss_val/len(val_dataloader), (epoch_i + 1))
            writer.add_scalar("Twist_epoch_val_loss", epoch_twist_loss_val/len(val_dataloader), (epoch_i + 1))
            writer.add_scalar("Pose_epoch_val_loss", epoch_global_loss_val/len(val_dataloader), (epoch_i + 1))
            print(f"Total Val Loss: {epoch_total_loss_val/len(val_dataloader)}")
            print(f"Twist Val Loss: {epoch_twist_loss_val/len(val_dataloader)}")
            print(f"Pose Val Loss: {epoch_global_loss_val/len(val_dataloader)}")
                

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
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=int, default=2, 
        help='0: VO, 1: IO, 2: VIO.')
    parser.add_argument('--traj_set', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--l_rate', type=float, default=1e-4)
    parser.add_argument('--log_path',default="./Phase2/Logs/",help="logs path")
    parser.add_argument('--run_name', default="test",help="folder to store images")
    parser.add_argument('--checkpoint_path',default="./Phase2/Checkpoints/",help="checkpoints path")
    parser.add_argument('--save_ckpt_epoch',default=5,help="num of iteration to save checkpoint")
    parser.add_argument('--traj_path_train',default="Phase2/Data/Trajectories")
    parser.add_argument('--traj_path_val',default="Phase2/Data/TrajectoriesVal")
    args = parser.parse_args()

    train(args)