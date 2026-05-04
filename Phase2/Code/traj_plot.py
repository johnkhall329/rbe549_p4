import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_traj(gt_poses, output_poses, times, name):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gt_poses[:,1], gt_poses[:,2], gt_poses[:,3], label='Ground Truth Trajectory', color='g', linewidth=2)
    est_points = output_poses[:,1:4].reshape(-1,1,3)
    segments = np.concatenate([est_points[:-1], est_points[1:]], axis=1)
    lc = Line3DCollection(segments, cmap='viridis')
    lc.set_array(times)
    lc.set_linewidth(2)
    lc.set_label('Estimated Trajectory')
    line = ax.add_collection3d(lc)
    ax.set_title(name)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_xlim(np.min([gt_poses[:,1],output_poses[:,1]]), np.max([gt_poses[:,1],output_poses[:,1]]))
    ax.set_ylim(np.min([gt_poses[:,2],output_poses[:,2]]), np.max([gt_poses[:,2],output_poses[:,2]]))
    ax.set_zlim(np.min([gt_poses[:,3],output_poses[:,3]]), np.max([gt_poses[:,3],output_poses[:,3]]))
    
    fig2 = plt.figure(figsize=(12, 8))
    gt_rpy = np.zeros((gt_poses.shape[0],3))
    for i, quat in enumerate(gt_poses[:,4:]):
        rot = Rot.from_quat(quat)
        gt_rpy[i] = rot.as_euler('xyz', degrees=True)
    est_rpy = np.zeros((gt_poses.shape[0],3))
    for i, quat in enumerate(output_poses[:,4:]):
        rot = Rot.from_quat(quat)
        est_rpy[i] = rot.as_euler('xyz', degrees=True)
    plt.plot(times, gt_rpy[:,0], label="GT Roll")
    plt.plot(times, gt_rpy[:,1], label="GT Pitch")
    plt.plot(times, gt_rpy[:,2], label="GT Yaw")

    plt.plot(times, est_rpy[:,0], label="Est Roll")
    plt.plot(times, est_rpy[:,1], label="Est Pitch")
    plt.plot(times, est_rpy[:,2], label="Est Yaw")
    plt.xlabel("Time")
    plt.ylabel("Angle (Deg)")
    plt.legend()

    plt.show()