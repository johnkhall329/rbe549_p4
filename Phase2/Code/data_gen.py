import os
import pickle
import subprocess
import argparse
from uav_trajectory import UAVTrajectoryGenerator
from imu_gen import generate_imu_data
import numpy as np
import random

def generate_start_end_points(min_dist=1.0):
    # Define your workspace bounds
    # x: 0-10, y: 0-10, z: 1-10
    low = np.array([-5, -5, 3])
    high = np.array([5, 5, 8])
    
    while True:
        # Generate random start and end points within bounds
        start_point = np.random.uniform(low, high)
        end_point = np.random.uniform(low, high)
        
        # Calculate Euclidean distance
        dist = np.linalg.norm(end_point - start_point)
        
        # Only return if the length constraint is satisfied
        if dist >= min_dist:
            return start_point, end_point


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender_path", default="/Downloads/blender-5.1.0-linux-x64/blender")
    parser.add_argument("--base_blender_scene", default="Phase2/Blender/test_scene.blend")
    parser.add_argument("--output_dir", default="Phase2/Data/Trajectories")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--num_samples", default=200)
    parser.add_argument("--texture_dir", default="Phase2/Data/Redlands - Packing House District/Images")
    args = parser.parse_args()

    # Create output directory if it doesn't exist [cite: 74, 85]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    generator = UAVTrajectoryGenerator()
    
    # generate list of randomly sampled textures
    all_files = os.listdir(args.texture_dir)
    sampled_paths = random.choices(all_files, k=args.num_samples)
    
    for i, path in enumerate(sampled_paths):
        
        os.makedirs(args.output_dir + f'/{i}_traj', exist_ok=True)
        # 1. Generate Trajectory and IMU Data [cite: 10, 31]
        s, e = generate_start_end_points()
        trajectory, imu_data, gt_data = generator.generate_polynomial_line(duration=5, frequency=1000, start=s, end=e)
        # trajectory, imu_data, gt_data = generator.generate_circle_changing_height(duration=5, frequency=100, z_base=8)

        # 2. Save to Pickle and Numpy [cite: 4]
        with open(args.output_dir + f'/{i}_traj/imu_data.npy', 'wb') as f:
            np.save(f, imu_data)

        with open(args.output_dir + f'/{i}_traj/pos_data.npy', 'wb') as f:
            np.save(f, gt_data)
        
        with open(args.output_dir + f'/{i}_traj/trajectory.pkl', 'wb') as f:
            pickle.dump(trajectory, f)

        # 3. Call Blender [cite: 16, 42]
        cmd = [
            os.path.expanduser("~") + args.blender_path,
            args.base_blender_scene,
            "-b" if args.headless else "", # Background flag [cite: 43, 70]
            "-P", "Phase2/Code/blender.py",
            "--", args.output_dir, os.path.join(args.texture_dir, path), f"{i}_traj"
        ]
        # # Remove empty strings if not headless
        # cmd = [c for c in cmd if c]
        
        print(f"Starting Blender render for {len(trajectory)} frames...")
        subprocess.run(cmd)

if __name__ == "__main__":
    main()


