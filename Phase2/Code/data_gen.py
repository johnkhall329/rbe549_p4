import os
import pickle
import subprocess
import argparse
from uav_trajectory import UAVTrajectoryGenerator
from imu_gen import generate_imu_data
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender_path", default="/Downloads/blender-5.1.0-linux-x64/blender")
    parser.add_argument("--base_blender_scene", default="Phase2/Blender/test_scene.blend")
    parser.add_argument("--output_dir", default="Phase2/Output/Sample")
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    # Create output directory if it doesn't exist [cite: 74, 85]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 1. Generate Trajectory [cite: 10, 31]
    generator = UAVTrajectoryGenerator()
    trajectory = generator.generate_circle(duration=5, frequency=10)

    # Generate IMU Data
    imu_data = generate_imu_data(trajectory)
    with open(args.output_dir + '/imu_data.npy', 'wb') as f:
        np.save(f, imu_data)

    # 2. Save to Pickle [cite: 4]
    pkl_path = os.path.abspath("Phase2/Data/trajectory.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(trajectory, f)

    # 3. Call Blender [cite: 16, 42]
    cmd = [
        os.path.expanduser("~") + args.blender_path,
        args.base_blender_scene,
        "-b" if args.headless else "", # Background flag [cite: 43, 70]
        "-P", "Phase2/Code/blender.py",
        "--", pkl_path, args.output_dir
    ]
    # Remove empty strings if not headless
    cmd = [c for c in cmd if c]
    
    print(f"Starting Blender render for {len(trajectory)} frames...")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()


