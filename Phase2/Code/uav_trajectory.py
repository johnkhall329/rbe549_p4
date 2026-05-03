import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

IMAGE_CAPTURE_MULTIPLIER = 10

from imu_gen import generate_imu_data_np

class UAVTrajectoryGenerator:
    def __init__(self, gravity=9.81):
        self.gravity = gravity

    def _compute_uav_state(self, t, pos, vel, acc, yaw=None):
        """
        Derives full UAV state (including Euler angles) from position derivatives
        using the property of differential flatness[cite: 8, 9].
        """
        # The thrust vector must counteract gravity and provide horizontal acceleration [cite: 6]
        thrust_vec = np.array([acc[0], acc[1], acc[2] + self.gravity])
        z_body = thrust_vec / np.linalg.norm(thrust_vec)

        # If no yaw is provided, point the nose in the direction of travel [cite: 20]
        if yaw is None:
            yaw = np.arctan2(vel[1], vel[0]) if np.linalg.norm(vel[:2]) > 0.1 else 0

        x_c = np.array([np.cos(yaw), np.sin(yaw), 0])
        y_body = np.cross(z_body, x_c)
        y_body = y_body / np.linalg.norm(y_body)
        x_body = np.cross(y_body, z_body)

        R = np.column_stack((x_body, y_body, z_body))

        big_r = Rot.from_matrix(R)
        roll, pitch, actual_yaw = big_r.as_euler('xyz')

        accel_body = R.T @ thrust_vec

        # r = Rot.from_euler('zyx', [actual_yaw, pitch, roll], degrees=False)
        quat = big_r.as_quat(scalar_first=True) # Returns [w, x, y, z]

        return {
            'time': round(t, 4),
            'xyz': [pos[0], pos[1], pos[2]],
            'rpy': [roll, pitch, actual_yaw], 
            'accel': accel_body,
            'quat': quat
        }

    def generate_polynomial_line(self, duration, frequency, start=(0,0,1), end=(10,10,10), start_time=0.0, yaw=None):
        dt = 1.0 / frequency
        times = np.arange(0, duration, dt)
        datapoints = len(times)
        start, end = np.array(start), np.array(end)
        dist_vec = end - start
        if yaw is None:
            yaw = np.arctan2(dist_vec[1], dist_vec[0])
        
        # Pre-allocate arrays for efficiency
        imu_acc = np.zeros((datapoints, 3))
        angle_vec = np.zeros((datapoints, 3))
        all_pos_vec = np.zeros((datapoints//IMAGE_CAPTURE_MULTIPLIER, 3))
        quat_pos_vec = np.zeros((datapoints//IMAGE_CAPTURE_MULTIPLIER, 4))

        ret_list = []
        for i, t in enumerate(times):
            # Normalized time (0 to 1)
            tau = t / duration
            
            # Quintic polynomial for smooth transition (Minimum Jerk Trajectory)
            # s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            # First derivative (velocity scaling)
            ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
            # Second derivative (acceleration scaling)
            dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / (duration**2)

            # Calculate 3D state based on the polynomial scaling
            pos_vec = start + dist_vec * s
            vel_vec = dist_vec * ds
            acc_vec = dist_vec * dds

            # Compute physics-based UAV state (roll, pitch, yaw) [cite: 524]
            state = self._compute_uav_state(t + start_time, pos_vec, vel_vec, acc_vec, yaw=yaw)

            # Store data for return
            imu_acc[i] = state['accel']
            angle_vec[i] = np.array(state['rpy'])

            # Selective capture for Blender visualization [cite: 526, 548]
            if i % IMAGE_CAPTURE_MULTIPLIER == 0:
                # Convert ZYX Euler angles to Quaternion [cite: 546, 547]

                all_pos_vec[i//IMAGE_CAPTURE_MULTIPLIER] = pos_vec
                quat_pos_vec[i//IMAGE_CAPTURE_MULTIPLIER] = state['quat']

                ret_list.append(state)

        # Compute angular velocity (gyro) from orientation changes 
        gyro_vec = np.gradient(angle_vec, dt, axis=0)
        
        # Generate IMU format for training [cite: 509]
        imu_data = generate_imu_data_np(imu_acc, gyro_vec, frequency)

        # Ground Truth data (Position + Quaternion)
        gt_data = np.zeros((datapoints//IMAGE_CAPTURE_MULTIPLIER, 7))
        gt_data[:, :3] = all_pos_vec
        gt_data[:, 3:] = quat_pos_vec

        return ret_list, imu_data, gt_data

    def generate_circle(self, duration, frequency, radius=5.0, z_height=10.0, speed=0.5):
        dt = 1.0 / frequency
        times = np.arange(0, duration, dt)
        states = []

        datapoints = len(times)
        # Pre-allocate arrays for efficiency
        imu_acc = np.zeros((datapoints, 3))
        angle_vec = np.zeros((datapoints, 3))
        all_pos_vec = np.zeros((datapoints//IMAGE_CAPTURE_MULTIPLIER, 3))
        quat_pos_vec = np.zeros((datapoints//IMAGE_CAPTURE_MULTIPLIER, 4))


        for i, t in enumerate(times):
            pos = np.array([radius * np.cos(speed * t), radius * np.sin(speed * t), z_height])
            vel = np.array([-radius * speed * np.sin(speed * t), radius * speed * np.cos(speed * t), 0])
            acc = np.array([-radius * speed**2 * np.cos(speed * t), -radius * speed**2 * np.sin(speed * t), 0])
            state = self._compute_uav_state(t, pos, vel, acc)

            # Store data for return
            imu_acc[i] = state['accel']
            angle_vec[i] = np.array(state['rpy'])

            # Selective capture for Blender visualization [cite: 526, 548]
            if i % IMAGE_CAPTURE_MULTIPLIER == 0:   

                all_pos_vec[i//IMAGE_CAPTURE_MULTIPLIER] = pos
                quat_pos_vec[i//IMAGE_CAPTURE_MULTIPLIER] = state['quat']

                states.append(state)

        # Compute angular velocity (gyro) from orientation changes 
        # wrap it so that imu gradient is reasonable
        wrap_angle_vec = angle_vec
        wrap_angle_vec[:, 2] = np.unwrap(wrap_angle_vec[:, 2])
        gyro_vec = np.gradient(wrap_angle_vec, dt, axis=0)
        
        # Generate IMU format for training [cite: 509]
        imu_data = generate_imu_data_np(imu_acc, gyro_vec, frequency)

        # Ground Truth data (Position + Quaternion)
        gt_data = np.zeros((datapoints//IMAGE_CAPTURE_MULTIPLIER, 7))
        gt_data[:, :3] = all_pos_vec
        gt_data[:, 3:] = quat_pos_vec

        return states, imu_data, gt_data
    
    def generate_circle_changing_height(self, duration, frequency, radius=5.0, z_base=10.0, z_amplitude=2.0, speed=0.5, z_speed=1.0):
        """
        Generates a circular trajectory where the altitude oscillates sinusoidally.
        
        :param z_base: The average height (midpoint of oscillation)
        :param z_amplitude: How far above and below z_base the UAV travels
        :param z_speed: The frequency multiplier for height oscillation
        """
        dt = 1.0 / frequency
        times = np.arange(0, duration, dt)
        states = []


        datapoints = len(times)
        # Pre-allocate arrays for efficiency
        imu_acc = np.zeros((datapoints, 3))
        angle_vec = np.zeros((datapoints, 3))
        all_pos_vec = np.zeros((datapoints//IMAGE_CAPTURE_MULTIPLIER, 3))
        quat_pos_vec = np.zeros((datapoints//IMAGE_CAPTURE_MULTIPLIER, 4))
        
        for i, t in enumerate(times):
            # 1. Position: Standard circle for XY, Sine wave for Z
            pos = np.array([
                radius * np.cos(speed * t), 
                radius * np.sin(speed * t), 
                z_base + z_amplitude * np.sin(z_speed * t)
            ])
            
            # 2. Velocity: Derivatives of the position functions
            vel = np.array([
                -radius * speed * np.sin(speed * t), 
                radius * speed * np.cos(speed * t), 
                z_amplitude * z_speed * np.cos(z_speed * t)
            ])
            
            # 3. Acceleration: Second derivatives of the position functions
            acc = np.array([
                -radius * speed**2 * np.cos(speed * t), 
                -radius * speed**2 * np.sin(speed * t), 
                -z_amplitude * z_speed**2 * np.sin(z_speed * t)
            ])
            
            # Use the modular physics solver to calculate Euler angles [cite: 11, 27]
            state = self._compute_uav_state(t, pos, vel, acc)

            # Store data for return
            imu_acc[i] = state['accel']
            angle_vec[i] = np.array(state['rpy'])

            # Selective capture for Blender visualization [cite: 526, 548]
            if i % IMAGE_CAPTURE_MULTIPLIER == 0:   

                all_pos_vec[i//IMAGE_CAPTURE_MULTIPLIER] = pos
                quat_pos_vec[i//IMAGE_CAPTURE_MULTIPLIER] = state['quat']

                states.append(state)

        # Compute angular velocity (gyro) from orientation changes 
        # wrap it so that imu gradient is reasonable
        wrap_angle_vec = angle_vec
        wrap_angle_vec[:, 2] = np.unwrap(wrap_angle_vec[:, 2])
        gyro_vec = np.gradient(wrap_angle_vec, dt, axis=0)
        
        # Generate IMU format for training [cite: 509]
        imu_data = generate_imu_data_np(imu_acc, gyro_vec, frequency)

        # Ground Truth data (Position + Quaternion)
        gt_data = np.zeros((datapoints//IMAGE_CAPTURE_MULTIPLIER, 7))
        gt_data[:, :3] = all_pos_vec
        gt_data[:, 3:] = quat_pos_vec

        return states, imu_data, gt_data

    # def generate_figure8(self, duration, frequency, radius_x=5.0, radius_y=5.0, z_height=10.0, speed=0.5):
    #     dt = 1.0 / frequency
    #     times = np.arange(0, duration, dt)
    #     states = []
    #     for t in times:
    #         pos = np.array([radius_x * np.sin(speed * t), radius_y * np.sin(2 * speed * t), z_height])
    #         vel = np.array([radius_x * speed * np.cos(speed * t), 2 * radius_y * speed * np.cos(2 * speed * t), 0])
    #         acc = np.array([-radius_x * speed**2 * np.sin(speed * t), -4 * radius_y * speed**2 * np.sin(2 * speed * t), 0])
    #         states.append(self._compute_uav_state(t, pos, vel, acc))
    #     return states
    
    def generate_figure8(self, duration, frequency, radius_x=5.0, radius_y=5.0, z_height=10.0, speed=0.5):
        dt = 1.0 / frequency
        times = np.arange(0, duration, dt)
        states = []

        datapoints = len(times)
        # Pre-allocate arrays for efficiency
        imu_acc = np.zeros((datapoints, 3))
        angle_vec = np.zeros((datapoints, 3))
        
        # Calculate sizes for decimated capture
        capture_count = datapoints // IMAGE_CAPTURE_MULTIPLIER
        all_pos_vec = np.zeros((capture_count, 3))
        quat_pos_vec = np.zeros((capture_count, 4))

        for i, t in enumerate(times):
            # Figure 8 parametric equations (Lissajous curve)
            pos = np.array([radius_x * np.sin(speed * t), radius_y * np.sin(2 * speed * t), z_height])
            vel = np.array([radius_x * speed * np.cos(speed * t), 2 * radius_y * speed * np.cos(2 * speed * t), 0])
            acc = np.array([-radius_x * speed**2 * np.sin(speed * t), -4 * radius_y * speed**2 * np.sin(2 * speed * t), 0])
            
            state = self._compute_uav_state(t, pos, vel, acc)

            # Store full-frequency data for IMU processing
            imu_acc[i] = state['accel']
            angle_vec[i] = np.array(state['rpy'])

            # Selective capture for Blender visualization and GT
            if i % IMAGE_CAPTURE_MULTIPLIER == 0:
                idx = i // IMAGE_CAPTURE_MULTIPLIER
                if idx < capture_count:
                    all_pos_vec[idx] = pos
                    quat_pos_vec[idx] = state['quat']
                    states.append(state)

        # Compute angular velocity (gyro) from orientation changes 
        # Use np.copy to ensure we don't accidentally mutate state['rpy'] if they share memory
        wrap_angle_vec = np.copy(angle_vec)
        # Unwrap yaw (index 2) to prevent huge spikes in the gradient at +/- PI
        wrap_angle_vec[:, 2] = np.unwrap(wrap_angle_vec[:, 2])
        gyro_vec = np.gradient(wrap_angle_vec, dt, axis=0)
        
        # Generate IMU format (combines Accel and Gyro)
        imu_data = generate_imu_data_np(imu_acc, gyro_vec, frequency)

        # Ground Truth data (Position XYZ + Quaternion XYZW)
        gt_data = np.zeros((capture_count, 7))
        gt_data[:, :3] = all_pos_vec
        gt_data[:, 3:] = quat_pos_vec

        return states, imu_data, gt_data
    
    def generate_square(self, duration, frequency, side_length=5.0, z_height=6.0, constant_yaw=0.0):
        # Duration per leg
        leg_duration = duration / 4
        half_s = side_length / 2
        
        # Define corners (anticlockwise square)
        c = [
            (-half_s, -half_s, z_height),
            (half_s, -half_s, z_height),
            (half_s, half_s, z_height),
            (-half_s, half_s, z_height),
            (-half_s, -half_s, z_height)
        ]

        all_states = []
        all_imu = []
        all_gt = []

        for i in range(4):
            # Calculate start time for this leg to keep timestamps continuous
            current_start_time = i * leg_duration
            
            states, imu, gt = self.generate_polynomial_line(
                duration=leg_duration,
                frequency=frequency,
                start=c[i],
                end=c[i+1],
                start_time=current_start_time,
                yaw=constant_yaw # Force constant yaw as requested
            )
            
            all_states.extend(states)
            all_imu.append(imu)
            all_gt.append(gt)

        # Vertically stack the numpy arrays for IMU and Ground Truth
        final_imu = np.vstack(all_imu)
        final_gt = np.vstack(all_gt)

        return all_states, final_imu, final_gt
    
    def generate_start_end_points(self, min_dist=1.25, max_dist = 5):
        # Define your workspace bounds
        # x: 0-10, y: 0-10, z: 1-10
        low = np.array([-4, -4, 3])
        high = np.array([4, 4, 8])
        yaw = np.random.uniform(0, 2*np.pi)
        
        while True:
            # Generate random start and end points within bounds
            start_point = np.random.uniform(low, high)
            end_point = np.random.uniform(low, high)
            
            # Calculate Euclidean distance
            dist = np.linalg.norm(end_point - start_point)
            
            # Only return if the length constraint is satisfied
            if dist >= min_dist and dist <= max_dist:
                return start_point, end_point, 0.0, yaw

    def generate_circle_changing_height_params(self):
        radius = np.random.uniform(1.0, 3.5)
        z_base = np.random.uniform(3.5, 7.0)
        z_amplitude = np.random.uniform(0.5, 1.5)
        speed = np.random.uniform(0.3, 0.8)
        z_speed = np.random.uniform(0.5, 1.5)

        return radius, z_base, z_amplitude, speed, z_speed

    def generate_square_params(self):
        yaw_val = np.random.uniform(0, 2 * np.pi)
        z_base = np.random.uniform(3.5, 7.0)
        side_length = np.random.uniform(1.5, 5.0)        
        
        return side_length, z_base, yaw_val

    def generate_figure8_params(self):
        z_base = np.random.uniform(3.5, 7.0)
        speed = np.random.uniform(0.4, 1.5)
        radius_x = np.random.uniform(1.5, 3.5)
        radius_y = np.random.uniform(1.5, 3.5)
                
        return radius_x, radius_y, z_base, speed

    def generate_circle_params(self):
        radius = np.random.uniform(1.0, 3.5)
        z_base = np.random.uniform(3.5, 7.0)
        speed = np.random.uniform(0.3, 0.8)

        return radius, z_base, speed

    def get_random_trajectory_params(self, trajectory_type: str):
        """
        Looks up and executes the appropriate parameter generation function
        based on the provided string key.
        """
        # Map your string keys directly to the function objects
        function_map = {
            "line": self.generate_start_end_points,
            "circle_changing_height": self.generate_circle_changing_height_params,
            "square": self.generate_square_params,
            "figure8": self.generate_figure8_params,
            "circle": self.generate_circle_params
        }
        
        # Retrieve the function from the dictionary
        func = function_map.get(trajectory_type.lower())
        
        # Check if the string matched any valid key
        if func is None:
            valid_keys = list(function_map.keys())
            raise ValueError(f"Unknown trajectory type '{trajectory_type}'. Valid options are: {valid_keys}")
        
        # Call the retrieved function and return its result
        return func()
    
    def generate_random_trajectory(self, trajectory_type: str, duration: float, frequency: float):
        """
        Generates dynamic randomized parameters and returns the resulting trajectory data
        by mapping string keys to the corresponding generator methods.
        """
        
        # The dictionary maps strings to: (1) The specific randomizer, (2) The generator function to call
        trajectory_map = {
            "line": (self.generate_start_end_points, self.generate_polynomial_line),
            "circle": (self.generate_circle_params, self.generate_circle),
            "square": (self.generate_square_params, self.generate_square),
            "circle_changing_height": (self.generate_circle_changing_height_params, self.generate_circle_changing_height),
            "figure8": (self.generate_figure8_params, self.generate_figure8)
        }
        
        # Retrieve lookup tuple
        lookup = trajectory_map.get(trajectory_type.lower())
        
        if lookup is None:
            valid_types = list(trajectory_map.keys())
            raise ValueError(f"Unknown trajectory type '{trajectory_type}'. Valid types: {valid_types}")
        
        # Extract the random parameter generator and the target method
        param_func, target_method = lookup
        
        # Step 1: Generate the randomized parameters
        random_params = param_func()
        
        # Step 2: Combine duration and frequency with the randomized parameters
        all_params = [duration, frequency, *random_params]
        
        # Step 3: Unpack and call the specific trajectory generation method
        return target_method(*all_params)

def visualize_trajectory_3d(states):
    """
    Plots the UAV's 3D path and camera orientation vectors.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract positions
    xs = [state['xyz'][0] for state in states]
    ys = [state['xyz'][1] for state in states]
    zs = [state['xyz'][2] for state in states]

    # Plot the main trajectory line
    ax.plot(xs, ys, zs, label='UAV Trajectory', color='royalblue', linewidth=2)

    # Plot camera direction vectors (Quiver plot)
    # We sample a subset of states to avoid visual clutter on the graph
    sample_rate = max(1, len(states) // 30) 
    
    qx, qy, qz = [], [], []
    u, v, w = [], [], []

    for i in range(0, len(states), sample_rate):
        state = states[i]
        qx.append(state['xyz'][0])
        qy.append(state['xyz'][1])
        qz.append(state['xyz'][2])
        
        roll = state['rpy'][0]
        pitch = state['rpy'][1]
        yaw = state['rpy'][2]

        # Reconstruct the rotation matrices based on Euler angles (ZYX order)
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        
        # Combined rotation matrix
        R = R_z @ R_y @ R_x
        
        # Assuming the camera faces straight down relative to the drone body (Negative Z axis)
        camera_vector_local = np.array([0, 0, -1])
        
        # Transform local camera vector to world space
        camera_vector_world = R @ camera_vector_local
        
        u.append(camera_vector_world[0])
        v.append(camera_vector_world[1])
        w.append(camera_vector_world[2])

    # Add the arrows representing the camera's line of sight
    ax.quiver(qx, qy, qz, u, v, w, length=1.5, color='darkorange', label='Camera Facing Vector', normalize=True)

    # Graph Formatting
    ax.set_title('UAV 3D Trajectory & Camera Orientation')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.legend()

    # Enforce an equal aspect ratio so the Figure-8 doesn't look stretched
    max_range = np.array([max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)]).max() / 2.0
    mid_x = (max(xs)+min(xs)) * 0.5
    mid_y = (max(ys)+min(ys)) * 0.5
    mid_z = (max(zs)+min(zs)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.savefig('Phase2/Output/test.png')
    plt.show()

# Example usage if run directly
if __name__ == "__main__":
    generator = UAVTrajectoryGenerator()
    # Generate a 20-second trajectory at 24 Hz (Standard Blender film frame rate)
    # trajectory, imu_data, gt_data = generator.generate_figure8(duration=10, frequency=100, radius_x = 2.5, radius_y=2.5, z_height=6, speed=0.7)
    # trajectory, imu_data, gt_data = generator.generate_polynomial_line(duration=5, frequency=100, start=(-5,5,5), end = (-5,-5,6))
    # trajectory, imu_data, gt_data = generator.generate_circle(duration=5, frequency=100, radius = 3, z_height=5, speed=1.25)
    trajectory, imu_data, gt_data = generator.generate_square(duration=20, frequency=100, side_length=5, z_height=6, constant_yaw=np.pi)
    # trajectory, imu_data, gt_data = generator.generate_circle_changing_height(duration=10, frequency=100, radius=3, z_base=6, z_amplitude=0.4, speed=0.5, z_speed=4)

    visualize_trajectory_3d(trajectory)

    time = np.linspace(0, 5, len(imu_data))
    gt = gt_data

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Subplot 1: Gyroscope Data (XYZ rates)
    axes[0].plot(time, imu_data[:, 3], label='Gyro X')
    axes[0].plot(time, imu_data[:, 4], label='Gyro Y')
    axes[0].plot(time, imu_data[:, 5], label='Gyro Z')
    axes[0].set_ylabel('Angular Velocity (rad/s)')
    axes[0].set_title('IMU Gyroscope Data')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)

    # Subplot 2: Position over time
    axes[1].plot(time, imu_data[:, 0], label='X')
    axes[1].plot(time, imu_data[:, 1], label='Y')
    axes[1].plot(time, imu_data[:, 2], label='Z')
    axes[1].set_ylabel('Acc (m/s^2)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Ground Truth Acceleromer')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    # Subplot 1: Gyroscope Data (XYZ rates)
    # axes[0].plot(time, gt[:, 3], label='quat1')
    # axes[0].plot(time, gt[:, 4], label='quat2')
    # axes[0].plot(time, gt[:, 5], label='quat3')
    # axes[0].plot(time, gt[:, 6], label='quat4')
    # axes[0].set_ylabel('angle')
    # axes[0].set_title('gt ang')
    # axes[0].legend(loc='upper right')
    # axes[0].grid(True)

    # # Subplot 2: Position over time
    # axes[1].plot(time, gt[:, 0], label='X')
    # axes[1].plot(time, gt[:, 1], label='Y')
    # axes[1].plot(time, gt[:, 2], label='Z')
    # axes[1].set_ylabel('Acc (m/s^2)')
    # axes[1].set_xlabel('Time (s)')
    # axes[1].set_title('Ground Truth Position')
    # axes[1].legend(loc='upper right')
    # axes[1].grid(True)

    # plt.tight_layout()
    # plt.show()