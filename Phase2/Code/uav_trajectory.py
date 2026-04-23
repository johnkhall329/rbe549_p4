import numpy as np
import matplotlib.pyplot as plt

class UAVTrajectoryGenerator:
    def __init__(self, gravity=9.81):
        self.gravity = gravity

    def generate_figure8(self, duration, frequency, radius_x=5.0, radius_y=5.0, z_height=10.0, speed=0.5):
        """
        Generates a physically plausible Figure-8 trajectory.
        
        :param duration: Total time in seconds
        :param frequency: Sampling frequency in Hz (e.g., 24, 30, 60 for Blender)
        :param radius_x: Amplitude of the X-axis motion
        :param radius_y: Amplitude of the Y-axis motion
        :param z_height: Hover height in meters
        :param speed: Angular speed multiplier (higher = faster, requires more tilt)
        :return: List of dictionaries containing the state at each timestep
        """
        dt = 1.0 / frequency
        times = np.arange(0, duration, dt)
        states = []

        for t in times:
            # 1. Calculate Position
            x = radius_x * np.sin(speed * t)
            y = radius_y * np.sin(2 * speed * t)
            z = z_height

            # 2. Calculate Velocity (First derivative of position)
            vx = radius_x * speed * np.cos(speed * t)
            vy = 2 * radius_y * speed * np.cos(2 * speed * t)
            vz = 0

            # 3. Calculate Acceleration (Second derivative of position)
            ax = -radius_x * speed**2 * np.sin(speed * t)
            ay = -4 * radius_y * speed**2 * np.sin(2 * speed * t)
            az = 0

            # 4. Apply Differential Flatness to find Roll and Pitch
            # The thrust vector must counteract gravity AND provide the required acceleration
            thrust_vec = np.array([ax, ay, az + self.gravity])
            
            # The drone's local Z-axis aligns with the thrust vector
            z_body = thrust_vec / np.linalg.norm(thrust_vec)

            # For a camera facing the floor, we'll point the nose (yaw) in the direction of travel
            yaw = np.arctan2(vy, vx)
            
            # Temporary X-axis based purely on desired yaw
            x_c = np.array([np.cos(yaw), np.sin(yaw), 0])

            # Calculate Body Y-axis (Right) and actual Body X-axis (Forward)
            y_body = np.cross(z_body, x_c)
            y_body = y_body / np.linalg.norm(y_body)
            x_body = np.cross(y_body, z_body)

            # Create the Rotation Matrix [X, Y, Z]
            R = np.column_stack((x_body, y_body, z_body))

            # Extract Euler Angles (Roll, Pitch, Yaw) assuming ZYX rotation order
            pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
            roll = np.arctan2(R[2, 1], R[2, 2])
            actual_yaw = np.arctan2(R[1, 0], R[0, 0])

            # Append the state
            states.append({
                'time': round(t, 4),
                'x': x,
                'y': y,
                'z': z,
                'roll': roll,     # In radians
                'pitch': pitch,   # In radians
                'yaw': actual_yaw # In radians
            })

        return states

    def visualize_trajectory_3d(self, states):
        """
        Plots the UAV's 3D path and camera orientation vectors.
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Extract positions
        xs = [state['x'] for state in states]
        ys = [state['y'] for state in states]
        zs = [state['z'] for state in states]

        # Plot the main trajectory line
        ax.plot(xs, ys, zs, label='UAV Trajectory', color='royalblue', linewidth=2)

        # Plot camera direction vectors (Quiver plot)
        # We sample a subset of states to avoid visual clutter on the graph
        sample_rate = max(1, len(states) // 30) 
        
        qx, qy, qz = [], [], []
        u, v, w = [], [], []

        for i in range(0, len(states), sample_rate):
            state = states[i]
            qx.append(state['x'])
            qy.append(state['y'])
            qz.append(state['z'])
            
            roll = state['roll']
            pitch = state['pitch']
            yaw = state['yaw']

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

# Example usage if run directly
if __name__ == "__main__":
    generator = UAVTrajectoryGenerator()
    # Generate a 20-second trajectory at 24 Hz (Standard Blender film frame rate)
    trajectory = generator.generate_figure8(duration=20, frequency=24, speed=0.4)
    
    generator.visualize_trajectory_3d(trajectory)