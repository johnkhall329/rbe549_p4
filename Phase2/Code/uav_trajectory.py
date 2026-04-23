import numpy as np
import matplotlib.pyplot as plt

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

        # Rotation Matrix to Euler Angles (ZYX order) 
        R = np.column_stack((x_body, y_body, z_body))
        pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
        roll = np.arctan2(R[2, 1], R[2, 2])
        actual_yaw = np.arctan2(R[1, 0], R[0, 0])

        return {
            'time': round(t, 4),
            'x': pos[0], 'y': pos[1], 'z': pos[2],
            'roll': roll, 'pitch': pitch, 'yaw': actual_yaw
        }

    def generate_line(self, duration, frequency, start=(0,0,1), end=(10,10,10)):
        dt = 1.0 / frequency
        times = np.arange(0, duration, dt)
        start, end = np.array(start), np.array(end)
        vel_vec = (end - start) / duration
        
        return [self._compute_uav_state(t, start + vel_vec * t, vel_vec, np.zeros(3)) for t in times]

    def generate_circle(self, duration, frequency, radius=5.0, z_height=10.0, speed=0.5):
        dt = 1.0 / frequency
        times = np.arange(0, duration, dt)
        states = []
        for t in times:
            pos = np.array([radius * np.cos(speed * t), radius * np.sin(speed * t), z_height])
            vel = np.array([-radius * speed * np.sin(speed * t), radius * speed * np.cos(speed * t), 0])
            acc = np.array([-radius * speed**2 * np.cos(speed * t), -radius * speed**2 * np.sin(speed * t), 0])
            states.append(self._compute_uav_state(t, pos, vel, acc))
        return states
    
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
        
        for t in times:
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
            states.append(self._compute_uav_state(t, pos, vel, acc))
            
        return states

    def generate_figure8(self, duration, frequency, radius_x=5.0, radius_y=5.0, z_height=10.0, speed=0.5):
        dt = 1.0 / frequency
        times = np.arange(0, duration, dt)
        states = []
        for t in times:
            pos = np.array([radius_x * np.sin(speed * t), radius_y * np.sin(2 * speed * t), z_height])
            vel = np.array([radius_x * speed * np.cos(speed * t), 2 * radius_y * speed * np.cos(2 * speed * t), 0])
            acc = np.array([-radius_x * speed**2 * np.sin(speed * t), -4 * radius_y * speed**2 * np.sin(2 * speed * t), 0])
            states.append(self._compute_uav_state(t, pos, vel, acc))
        return states

    def generate_square(self, duration, frequency, side_length=10.0, z_height=10.0):
        dt = 1.0 / frequency
        times = np.arange(0, duration, dt)
        states = []
        half_s = side_length / 2
        # Define corners
        pts = [(-half_s, -half_s), (half_s, -half_s), (half_s, half_s), (-half_s, half_s), (-half_s, -half_s)]
        
        for t in times:
            # Determine which leg of the square we are on
            progress = (t / duration) % 1.0
            idx = int(progress * 4)
            leg_t = (progress * 4) % 1.0
            
            p1, p2 = np.array(pts[idx]), np.array(pts[idx+1])
            pos_2d = p1 + (p2 - p1) * leg_t
            pos = np.array([pos_2d[0], pos_2d[1], z_height])
            # For square, velocity is constant per leg, acceleration is zero (except at corners)
            vel = np.append((p2 - p1) * (4 / duration), 0)
            states.append(self._compute_uav_state(t, pos, vel, np.zeros(3)))
        return states

def visualize_trajectory_3d(states):
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
    trajectory_8 = generator.generate_figure8(duration=20, frequency=24, speed=0.4)
    trajectory_l = generator.generate_line(duration=20, frequency=24)
    trajectory_c = generator.generate_circle(duration=20, frequency=24)
    trajectory_s = generator.generate_square(duration=20, frequency=24)
    trajectory_cc = generator.generate_circle_changing_height(duration=20, frequency=24)
    

    visualize_trajectory_3d(trajectory_cc)