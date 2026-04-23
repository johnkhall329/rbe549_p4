import bpy
import pickle
import sys
import os

# Get arguments after "--" [cite: 40]
argv = sys.argv
argv = argv[argv.index("--") + 1:]
pkl_path = argv[0]
output_dir = argv[1]

def render_trajectory(pkl_path, output_dir):
    # Load the trajectory data
    with open(pkl_path, 'rb') as f:
        trajectory = pickle.load(f)

    # Ensure we have the correct camera [cite: 82, 137]
    cam = bpy.data.objects.get('Camera')
    if not cam:
        print("Error: No object named 'Camera' found!")
        return

    bpy.context.scene.camera = cam
    # Set rotation mode to match trajectory generator 
    cam.rotation_mode = 'ZYX' 

    for i, state in enumerate(trajectory):
        # Update camera state [cite: 29, 45]
        cam.location = (state['x'], state['y'], state['z'])
        cam.rotation_euler = (state['roll'], state['pitch'], state['yaw'])
        
        # Set output path for this specific frame
        frame_name = f"frame_{i:04d}.png"
        render_path = os.path.join(output_dir, frame_name)
        bpy.context.scene.render.filepath = render_path
        
        # Render the frame 
        print(f"Rendering {frame_name}...")
        bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    render_trajectory(pkl_path, output_dir)