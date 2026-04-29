import bpy
import pickle
import sys
import os

# Get arguments after "--" [cite: 40]
argv = sys.argv
argv = argv[argv.index("--") + 1:]
output_dir = argv[0]
texture_path = argv[1]
traj_folder = argv[2]

def apply_texture_to_plane(plane_name, image_path):
    # 1. Get the plane object
    plane = bpy.data.objects.get(plane_name)
    if not plane:
        print(f"Error: Plane '{plane_name}' not found.")
        return

    # 2. Create a new material
    mat = bpy.data.materials.new(name="FloorMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    # 3. Clear existing nodes and add necessary ones
    nodes.clear()
    node_tex = nodes.new(type='ShaderNodeTexImage')
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_out = nodes.new(type='ShaderNodeOutputMaterial')

    # 4. Load the image
    try:
        img = bpy.data.images.load(image_path)
        node_tex.image = img
    except Exception as e:
        print(f"Failed to load image: {e}")
        return

    # 5. Link the nodes together
    links = mat.node_tree.links
    links.new(node_tex.outputs['Color'], node_bsdf.inputs['Base Color'])
    links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])
    
    # 6. Assign the material to the plane
    if plane.data.materials:
        plane.data.materials[0] = mat
    else:
        plane.data.materials.append(mat)


def render_trajectory():
    # Load the trajectory data
    with open(os.path.join(output_dir, traj_folder, 'trajectory.pkl'), 'rb') as f:
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
        render_path = os.path.join(output_dir, traj_folder, frame_name)
        bpy.context.scene.render.filepath = render_path
        
        # Render the frame 
        print(f"Rendering {frame_name}...")
        bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    # --- Start of your existing script logic ---
    # Set the path to your texture image
    # texture_path = "/home/wyatt/Documents/CV/P4/rbe549_p4/Phase2/Data/Textures/coast_land_rocks_01/coast_land_rocks_01_primary.png"
    apply_texture_to_plane("Plane", texture_path)
    render_trajectory()