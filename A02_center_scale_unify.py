
import open3d as o3d
import torch
import numpy as np
import random
import os

def set_seed(seed):
    # Fix the seed for Python's built-in random module
    random.seed(seed)

    # Fix the seed for NumPy
    np.random.seed(seed)

    # Fix the seed for PyTorch
    torch.manual_seed(seed)

    # If using CUDA (GPU), also fix the seed for the GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Ensure reproducibility for operations on CUDA (if used)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_seed_from_env(env_var='SEED'):
    seed = os.getenv(env_var)
    if seed is not None:
        return int(seed)
    raise ValueError(f"Environment variable {env_var} is not set")

# Load the seed from the environment variable
seed = load_seed_from_env()

# Set the seed
set_seed(seed)

# Save the seed to a file if needed
def save_seed(seed, file_path='seed.txt'):
    with open(file_path, 'w') as f:
        f.write(str(seed))

save_seed(seed)

# Example operations to demonstrate reproducibility
random_number = random.random()
numpy_array = np.random.rand(3)
torch_tensor = torch.rand(3)

print("Random number (random):", random_number)
print("Numpy array (numpy):", numpy_array)
print("Torch tensor (torch):", torch_tensor)
print("Used seed:", seed)


def scale_mesh(input_file, output_file, centroid, max_dist):
    """
    Scales an OBJ mesh based on a given centroid and maximum distance.

    Parameters:
    input_file (str): Path to the input OBJ file.
    output_file (str): Path where the scaled OBJ file will be saved.
    centroid (list or np.array): The centroid coordinates as [x, y, z].
    max_dist (float): Desired maximum distance from the centroid to any vertex in the mesh.

    Returns:
    None: The function saves the scaled mesh to the specified output file.
    """
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(input_file)

    scale_factor = 1/(max_dist*2)

    # Scale the mesh
    mesh.vertices = o3d.utility.Vector3dVector((np.asarray(mesh.vertices)-centroid)* scale_factor)
    mesh.compute_vertex_normals()  # Recompute the normals for the scaled mesh

    # Save the scaled mesh
    o3d.io.write_triangle_mesh(output_file, mesh)

    print(f"The mesh has been scaled and saved as '{output_file}'")
    
def save_frame_as_ply(points, colors, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)
    
# Example usage
all_points = []
N=10000
for i in list(range(25,71,1)):
    number_str = "{:04d}".format(i)
    print(number_str)
    input_xyz = f"./swing_pick_objs_sample1w/{number_str}.xyz"
    all_points.append(np.loadtxt(input_xyz))

T=len(all_points)
all_points = np.vstack(all_points)  # Combine into one TNx3 array
# Centering and scaling
centroid = np.mean(all_points, axis=0)
all_points -= centroid
max_dist = np.max(np.sqrt(np.sum(all_points ** 2, axis=1)))
all_points /= max_dist * 2  # scale to fit into [-0.5, 0.5]

print(all_points.shape)
print(all_points.min())
print(all_points.max())
# Reshape the point cloud back to TxNx3
all_points_reshaped = all_points.reshape(T, N, 3)
print(all_points.shape)

os.makedirs('./swing_pick_objs_sample1w_fpsUnify')
os.makedirs('./swing_pick_objs_scaled')
# Example usage
t=0
for i in list(range(25,71,1)):
    number_str = "{:04d}".format(i)
    output_xyz = f"./swing_pick_objs_sample1w_fpsUnify/{number_str}.xyz"
    np.savetxt(output_xyz, all_points_reshaped[t])
    
    input_file = f'./swing_pick_objs/mesh_{number_str}.obj'
    output_file = f'./swing_pick_objs_scaled/mesh_{number_str}.obj'
    scale_mesh(input_file, output_file, centroid, max_dist)
    
    t+=1
print(t)
print(T)