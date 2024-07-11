import open3d as o3d
import trimesh
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


def square_distance(src, dst):
    """
    Calculate Euclid distance between each pair of the two collections of points.
    src: [B, N, C]
    dst: [B, M, C]
    Output dist: [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def furthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    return centroids

def furthest_point_sampling(point_cloud, num_samples):
    """
    Perform furthest point sampling on a point cloud.

    :param point_cloud: Numpy array of shape (N, 3) representing the point cloud.
    :param num_samples: Number of points to sample.
    :return: Indices of the sampled points.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pcd = torch.tensor(point_cloud, dtype=torch.float).to(device)
    pcd = pcd.unsqueeze(0)  # Add batch dimension

    indices = furthest_point_sample(pcd, num_samples).squeeze().long()

    return indices.cpu().numpy()

def fps(input_file, output_file, num_samples):
    """
    Apply furthest point sampling on a colored .ply file and save the result.

    :param input_file: Path to the input .ply file.
    :param output_file: Path to the output .ply file.
    :param num_samples: Number of points to sample.
    """
 
    mesh = trimesh.load(input_file)
    points = mesh.sample(num_samples*4)

    sampled_indices = furthest_point_sampling(points, num_samples)
    sampled_points = points[sampled_indices]

    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
 
    np.savetxt(output_file, np.asarray(sampled_pcd.points))
    




os.makedirs('./swing_pick_objs_sample1w')
# Example usage
for i in list(range(25,71,1)):
    number_str = "{:04d}".format(i)
    print(number_str)
    
    input_ = "./swing_pick_objs/mesh_{}.obj".format(number_str)
    output_xyz = "./swing_pick_objs_sample1w/{}.xyz".format(number_str)
    num_samples = 10000  
    fps(input_, output_xyz, num_samples)
