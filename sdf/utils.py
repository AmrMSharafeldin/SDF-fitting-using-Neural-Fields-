import numpy as np
import os
import matplotlib.pyplot as plt
import trimesh
import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from skimage import measure


#Instructor Helper functions 



def generate_grid(point_cloud, res):
    """Generate grid over the point cloud with given resolution
    Args:
        point_cloud (np.array, [N, 3]): 3D coordinates of N points in space
        res (int): grid resolution
    Returns:
        coords (np.array, [res*res*res, 3]): grid vertices
        coords_matrix (np.array, [4, 4]): transform matrix: [0,res]x[0,res]x[0,res] -> [x_min, x_max]x[y_min, y_max]x[z_min, z_max]
    """
    b_min = np.min(point_cloud, axis=0)
    b_max = np.max(point_cloud, axis=0)

    coords = np.mgrid[:res, :res, :res]
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4)
    length = b_max - b_min
    length += length / res
    coords_matrix[0, 0] = length[0] / res
    coords_matrix[1, 1] = length[1] / res
    coords_matrix[2, 2] = length[2] / res
    coords_matrix[0:3, 3] = b_min
    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    coords = coords.T

    return coords, coords_matrix


def batch_eval(points, eval_func, num_samples):
    """Predict occupancy of values batch-wise
    Args:
        points (np.array, [N, 3]): 3D coordinates of N points in space
        eval_func (function): function that takes a batch of points and returns occupancy values
        num_samples (int): number of points to evaluate at once
    Returns:
        occ (np.array, [N,]): occupancy values for each point
    """
    points = torch.tensor(points).to(torch.float32).to(device='cuda')
    num_pts = points.shape[0]
    occ = np.zeros(num_pts)

    num_batches = num_pts // num_samples
    for i in range(num_batches):
        occ[i * num_samples : i * num_samples + num_samples] = torch.squeeze(eval_func(
            points[i * num_samples : i * num_samples + num_samples]
        )).cpu().detach().numpy()
    if num_pts % num_samples:
        
        occ[num_batches * num_samples :] = torch.squeeze(eval_func(
            points[num_batches * num_samples :]
        )).cpu().detach().numpy()

    return occ


def eval_grid(coords, eval_func, num_per_sample=1024):
    """Predict occupancy of values on a grid
    Args:
        coords (np.array, [N, 3]): 3D coordinates of N points in space
        eval_func (function): function that takes a batch of points and returns occupancy values
        num_per_sample (int): number of points to evaluate at once

    Returns:
        occ (np.array, [N,]): occupancy values for each point
    """
    coords = coords.reshape([-1, 3])
    occ = batch_eval(coords, eval_func, num_samples=num_per_sample)
    return occ


def reconstruct(model, grid, res, transform):
    """Reconstruct mesh by predicting occupancy values on a grid
    Args:
        model (function): function that takes a batch of points and returns occupancy values
        grid (np.array, [N, 3]): 3D coordinates of N points in space
        res (int): grid resolution
        transform (np.array, [4, 4]): transform matrix: [0,res]x[0,res]x[0,res] -> [x_min, x_max]x[y_min, y_max]x[z_min, z_max]

    Returns:
        verts (np.array, [M, 3]): 3D coordinates of M vertices
        faces (np.array, [K, 3]): indices of K faces
    """

    occ = eval_grid(grid, model)
    occ = occ.reshape([res, res, res])

    verts, faces, normals, values = measure.marching_cubes(occ, -0.5)
    verts = np.matmul(transform[:3, :3], verts.T) + transform[:3, 3:4]
    verts = verts.T

    return verts, faces


def compute_metrics(reconstr_path, gt_path, num_samples=1000000):
    """Compute chamfer and hausdorff distances between the reconstructed mesh and the ground truth mesh
    Args:
        reconstr_path (str): path to the reconstructed mesh
        gt_path (str): path to the ground truth mesh
        num_samples (int): number of points to sample from each mesh

    Returns:
        chamfer_dist (float): chamfer distance between the two meshes
        hausdorff_dist (float): hausdorff distance between the two meshes
    """
    reconstr = trimesh.load(reconstr_path)
    gt = trimesh.load(gt_path)

    # sample points on the mesh surfaces using trimesh
    reconstr_pts = reconstr.sample(num_samples)
    gt_pts = gt.sample(num_samples)

    # compute chamfer distance between the two point clouds
    reconstr_tree = KDTree(reconstr_pts)
    gt_tree = KDTree(gt_pts)
    dist1, _ = reconstr_tree.query(gt_pts)
    dist2, _ = gt_tree.query(reconstr_pts)
    chamfer_dist = (dist1.mean() + dist2.mean()) / 2
    hausdorff_dist = max(dist1.max(), dist2.max())

    return chamfer_dist, hausdorff_dist


# My helper functions:


def load_data(path):
    obj = trimesh.load(path)
    verts = np.array(obj.vertices)
    gt_occ = np.array(obj.visual.vertex_colors)[:, 0]
    gt_occ = (gt_occ == 0).astype("float32") * -2 + 1
    return verts , gt_occ


save_dir = 'plot_images'
os.makedirs(save_dir, exist_ok=True)

def viz(coords, values, epoch):
    coords = coords.cpu().detach().numpy()[:5000]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    mask = (values[:].cpu().detach().numpy()) >= 0.9
    mask = mask[:5000]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(
        x[mask[:,0]], y[mask[:,0]], zs=z[mask[:,0]], zdir="y", alpha=1, c='r', label='Predicted'
    )
    ax.scatter(
        x[~mask[:,0]], y[~mask[:,0]], zs=z[~mask[:,0]], zdir="y", alpha=0.1, c='b', label='Ground Truth'
    )
    
    ax.set_title(f'Epoch {epoch}')
    ax.legend()
    
    # Save the plot as a JPEG file
    plt.savefig(os.path.join(save_dir, f'plot_epoch_{epoch}.jpeg'))
    plt.close()


def trilinear_interpolation(res, grid, points, grid_type):
    """
    Performs trilinear interpolation of points with respect to a grid.

    Parameters:
        res (int): Resolution of the grid in each dimension.
        grid (torch.Tensor): A 3D torch tensor representing the grid.
        points (torch.Tensor): A 2D torch tensor of shape (n, 3) representing
            the points to interpolate.
        grid_type (str): Type of grid.

    Returns:
        torch.Tensor: A 1D torch tensor of shape (n,) representing the interpolated
            values at the given points.
    """
    PRIMES = [1, 265443567, 805459861]

    # Get the dimensions of the grid
    grid_size, feat_size = grid.shape
    points = points.unsqueeze(0)
    _, N, _ = points.shape

    # Get the x, y, and z coordinates of the eight nearest points for each input point
    x = points[:, :, 0] * (res - 1)
    y = points[:, :, 1] * (res - 1)
    z = points[:, :, 2] * (res - 1)

    x1 = torch.floor(torch.clip(x, 0, res - 1 - 1e-5)).int()
    y1 = torch.floor(torch.clip(y, 0, res - 1 - 1e-5)).int()
    z1 = torch.floor(torch.clip(z, 0, res - 1 - 1e-5)).int()

    x2 = torch.clip(x1 + 1, 0, res - 1).int()
    y2 = torch.clip(y1 + 1, 0, res - 1).int()
    z2 = torch.clip(z1 + 1, 0, res - 1).int()

    # Compute the weights for each of the eight points
    w1 = (x2 - x) * (y2 - y) * (z2 - z)
    w2 = (x - x1) * (y2 - y) * (z2 - z)
    w3 = (x2 - x) * (y - y1) * (z2 - z)
    w4 = (x - x1) * (y - y1) * (z2 - z)
    w5 = (x2 - x) * (y2 - y) * (z - z1)
    w6 = (x - x1) * (y2 - y) * (z - z1)
    w7 = (x2 - x) * (y - y1) * (z - z1)
    w8 = (x - x1) * (y - y1) * (z - z1)

    if grid_type == "NGLOD":
        # Interpolate the values for each point
        id1 = (x1 + y1 * res + z1 * res * res).long()
        id2 = (x2 + y1 * res + z1 * res * res).long()
        id3 = (x1 + y2 * res + z1 * res * res).long()
        id4 = (x2 + y2 * res + z1 * res * res).long()
        id5 = (x1 + y1 * res + z2 * res * res).long()
        id6 = (x2 + y1 * res + z2 * res * res).long()
        id7 = (x1 + y2 * res + z2 * res * res).long()
        id8 = (x2 + y2 * res + z2 * res * res).long()

    elif grid_type == "HASH":
        npts = res**3
        if npts > grid_size:
            id1 = ((x1 * PRIMES[0]) ^ (y1 * PRIMES[1]) ^ (z1 * PRIMES[2])) % grid_size
            id2 = ((x2 * PRIMES[0]) ^ (y1 * PRIMES[1]) ^ (z1 * PRIMES[2])) % grid_size
            id3 = ((x1 * PRIMES[0]) ^ (y2 * PRIMES[1]) ^ (z1 * PRIMES[2])) % grid_size
            id4 = ((x2 * PRIMES[0]) ^ (y2 * PRIMES[1]) ^ (z1 * PRIMES[2])) % grid_size
            id5 = ((x1 * PRIMES[0]) ^ (y1 * PRIMES[1]) ^ (z2 * PRIMES[2])) % grid_size
            id6 = ((x2 * PRIMES[0]) ^ (y1 * PRIMES[1]) ^ (z2 * PRIMES[2])) % grid_size
            id7 = ((x1 * PRIMES[0]) ^ (y2 * PRIMES[1]) ^ (z2 * PRIMES[2])) % grid_size
            id8 = ((x2 * PRIMES[0]) ^ (y2 * PRIMES[1]) ^ (z2 * PRIMES[2])) % grid_size
        else:
            id1 = (x1 + y1 * res + z1 * res * res).long()
            id2 = (x2 + y1 * res + z1 * res * res).long()
            id3 = (x1 + y2 * res + z1 * res * res).long()
            id4 = (x2 + y2 * res + z1 * res * res).long()
            id5 = (x1 + y1 * res + z2 * res * res).long()
            id6 = (x2 + y1 * res + z2 * res * res).long()
            id7 = (x1 + y2 * res + z2 * res * res).long()
            id8 = (x2 + y2 * res + z2 * res * res).long()
    else:
        print("NOT IMPLEMENTED")

    values = (
        torch.einsum("ab,abc->abc", w1, grid[(id1).long()])
        + torch.einsum("ab,abc->abc", w2, grid[(id2).long()])
        + torch.einsum("ab,abc->abc", w3, grid[(id3).long()])
        + torch.einsum("ab,abc->abc", w4, grid[(id4).long()])
        + torch.einsum("ab,abc->abc", w5, grid[(id5).long()])
        + torch.einsum("ab,abc->abc", w6, grid[(id6).long()])
        + torch.einsum("ab,abc->abc", w7, grid[(id7).long()])
        + torch.einsum("ab,abc->abc", w8, grid[(id8).long()])
    )
    return values[0]


def MAPE(predictions, actuals):
    absolute_error = torch.abs(predictions - actuals)
    percentage_error = absolute_error / torch.abs(actuals)
    mape = torch.mean(percentage_error)
    return mape  



def get_model_size(model):
    torch.save(model.state_dict(), 'temp.pth')
    file_size_bytes = os.path.getsize('temp.pth')
    file_size_kb = file_size_bytes / 1024
    os.remove('temp.pth')
    return file_size_kb





