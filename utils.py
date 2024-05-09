import numpy as np
import os
import matplotlib.pyplot as plt
import trimesh
from sklearn.preprocessing import minmax_scale
from scipy.interpolate import RegularGridInterpolator


import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def plot_points(path):
    ax = plt.figure().add_subplot(projection="3d")
    obj = trimesh.load(path)
    x, y, z = obj.vertices[:, 0], obj.vertices[:, 1], obj.vertices[:, 2]
    mask = obj.colors[:, 1] == 255
    ax.scatter(
        x[mask], y[mask], zs=z[mask], zdir="y", alpha=1, c=obj.colors[mask] / 255
    )
    ax.scatter(
        x[~mask], y[~mask], zs=z[~mask], zdir="y", alpha=0.01, c=obj.colors[~mask] / 255
    )
    plt.show()


def download_data():
    import gdown

    if not os.path.exists("./data"):
        gdown.download_folder(
            "https://drive.google.com/drive/folders/1EKWU_daQL3pxFkjFUomGs25_qekyfeAd",
            quiet=False,
        )

    if not os.path.exists("./processed"):
        gdown.download_folder(
            "https://drive.google.com/drive/folders/175_LtuWh1LknbbMjUumPjGzeSzgQ4ett",
            quiet=False,
        )



def load_data(path):
    obj = trimesh.load(path)
    verts = np.array(obj.vertices)
    gt_occ = np.array(obj.visual.vertex_colors)[:, 0]
    gt_occ = (gt_occ == 0).astype("float32") * -2 + 1
    return verts , gt_occ

def viz(coords ,values):
    ax = plt.figure().add_subplot(projection="3d")
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    mask = values[:] > 0
    
    ax.scatter(
        x[mask], y[mask], zs=z[mask], zdir="y", alpha=1, c='r'
    )
    ax.scatter(
        x[~mask], y[~mask], zs=z[~mask], zdir="y", alpha=0.1, c='b'
    )
    plt.show()



