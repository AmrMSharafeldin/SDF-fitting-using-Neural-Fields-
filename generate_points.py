import numpy as np
import igl
import trimesh


def signed_distance(queries, vert, face):  # remove NAN's
    S, I, C = igl.signed_distance(queries, vert, face)
    if len(S.shape) == 0:
        S = S.reshape(1)
    return np.nan_to_num(S), I, C


def apply_transform(Xbd):
    shift = (Xbd.max(axis=0) + Xbd.min(axis=0)) / 2
    Xbd -= shift[None, ...]
    return Xbd


def generate_gt_samples(shape_path, sample_N=64**3, near_std=0.015, far_std=0.2):
    mesh = trimesh.load(shape_path)
    vert, face = np.array(mesh.vertices), np.array(mesh.faces)

    if np.abs(vert).max() > 1.0:
        print("Warning, data exceeds bbox 1.", shape_path, np.abs(vert).max())
    Xbd = trimesh.sample.sample_surface(mesh, sample_N)[0]

    near_num = int(sample_N * 4 / 5)
    far_num = sample_N - near_num

    near_pts = Xbd[:near_num].copy()
    far_pts = Xbd[near_num:].copy()

    near_pts += near_std * np.random.randn(*near_pts.shape)
    far_pts += far_std * np.random.randn(*far_pts.shape)

    Xtg = np.concatenate([near_pts, far_pts], axis=0)
    mask = np.logical_or(Xtg > 0.99, Xtg < -0.99)
    Xtg[mask] = np.random.rand(mask.sum()) * 2 - 1
    Xtg = Xtg.clip(-0.99, 0.99)
    assert Xtg.min() >= -1.00001 and Xtg.max() <= 1.00001
    Ytg, _, _ = signed_distance(Xtg, vert, face)

    Xtg = Xtg.astype(np.float16)
    Ytg = np.sign(Ytg.astype(np.float16))
    Xbd = Xbd.astype(np.float16)
    return Xbd, Xtg, Ytg


def save_samples(shape_path, out_name):
    vert, face = igl.read_triangle_mesh(shape_path)
    for i in range(3):
        unq = np.unique(vert[:, i])
        if ((unq > 1) * (unq < -1)).any():
            assert 0

    _, x, y = generate_gt_samples(shape_path)
    colors = np.zeros((x.shape[0], 3))
    colors[y == 1] = [1, 0, 0]
    colors[y == -1] = [0, 1, 0]
    trimesh.points.PointCloud(x, colors=colors).export(out_name)
    print(x.shape)


if __name__ == "__main__":
    from tqdm import tqdm
    from utils import plot_points, download_data
    download_data()
    plot_points('processed/bunny.obj')
    for name in tqdm(["bunny", "column", "dragon_original", "serapis", "utah_teapot"]):
        save_samples(f"data/{name}.obj", f"processed/{name}.obj")
