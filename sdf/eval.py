from sdf.utils import generate_grid , compute_metrics , reconstruct
import numpy as np
import os
import matplotlib.pyplot as plt
import trimesh
from sdf.utils import  get_model_size
from sdf import config



def display_stats( obj_name ,model , chamfer_dist , hausdorff_dist , model_type):
    LODS = []
    if(model_type == "hash"):
        b = np.exp((np.log(config.hash_param.get("max_grid_res")) - np.log(config.hash_param.get("min_grid_res"))) / 
                   (config.hash_param.get("num_LOD") - 1))
        LODS = [int(1 + np.floor(config.hash_param.get("min_grid_res") * (b**l))) for l in range(config.hash_param.get("num_LOD"))]
    elif(model_type == "dense"):
        LODS = [2**L for L in range(config.dense_param.get("base_lod"), 
                                    config.dense_param.get("base_lod") + config.dense_param.get("num_lod"))]


    
    print(obj_name)
    print(model_type)
    print(LODS)
    print(f"OBJ Dataset: {obj_name}")
    print(f"Grid Type: {model_type}")
    print(f"Level of Detal: {LODS}")
    print(f"Chamfer distance: {chamfer_dist:.4f}")
    print(f"Hausdorff distance: {hausdorff_dist:.4f}")
    print(f"Model size in KB : {get_model_size(model):.4f}")
    print("##################")



def eval_model(model , obJ_name , verts , model_type ,  res = 128):
    grid, transform = generate_grid(verts, res=res)
    rec_verts, rec_faces = reconstruct(model, grid, res, transform)

    reconstr_path = f"reconstructions/{obJ_name}"
    os.makedirs(os.path.dirname(reconstr_path), exist_ok=True)
    trimesh.Trimesh(rec_verts, rec_faces).export(reconstr_path)

    gt_path = f"data/{obJ_name}"

    chamfer_dist, hausdorff_dist = compute_metrics(
        reconstr_path, gt_path, num_samples=1000
    )


    display_stats(obJ_name,model , chamfer_dist , hausdorff_dist , model_type)



