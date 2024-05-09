import os
import argparse
import trimesh
import numpy as np
from sdf.model import SDF_NN
from sdf.eval import eval_model

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SDF model")
    parser.add_argument("--model_type", type=str, default="dense", choices=["dense", "hash"],
                        help="Type of Hyprid Representation model to use: 'dense' or 'hash'")
    parser.add_argument("--res", type=int, default=128,
                        help="Resolution for evaluation")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    for cur_obj in os.listdir("processed"):
        pc = trimesh.load(f"processed/{cur_obj}")
        verts = np.array(pc.vertices)
        gt_occ = np.array(pc.visual.vertex_colors)[:, 0]
        gt_occ = (gt_occ == 0).astype("float32") * -2 + 1
        sdf = SDF_NN(verts, gt_occ, args.model_type)
        eval_model(sdf.model, cur_obj, verts, args.model_type, res=args.res)
