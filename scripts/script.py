#!/usr/bin/python

import argparse
import os
import sys
import glob
import pathlib
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
from pathlib import Path

# Helper script to allow importing from parent folder.
import set_path  # noqa: F401
import tree as tree_utils
import utils.o3d_utils as o3d_utils
import utils.ahn_utils as ahn_utils
import utils.las_utils as las_utils

adTree_exe = '../../AdTree-single/build/bin/AdTree.app/Contents/MacOS/AdTree'


def _process_file(file, npz_reader, adTree_exe):

    tree_cloud = o3d_utils.read_las(file)
    treecode = las_utils.get_treecode_from_filename(file.name)
    ground_points = npz_reader.get_tree_surface(treecode)
    ground_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ground_points))

    processed_tree = tree_utils.process_tree(tree_cloud, ground_cloud, adTree_exe, crown_model_method='Convex_Hull')

    return processed_tree


if __name__ == '__main__':
    global args

    desc_str = '''This script provides processing of a folder of tree
                  point clouds to extract features. The results are saved as csv.'''
    parser = argparse.ArgumentParser(description=desc_str)
    parser.add_argument('--in_folder', metavar='path', action='store',
                        type=str, required=True)
    parser.add_argument('--ahn_folder', metavar='path', action='store',
                        type=str, required=True)
    parser.add_argument('--out_folder', metavar='path', action='store',
                        type=str, required=False)
    parser.add_argument('--resolution', metavar='float', action='store',
                        type=float, required=False, default=0.1)
    args = parser.parse_args()

    if args.out_folder is None:
        args.out_folder = args.in_folder

    if not os.path.isdir(args.in_folder):
        print('The input path does not exist')
        sys.exit()

    if not os.path.isdir(args.ahn_folder):
        print('The ahn path does not exist')
        sys.exit()

    if args.out_folder != args.in_folder:
        Path(args.out_folder).mkdir(parents=True, exist_ok=True)

    in_folder = Path(args.in_folder)

    file_types = ('.LAS', '.las', '.LAZ', '.laz')
    files = [f for f in in_folder.glob('*')
             if f.name.endswith(file_types)]

    files_tqdm = tqdm(files, unit="file",
                    disable=False, smoothing=0)

    npz_reader = ahn_utils.NPZReader(args.ahn_folder)

    results = []
    for file in files:
        print(file.name)
        result = _process_file(file, npz_reader=npz_reader, adTree_exe=adTree_exe)
        result = {key:result[key] for key in ['stem_startpoint', 'stem_height', 'stem_angle', 'DBH', 'crown_baseheight', 'crown_height', 'crown_diameter', 'crown_volume', 'crown_volume']}
        result['Boom'] = file.name
        results.append(result)

    df = pd.DataFrame(results)

    df.to_csv(os.path.join(args.out_folder, 'results.csv'))