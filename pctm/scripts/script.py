#!/usr/bin/python

# PointCloud_Tree_Modelling by Amsterdam Intelligence, GPL-3.0 license

"""
Processing - Scipt (Python)
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Helper script to allow importing from parent folder.
import set_path  # noqa: F401
from utils import (
      ahn_utils,
      las_utils,
      o3d_utils,
      tree_utils
  )


os.environ['KMP_WARNINGS'] = 'off'

# set-up error logging
logging.basicConfig(filename='python.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    encoding='utf-8',
                    level=logging.INFO)

ADTREE_EXE = '../../AdTree/build/bin/AdTree.app/Contents/MacOS/AdTree'
DATA_KEYS = ['source', 'treecode', 'stem_basepoint', 'tree_height', 'stem_height',
             'crown_height', 'crown_baseheight', 'stem_angle', 'DBH', 'CBH',
             'crown_shape', 'crown_diameter', 'crown_volume-alpha', 'crown_volume-convex']

def _export_lods(out_path, tree_cloud, tree_data):
    path_close = tree_data['source']+'_'+tree_data['treecode']+'.obj'
    try: # LOD 2
        lod = tree_utils.generate_LOD_v2(tree_cloud, tree_data['DBH']/2, tree_data['stem_basepoint'],
                                tree_data['crown_basepoint'], tree_data['crown_height'])
        o3d_utils.to_trimesh(lod).export(
            out_path.joinpath('lods/lod_2_' + path_close))
    except Exception:
        pass

    try: # LOD 3
        lod = tree_utils.generate_LOD_v3(tree_data['DBH']/2, tree_data['stem_basepoint'],
                              tree_data['crown_basepoint'], tree_data['crown_mesh-convex'])
        o3d_utils.to_trimesh(lod).export(
            out_path.joinpath('lods/lod_3_' + path_close))
    except Exception:
        pass

    try: # LOD 3.1
        if tree_data['crown_mesh-alpha'].get_volume() > 2:
            lod = tree_utils.generate_LOD_v3_1(tree_data['crown_mesh-alpha'], tree_data['stem_mesh'])
            o3d_utils.to_trimesh(lod).export(
                out_path.joinpath('lods/lod_31_' + path_close))
    except Exception:
        pass


def _process_file(out_path, ahn_reader, file, lod=False):
    '''Processes a file.'''
    treecode = las_utils.get_treecode_from_filename(file.name)

    try:
        logging.info("Processing file "+ str(file))
        tree_cloud = o3d_utils.read_las(file)
        ground_cloud = ahn_reader.get_surface(treecode)

        # process
        tree_data, _ = tree_utils.process_tree(tree_cloud, ground_cloud, ADTREE_EXE)
        tree_data['source'] = file.parent.name
        tree_data['treecode'] = treecode

        # lods
        _export_lods(out_path, tree_cloud, tree_data)

    except Exception as error_msg:
        logging.error("in "+ str(file), exc_info=error_msg)
        tree_data = {
            'source': file.parent.name,
            'treecode': treecode
        }

    tree_data = {key:tree_data[key] for key in DATA_KEYS if key in tree_data.keys()}

    return tree_data


if __name__ == '__main__':

    desc_str = '''This script provides processing of a folder of tree
                  point clouds to extract geometric features and 3D models. The
                  results are saved to .csv.'''
    parser = argparse.ArgumentParser(description=desc_str)
    parser.add_argument('--in_folder', metavar='path', action='store',
                        type=str, required=True)
    parser.add_argument('--lod',  action='store_true')
    args = parser.parse_args()

    in_folder = Path(args.in_folder)

    if not in_folder.is_dir():
        print('The input path does not exist')
        sys.exit()

    ahn_data_folder = in_folder.joinpath('ahn_surf')
    if not ahn_data_folder.is_dir():
        print('The ahn data folder does not exist')
        sys.exit()

    if args.lod:
        in_folder.joinpath('lods').mkdir(exist_ok=True)

    npz_reader = ahn_utils.NPZReader(ahn_data_folder)

    file_types = ('.LAS', '.las', '.LAZ', '.laz')
    files = [f for f in in_folder.glob('*/filtered_tree_*')
             if f.name.endswith(file_types)]

    files_tqdm = tqdm(files, unit="file",
                    disable=False, smoothing=0)

    results = []
    for file in files_tqdm:
        r = _process_file(in_folder, npz_reader, file, args.lod)
        results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(in_folder.joinpath('results.csv'))
