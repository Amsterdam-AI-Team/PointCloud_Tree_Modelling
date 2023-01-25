#!/usr/bin/python

import os
import sys
import logging
import argparse
from pathlib import Path
from tqdm.contrib.concurrent import process_map  # or thread_map
from functools import partial
import pandas as pd
import open3d as o3d

# Helper script to allow importing from parent folder.
import set_path  # noqa: F401
import utils.tree_utils as tree_utils
import utils.o3d_utils as o3d_utils
import utils.ahn_utils as ahn_utils
import utils.las_utils as las_utils
import utils.lod_utils as lod_utils

os.environ['KMP_WARNINGS'] = 'off'

# set-up error logging
logging.basicConfig(filename='python.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S', 
                    encoding='utf-8',
                    level=logging.INFO)

ADTREE_EXE = '../../AdTree-single/build/bin/AdTree.app/Contents/MacOS/AdTree'
DATA_KEYS = ['source', 'treecode', 'stem_basepoint', 'tree_height', 'stem_height', 
             'crown_height', 'crown_baseheight', 'stem_angle', 'DBH', 'CBH', 
             'crown_shape', 'crown_diameter', 'crown_volume-alpha', 'crown_volume-convex']

def _export_lods(out_path, tree_cloud, tree_data):

    try: # LOD 2
        lod = lod_utils.lod_2(tree_cloud, tree_data['DBH']/2, tree_data['stem_basepoint'],
                                tree_data['crown_basepoint'], tree_data['crown_height'])
        o3d_utils.to_trimesh(lod).export(
            out_path.joinpath('lods/lod_2_'+tree_data['source']+'_'+tree_data['treecode']+'.obj'))
    except:
        pass

    try: # LOD 3
        lod = lod_utils.lod_3(tree_data['DBH']/2, tree_data['stem_basepoint'],
                              tree_data['crown_basepoint'], tree_data['crown_mesh-convex'])
        o3d_utils.to_trimesh(lod).export(
            out_path.joinpath('lods/lod_3_'+tree_data['source']+'_'+tree_data['treecode']+'.obj'))
    except:
        pass
    
    try: # LOD 3.1
        if tree_data['crown_mesh-alpha'].get_volume() > 2:
            lod = lod_utils.lod_31(tree_data['crown_mesh-alpha'], tree_data['stem_mesh'])
            o3d_utils.to_trimesh(lod).export(
                out_path.joinpath('lods/lod_31_'+tree_data['source']+'_'+tree_data['treecode']+'.obj'))
    except:
        pass


def _process_file(out_path, ahn_reader, file):
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
        
    except Exception as e:
        logging.error("in "+ str(file), exc_info=e)
        tree_data = {
            'source': file.parent.name,
            'treecode': treecode
        }

    tree_data = {key:tree_data[key] for key in DATA_KEYS if key in tree_data.keys()}

    return tree_data


if __name__ == '__main__':
    desc_str = '''This script provides batch processing of a folder of AHN LAS
                  point clouds to extract ground and building surfaces. The
                  results are saved to .npz.'''
    parser = argparse.ArgumentParser(description=desc_str)
    parser.add_argument('--in_folder', metavar='path', action='store',
                        type=str, required=True)
    parser.add_argument('--workers', metavar='int', action='store',
                        type=int, required=False, default=1)
    args = parser.parse_args()

    in_folder = Path(args.in_folder)

    if not in_folder.is_dir():
        print('The input path does not exist')
        sys.exit()

    ahn_data_folder = in_folder.joinpath('ahn_surf')
    if not ahn_data_folder.is_dir():
        print('The ahn data folder does not exist')
        sys.exit()

    in_folder.joinpath('lods').mkdir(exist_ok=True)

    npz_reader = ahn_utils.NPZReader(ahn_data_folder)

    file_types = ('.LAS', '.las', '.LAZ', '.laz')
    files = [f for f in in_folder.glob('*/filtered_tree_*')
             if f.name.endswith(file_types)]

    # Chunk size can be used to reduce overhead for a large number of files.
    CHUNK = 1
    if len(files) > 100:
        CHUNK = 5
    if len(files) > 1000:
        CHUNK = 10

    # Distribute the batch over _max_workers_ cores.
    r = process_map(partial(_process_file, in_folder, npz_reader), files,
                    max_workers=args.workers, chunksize=CHUNK)

    df = pd.DataFrame(r)
    df.to_csv(in_folder.joinpath('batch_porcess.csv'))
