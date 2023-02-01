# PointCloud_Tree_Modelling by Amsterdam Intelligence, GPL-3.0 license

""" 
Laspy utility methods - Module (Python)

The module is adapted from:
https://github.com/Amsterdam-AI-Team/Urban_PointCloud_Processing
"""

import re
import laspy
import pathlib
import numpy as np
import open3d as o3d
from tqdm import tqdm


def to_o3d(las):
    """Convert laspy to open3d point cloud."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(las.xyz))
    colors = np.vstack([las.red,las.green,las.blue]).T
    colors -= colors.min()
    colors = colors / colors.max()
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def print_statistics(las):
    """Provides statistics on las file."""
    n = las.header.point_count
    x_width = np.round(las.header.x_max-las.header.x_min,2)
    x_range = np.round((las.header.x_min, las.header.x_max),2)
    y_width = np.round(las.header.y_max-las.header.y_min,2)
    y_range = np.round((las.header.y_min, las.header.y_max),2)
    z_width = np.round(las.header.z_max-las.header.z_min,2)
    z_range = np.round((las.header.z_min, las.header.z_max),2)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(las.xyz))
    mean_point_distance = np.mean(pcd.compute_nearest_neighbor_distance())

    print('Pointcloud Statistics:')
    print('------------------------')
    print('No. points:',n)
    print('Mean distance:', np.round(mean_point_distance,2),'m')
    print('x:', x_width,'m',x_range)
    print('y:', y_width,'m',y_range)
    print('z:', z_width,'m',z_range)
    print('')


def get_treecode_from_filename(filename):
    """Extract the tile code from a file name."""
    return re.match(r'.*(\d{6}_\d{6}).*', filename)[1]


def get_tilecode_from_filename(filename):
    """Extract the tile code from a file name."""
    return re.match(r'.*(\d{4}_\d{4}).*', filename)[1]


def get_tilecodes_from_folder(las_folder, las_prefix=''):
    """Get a set of unique tilecodes for the LAS files in a given folder."""
    files = pathlib.Path(las_folder).glob(f'{las_prefix}*.laz')
    tilecodes = set([get_tilecode_from_filename(file.name) for file in files])
    return tilecodes


def get_bbox_from_tree_code(treecode, padding=20):
    """Get bbox around treecode: (x_min, x_max, y_min, y_max)"""
    x, y = np.asarray(treecode.split('_'), dtype=int)
    return (x - padding, x + padding, y - padding, y + padding)


def get_bbox_from_tile_code(tile_code, padding=0, width=50, height=50):
    """Get bbox for a given tilecode: ((x_min, y_max), (x_max, y_min))"""
    tile_split = tile_code.split('_')

    # The tile code of each tile is defined as
    # 'X-coordinaat/50'_'Y-coordinaat/50'
    x_min = int(tile_split[0]) * 50
    y_min = int(tile_split[1]) * 50

    return ((x_min - padding, y_min + height + padding),
            (x_min + height + padding, y_min - padding))


def get_bbox_from_las_tile(laz_file, padding=0):
    """Get bbox for a given las tile file: ((x_min, y_max), (x_max, y_min))"""
    if type(laz_file) == str:
        laz_file = pathlib.Path(laz_file)
    tile_code = get_tilecode_from_filename(laz_file.name)

    return get_bbox_from_tile_code(tile_code, padding=padding)


def get_bbox_from_las_folder(in_folder):
    """Get bboxes for the las file in a given folder: (x_min, x_max, y_min, y_max)"""
    if type(in_folder) == str:
        in_folder = pathlib.Path(in_folder)

    bboxes = {}
    file_types = ('.LAS', '.las', '.LAZ', '.laz')
    for las_file in in_folder.glob('*'):
        if las_file.name.endswith(file_types):
            with laspy.open(las_file, mode='r') as f:
                header = f.header
                bbox = (header.x_min,header.x_max,header.y_min,header.y_max)
                bboxes[las_file.name] = bbox

    return bboxes


def read_las(las_file):
    """Read a las file and return the las object."""
    return laspy.read(las_file)


def merge_las_folder(in_folder):
    """Function to merge lasfiles"""

    if type(in_folder) == str:
        in_folder = pathlib.Path(in_folder)
    outfile = in_folder.joinpath('merged.las')

    file_types = ('.LAS', '.las', '.LAZ', '.laz')
    files = [f for f in in_folder.glob('*')
                if f.name.endswith(file_types)]

    files = [f for f in files if f != outfile]

    files_tqdm = tqdm(files, unit="file",
                        disable=False, smoothing=0)

    points = np.zeros((0,3))
    classification = np.array([], dtype="uint8")
    for in_file in files_tqdm:
        in_las = read_las(in_file)
        points = np.vstack([points, in_las.xyz])
        classification = np.hstack([classification, in_las.classification])

    # 1. Create a new header
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="classification", type="uint8"))
    header.offsets = np.min(points, axis=0)
    header.scales = np.array([0.1, 0.1, 0.1])

    # 2. Create a Las
    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.classification = classification

    las.write(outfile)

    return las
