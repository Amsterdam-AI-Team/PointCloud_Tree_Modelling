# Urban_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

"""
This module provides methods to pre-process AHN data. In particular, starting
from an AHN point cloud, there are methods to clip specific tiles from this
point cloud, and further methods to extract ground and building surfaces from
each tile that can be used for automatic labelling of street-level pointclouds.

For an example, see notebooks/1.AHN_preprocessing.ipynb
"""

import numpy as np
import os
import pathlib
import laspy
import re
from tqdm import tqdm

import utils.las_utils as las_utils
from utils.interpolation import SpatialInterpolator

# AHN classification codes (see https://www.ahn.nl/4-classificatie)
# AHN_ARTIFACT is 'Kunstwerk' which includes 'vlonders/steigers, bruggen en ook
# portaalmasten van autosnelwegen'
AHN_OTHER = 1
AHN_GROUND = 2


def _get_ahn_surface(ahn_las, grid_x, grid_y, si_method, n_neighbors=8,
                     max_dist=1.0, power=2., fill_value=np.nan):
    """
    Use maximum-based interpolation or inverse distance weighted interpolation
    (IDW) to generate a surface (grid) from a given AHN cloud.

    For more information on IDW see:
    utils.interpolation.SpatialInterpolator

    Parameters
    ----------
    ahn_las : laspy point cloud
        The AHN point cloud.

    grid_x : list of floats
        X-values for the interpolation grid.

    grid_y : list of floats
        Y-values for the interpolation grid.

    si_method : string
        Spatial interpolator method

    n_neighbours : int (default: 8)
        Maximum number of neighbours to use for IDW.

    max_dist : float (default: 1.0)
        Maximum distance of neighbours to consider for IDW.

    power : float (default: 2.0)
        Power to use for IDW.

    fill_value : float (default: np.nan)
        Fill value to use for 'empty' grid cells for which no interpolation
        could be computed.

    Returns
    -------
    2d array of interpolation values for each <y,x> grid cell.
    """

    if len(ahn_las.points) <= 1:
        return np.full(grid_x.shape, np.nan, dtype='float16')

    points = np.vstack((ahn_las.x, ahn_las.y, ahn_las.z)).T
    positions = np.vstack((grid_x.reshape(-1), grid_y.reshape(-1))).T

    idw = SpatialInterpolator(points[:, 0:2], points[:, 2], method=si_method)
    ahn_gnd_grid = idw(positions, n_neighbors=n_neighbors, max_dist=max_dist,
                       power=power, fill_value=fill_value)

    return (np.around(ahn_gnd_grid.reshape(grid_x.shape), decimals=2)
            .astype('float16'))


def clip_ahn_las_tree(ahn_cloud, las_file, out_folder='', buffer=1, resolution=0.1):
    """
    Clip a tile from the AHN cloud to match the dimensions of a given
    CycloMedia LAS tile, and save the result using the same naming convention.

    Parameters
    ----------
    ahn_cloud : laspy point cloud
        The full AHN point cloud. This is assumed to include the full area of
        the given CycloMedia tile.

    las_file : Path or str
        The CycloMedia tile on which the clip should be based.

    out_folder : str, optional
        Output folder to which the clipped file should be saved. Defaults to
        the current folder.

    buffer : int, optional (default: 1)
        Buffer around the CycloMedia tile (in m) to include, used for further
        processing (e.g. interpolation).
    """
    if type(las_file) == str:
        las_file = pathlib.Path(las_file)
    tree_code = las_utils.get_treecode_from_filename(las_file.name)

    # based on input las
    with laspy.lib.open_las(las_file, mode='r') as flas:
        x_min, x_max = flas.header.x_min - buffer, flas.header.x_max + buffer
        y_min, y_max = flas.header.y_min - buffer, flas.header.y_max + buffer

    mask = np.where(ahn_cloud.classification == AHN_GROUND)[0]
    ahn_x = ahn_cloud.x[mask]
    ahn_y = ahn_cloud.y[mask]

    clip_idx = np.where((x_min <= ahn_x) & (ahn_x <= x_max)
                        & (y_min <= ahn_y) & (ahn_y <= y_max))[0]

    ahn_file = laspy.LasData(ahn_cloud.header)
    ahn_file.points = ahn_cloud.points[mask[clip_idx]]

    if out_folder != '':
        pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)
    ahn_file.write(os.path.join(out_folder, 'ahn_surf_' + tree_code + '.laz'))

    # Create a grid with 0.1m resolution
    grid_y, grid_x = np.mgrid[y_max-resolution/2:y_min:-resolution,
                              x_min+resolution/2:x_max:resolution]

    # Methods for generating surfaces (grids)
    ground_surface = _get_ahn_surface(ahn_file, grid_x, grid_y, 'idw')

    
    filename = os.path.join(out_folder, 'ahn_surf_' + tree_code + '.npz')
    np.savez_compressed(filename,
                        x=grid_x[0, :],
                        y=grid_y[:, 0],
                        ground=ground_surface)
    return filename


def clip_ahn_las_folder(ahn_cloud, in_folder, out_folder=None, buffer=1,
                        resume=False, hide_progress=False):
    """
    Clip a tiles from the AHN cloud to match all CycloMedia LAS tiles in a
    given folder, and save the result using the same naming convention.

    Parameters
    ----------
    ahn_cloud : laspy point cloud
        The full AHN point cloud. This is assumed to include the full area of
        the given CycloMedia tiles.

    in_folder : Path or str
        The input folder (containing the point cloud trees.)

    out_folder : Path or str, optional
        The output folder. Defaults to the input folder.

    buffer : int, optional (default: 1)
        Buffer around the input point cloud (in m) to include, used for further
        processing (e.g. interpolation).

    resume : bool (default: False)
        Whether to resume, i.e. skip existing files in the output folder. If
        set to False, existing files will be overwritten.

    hide_progress : bool (default: False)
        Hide the progress bar.
    """
    if not os.path.isdir(in_folder):
        print('The input path specified does not exist')
        return None

    if type(in_folder) == str:
        in_folder = pathlib.Path(in_folder)

    if out_folder is None:
        out_folder = in_folder

    if out_folder != in_folder:
        pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)

    file_types = ('.LAS', '.las', '.LAZ', '.laz')
    files = [f for f in in_folder.glob('*')
             if f.name.endswith(file_types)]

    if resume:
        # Find which files have already been processed.
        done = set(las_utils.get_treecode_from_filename(file.name)
                   for file in pathlib.Path(out_folder).glob('ahn_surf_*.laz'))
        files = [f for f in files if las_utils.get_treecode_from_filename(f.name) not in done]

    for file in tqdm(files, unit="file", disable=hide_progress, smoothing=0):
        clip_ahn_las_tree(ahn_cloud, file, out_folder=out_folder,
                          buffer=buffer)

