# Urban_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

"""This module provides utility methods for AHN data."""

import numpy as np
import pandas as pd
import copy
import warnings
import os
import logging
from abc import ABC, abstractmethod
from tifffile import TiffFile, imread
from pathlib import Path
from scipy import interpolate
from scipy.ndimage import measurements, generic_filter
from scipy.ndimage.morphology import binary_dilation

from .las_utils import get_bbox_from_tile_code
from .interpolation import FastGridInterpolator

logger = logging.getLogger(__name__)


class AHNReader(ABC):
    """Abstract class for reading AHN data."""

    @property
    @classmethod
    @abstractmethod
    def NAME(cls):
        return NotImplementedError

    def __init__(self, data_folder, caching):
        super().__init__()
        self.path = Path(data_folder)
        self.set_caching(caching)
        self._clear_cache()
        if not self.path.exists():
            print('Input folder does not exist.')
            raise ValueError

    @abstractmethod
    def filter_file(self, treecode):
        pass

    def set_caching(self, state):
        if not hasattr(self, 'caching') or (self.caching is not state):
            self.caching = state
            if not self.caching:
                self._clear_cache()
                logger.debug('Caching disabled.')
            else:
                logger.debug('Caching enabled.')

    def _clear_cache(self):
        self.cache = {'treecode': ''}

    def cache_interpolator(self, treecode, points, surface='ground_surface'):
        logger.info(f'Caching {surface} for tile {treecode}.')
        self.set_caching(True)
        if self.cache['treecode'] != treecode:
            # Clear cache.
            self._clear_cache()
        ahn_tile = self.filter_file(treecode)
        if surface not in ahn_tile:
            logger.error(f'Unknown surface: {surface}.')
            raise ValueError
        fast_z = FastGridInterpolator(
            ahn_tile['x'], ahn_tile['y'], ahn_tile[surface])
        self.cache['treecode'] = treecode
        self.cache[surface] = fast_z(points)

    def interpolate(self, treecode, points=None, mask=None,
                    surface='ground_surface'):
        if points is None and mask is None:
            logger.error('Must provide either points or mask.')
            raise ValueError
        if self.caching and mask is not None:
            # Try retrieving cache.
            if self.cache['treecode'] == treecode:
                if surface in self.cache:
                    return self.cache[surface][mask]
                else:
                    logger.debug(
                        f'Surface {surface} not in cache for tile {treecode}.')
        elif self.caching:
            logger.debug('Caching enabled but no mask provided.')

        # No cache, fall back to FastGridInterpolator.
        if self.cache['treecode'] != treecode and points is None:
            logger.error(
                f'Tile {treecode} not cached and no points provided.')
            raise ValueError

        ahn_tile = self.filter_file(treecode)
        if surface not in ahn_tile:
            logger.error(f'Unknown surface: {surface}.')
            raise ValueError
        fast_z = FastGridInterpolator(
            ahn_tile['x'], ahn_tile['y'], ahn_tile[surface])
        return fast_z(points)


class NPZReader(AHNReader):
    """
    NPZReader for AHN3 data. The data folder should contain the pre-processed
    .npz files.

    Parameters
    ----------
    data_folder : str or Path
        Folder containing the .npz files.
    caching : bool (default: True)
        Enable caching of the current ahn tile and interpolation data.
    """

    NAME = 'npz'

    def __init__(self, data_folder, caching=True):
        super().__init__(data_folder, caching)

    def filter_file(self, treecode):
        """
        Returns an AHN tile dict for the area represented by the given
        CycloMedia tile-code. TODO also implement geotiff?
        """
        if self.caching:
            if self.cache['treecode'] != treecode:
                self._clear_cache()
                self.cache['treecode'] = treecode
            if 'ahn_tile' not in self.cache:
                self.cache['ahn_tile'] = load_ahn_file(
                        os.path.join(self.path, 'ahn_surf_' + treecode + '.npz'))
            return self.cache['ahn_tile']
        else:
            return load_ahn_file(
                        os.path.join(self.path, 'ahn_surf_' + treecode + '.npz'))

def load_ahn_file(ahn_file):
    """
    Load the ground and building surface grids in a given AHN .npz file and
    return the results as a dict with keys 'x', 'y', 'ground_surface' and
    'building_surface'.
    """
    if not os.path.isfile(ahn_file):
        msg = f'Tried loading {ahn_file} but file does not exist.'
        raise AHNFileNotFoundError(msg)

    ahn = np.load(ahn_file)
    ahn_tile = {'x': ahn['x'],
                'y': ahn['y'],
                'ground_surface': ahn['ground'].astype(float)}
    return ahn_tile


def _get_gap_coordinates(ahn_tile, max_gap_size=50, gap_flag=np.nan):
    """
    Helper method. Get the coordinates of gap pixels in the AHN data. The
    max_gap_size determines the maximum size of gaps (in AHN pixels) that will
    be considered.

    Parameters
    ----------
    ahn_tile : dict
        E.g., output of GeoTIFFReader.filter_file(.).
    max_gap_size : int (default: 50)
        The maximum size (in grid cells) for gaps to be considered.
    gap_flag : float (default: np.nan)
        Flag used for missing data.

    Returns
    -------
    An array of shape (n_pixes, 2) containing the [x, y] coordinates of the gap
    pixels.
    """
    # Create a boolean mask for gaps.
    if np.isnan(gap_flag):
        gaps = np.isnan(ahn_tile['ground_surface'])
    else:
        gaps = (ahn_tile['ground_surface'] == gap_flag)

    # Find connected components in the gaps mask and compute their sizes.
    gap_ids, num_gaps = measurements.label(gaps)
    ids = np.arange(num_gaps + 1)
    gap_sizes = measurements.sum(gaps, gap_ids, index=ids)

    # Collect all gap coordinates.
    gap_coords = np.empty(shape=(0, 2), dtype=int)
    for i in ids:
        if 0 < gap_sizes[i] <= max_gap_size:
            # The lower bound 0 is used to ignore the 'non-gap' cluster which
            # has size 0.
            gap_coords = np.vstack([gap_coords, np.argwhere(gap_ids == i)])
    return gap_coords


def fill_gaps(ahn_tile, max_gap_size=50, gap_flag=np.nan, inplace=False):
    """
    Fill gaps in the AHN ground surface by interpolation. The max_gap_size
    determines the maximum size of gaps (in AHN pixels) that will be
    considered. A copy of the AHN tile will be returned, unless 'inplace' is
    set to True in which case None will be returned.

    Parameters
    ----------
    ahn_tile : dict
        E.g., output of GeoTIFFReader.filter_file(.).
    max_gap_size : int (default: 50)
        The maximum size for gaps to be considered.
    gap_flag : float (default: np.nan)
        Flag used for missing data.
    inplace: bool (default: False)
        Whether or not to modify the AHN tile in place.

    Returns
    -------
    If inplace=false, a copy of the AHN tile with filled gaps is returned.
    Else, None is returned.
    """
    # Get the coodinates of gap pizels to consider.
    gap_coords = _get_gap_coordinates(ahn_tile, max_gap_size=max_gap_size,
                                      gap_flag=gap_flag)

    # Mask the z-data to exclude gaps.
    if np.isnan(gap_flag):
        mask = ~np.isnan(ahn_tile['ground_surface'])
    else:
        mask = ~(ahn_tile['ground_surface'] == gap_flag)

    # Get the interpolation values for the gaps.
    x = np.arange(0, len(ahn_tile['x']))
    y = np.arange(0, len(ahn_tile['y']))
    xx, yy = np.meshgrid(x, y)
    # TODO: method='cubic' is just a default, we should check what works best
    # for us.
    int_values = interpolate.griddata(
        points=(xx[mask], yy[mask]),
        values=ahn_tile['ground_surface'][mask].ravel(),
        xi=(gap_coords[:, 1], gap_coords[:, 0]),
        method='cubic')

    # Return the filled AHN tile.
    if not inplace:
        filled_ahn = copy.deepcopy(ahn_tile)
        filled_ahn['ground_surface'][gap_coords[:, 0], gap_coords[:, 1]] \
            = int_values
        return filled_ahn
    else:
        ahn_tile['ground_surface'][gap_coords[:, 0], gap_coords[:, 1]] \
            = int_values
        return None

def fill_gaps_intuitive(ahn_tile):
    """
    Fill nans in the AHN ground surface by interpolation. First, linear interpolation is used.
    The remaining gaps are filled using the z-value of nearest point. 
    The AHN tile will be returned

    Parameters
    ----------
    ahn_tile : dict
        E.g., output of GeoTIFFReader.filter_file(.).
    inplace: bool (default: False)
        Whether or not to modify the AHN tile in place.

    Returns
    -------
    If inplace=false, a copy of the AHN tile with filled gaps is returned.
    Else, None is returned.
    """
    # Copy ahn_tile
    filled_ahn = copy.deepcopy(ahn_tile)

    # Get the coodinates of gap pixels to consider.
    gaps = np.isnan(filled_ahn['ground_surface'])
    if 'artifact_surface' in ahn_tile.keys():
        filled_ahn['ground_surface'][gaps] = filled_ahn['artifact_surface'][gaps]

    # Get the coodinates of gap pixels to consider.
    gaps = np.isnan(filled_ahn['ground_surface'])
    gap_coords = np.argwhere(gaps)

    # Get the interpolation values for the gaps.
    x = np.arange(0, len(filled_ahn['x']))
    y = np.arange(0, len(filled_ahn['y']))
    xx, yy = np.meshgrid(x, y)
    int_values = interpolate.griddata(
                        points=(xx[~gaps], yy[~gaps]),
                        values=filled_ahn['ground_surface'][~gaps].ravel(),
                        xi=(gap_coords[:, 1], gap_coords[:, 0]),
                        method='linear')

    # Return the filled AHN tile.
    filled_ahn['ground_surface'][gap_coords[:, 0], gap_coords[:, 1]] \
            = int_values

    # Get the interpolation values for the remaining gaps.
    gaps = np.isnan(filled_ahn['ground_surface'])
    gap_coords = np.argwhere(gaps)
    int_values = interpolate.griddata(
                        points=(xx[~gaps], yy[~gaps]),
                        values=filled_ahn['ground_surface'][~gaps].ravel(),
                        xi=(gap_coords[:, 1], gap_coords[:, 0]),
                        method='nearest')

    filled_ahn['ground_surface'][gap_coords[:, 0], gap_coords[:, 1]] = int_values
    return filled_ahn

def smoothen_edges(ahn_tile, thickness=1, gap_flag=np.nan, inplace=False):
    """
    Smoothen the edges of missing AHN ground surface data in the ahn_tile. In
    effect, this 'pads' the ground surface around gaps by the given 'thickness'
    and prevents small gaps around e.g. buildings when labelling a point cloud.
    A copy of the AHN tile will be returned, unless 'inplace' is set to True in
    which case None will be returned.

    Parameters
    ----------
    ahn_tile : dict
        E.g., output of GeoTIFFReader.filter_file(.).
    thickness : int (default: 1)
        Thickness of the edge, for now only a value of 1 or 2 makes sense.
    gap_flag : float (default: np.nan)
        Flag used for missing data.
    inplace: bool (default: False)
        Whether or not to modify the AHN tile in place.

    Returns
    -------
    If inplace=false, a copy of the AHN tile with smoothened edges is returned.
    Else, None is returned.
    """
    if np.isnan(gap_flag):
        mask = ~np.isnan(ahn_tile['ground_surface'])
        z_data = ahn_tile['ground_surface']
    else:
        mask = ~(ahn_tile['ground_surface'] == gap_flag)
        z_data = copy.deepcopy(ahn_tile['ground_surface'])
        z_data[~mask] = np.nan

    # Find the edges of data gaps.
    edges = mask ^ binary_dilation(mask, iterations=thickness)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Compute smoothened AHN data by taking the mean of surrounding pixel
        # values (ignoring NaNs).
        # TODO: a 'thickness' of more than 2 would require a bigger footprint.
        smoother = generic_filter(z_data, np.nanmean,
                                  footprint=np.ones((3, 3), dtype=int),
                                  mode='constant', cval=np.nan)

    if inplace:
        ahn_tile['ground_surface'][edges] = smoother[edges]
        return None
    else:
        smoothened_ahn = copy.deepcopy(ahn_tile)
        smoothened_ahn['ground_surface'][edges] = smoother[edges]
        return smoothened_ahn


class AHNFileNotFoundError(Exception):
    """Exception raised for missing AHN files."""
