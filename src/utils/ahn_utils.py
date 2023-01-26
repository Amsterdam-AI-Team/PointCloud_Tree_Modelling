# Tree_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

""" 
AHN utility methods - Module (Python)

The module is adapted from:
https://github.com/Amsterdam-AI-Team/Urban_PointCloud_Processing
"""

import os, logging

import numpy as np
import open3d as o3d
from pathlib import Path
from abc import ABC, abstractmethod

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

    @abstractmethod
    def get_tree_surface(self, treecode):
        pass

    @abstractmethod
    def get_surface(self, treecode):
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

    def tree_surface_z(self, treecode):
        """Get z level of treecode"""
        treecode_split = treecode.split('_')
        points = np.array([[int(treecode_split[0]), int(treecode_split[1])]])
        return self.interpolate(treecode, points)[0]


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


    def get_tree_surface(self, treecode):
        """Function to get the surface points of a tree."""
        ahn_tile = self.filter_file(treecode)
        X, Y = np.meshgrid(ahn_tile['x'], ahn_tile['y'])
        ahn_points = np.vstack(map(np.ravel, [X,Y,ahn_tile['ground_surface']])).T
        ahn_points = ahn_points[~np.isnan(ahn_points).any(axis=1)]
        return ahn_points

    
    def get_surface(self, treecode):
        """Function to get the surface points of a tree."""
        ahn_tile = self.filter_file(treecode)
        X, Y = np.meshgrid(ahn_tile['x'], ahn_tile['y'])
        ahn_points = np.vstack(map(np.ravel, [X,Y,ahn_tile['ground_surface']])).T
        ahn_points = ahn_points[~np.isnan(ahn_points).any(axis=1)]
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ahn_points))
        return cloud


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


class AHNFileNotFoundError(Exception):
    """Exception raised for missing AHN files."""
