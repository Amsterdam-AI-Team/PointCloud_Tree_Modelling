# Tree_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

""" 
Point cloud clipping tools - Module (Python)

The module is adapted from:
https://github.com/Amsterdam-AI-Team/Urban_PointCloud_Processing
"""

import logging

import numba
import numpy as np
from numba import jit

import utils.math_utils as math_utils

logger = logging.getLogger(__name__)


@jit(nopython=True, cache=True)
def axis_clip(points, axis, lower=-np.inf, upper=np.inf):
    """
    Clip all points within bounds of a certain axis.

    Parameters
    ----------
    points : array of shape (n_points, 2)
        The points.
    axis : int
        The axis to clip along.
    lower : float (default: -inf)
        Lower bound of the axis.
    upper : float (default: inf)
        Upperbound of the axis.

    Returns
    -------
    A boolean mask with True entries for all points within the rectangle.
    """
    clip_mask = ((points[:, axis] <= upper) & (points[:, axis] >= lower))
    return clip_mask


@jit(nopython=True, cache=True)
def rectangle_clip(points, rect):
    """
    Clip all points within a rectangle.

    Parameters
    ----------
    points : array of shape (n_points, 2)
        The points.
    rect : tuple of floats
        (x_min, y_min, x_max, y_max)

    Returns
    -------
    A boolean mask with True entries for all points within the rectangle.
    """
    clip_mask = ((points[:, 0] >= rect[0]) & (points[:, 0] <= rect[2])
                 & (points[:, 1] >= rect[1]) & (points[:, 1] <= rect[3]))
    return clip_mask


@jit(nopython=True, cache=True)
def circle_clip(points, center, radius):
    """
    Clip all points within a circle (or unbounded cylinder).

    Parameters
    ----------
    points : array of shape (n_points, 2)
        The points.
    center : tuple of floats (x, y)
        Center point of the circle.
    radius : float
        Radius of the circle.

    Returns
    -------
    A boolean mask with True entries for all points within the circle.
    """
    clip_mask = (np.power((points[:, 0] - center[0]), 2)
                 + np.power((points[:, 1] - center[1]), 2)
                 <= np.power(radius, 2))
    return clip_mask
