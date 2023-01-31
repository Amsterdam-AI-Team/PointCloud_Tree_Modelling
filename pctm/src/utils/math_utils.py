# Tree_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

"""
Math utility methods - Module (Python)

The module is adapted from:
https://github.com/Amsterdam-AI-Team/Urban_PointCloud_Processing

The method rodrigues_rot() is adapted from:
https://github.com/SKrisanski/FSCT/blob/main/scripts/measure.py
"""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True, parallel=True)
def vector_angle(u, v=np.array([0., 0., 1.])):
    """
    Returns the angle in degrees between vectors 'u' and 'v'. If only 'u' is
    provided, the angle between 'u' and the vertical axis is returned.
    """
    # see https://stackoverflow.com/a/2827466/425458
    c = np.dot(u/np.linalg.norm(u), v/np.linalg.norm(v))
    clip = np.minimum(1, np.maximum(c, -1))
    return np.rad2deg(np.arccos(clip))


def vector_bearing(vector):
    """Function that converts vector (x,y) to [0,360] bearing."""
    v_u = np.array(vector) / np.linalg.norm(vector)
    initial_bearing = np.rad2deg(np.arctan2(v_u[1], -v_u[0]) - np.arctan2(1, 0))
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing


def bbox_in_bbox(bbox_a, bbox_b):
    """Function to test if bbox_a in bbox_b"""
    return bbox_b[0] <= bbox_a[0] and bbox_b[2] <= bbox_a[2] and \
        bbox_b[1] >= bbox_a[1] and bbox_b[3] >= bbox_a[3]


@jit(nopython=True, cache=True, parallel=True)
def compute_bounding_box(points):
    """
    Get the min/max values of a point list.

    Parameters
    ----------
    points : array of shape (n_points, 2)
        The (x, y) coordinates of the points. Any further dimensions will be
        ignored.

    Returns
    -------
    tuple
        (x_min, y_min, x_max, y_max)
    """
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])

    return (x_min, y_min, x_max, y_max)


def rodrigues_rot(points, vector1, vector2):
    """RODRIGUES ROTATION
    - Rotate given points based on a starting and ending vector
    - Axis k and angle of rotation theta given by vectors n0,n1
    P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))"""

    if points.ndim == 1:
        points = points[np.newaxis, :]

    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector2)
    k = np.cross(vector1, vector2)
    if np.sum(k) != 0:
        k = k / np.linalg.norm(k)
    theta = np.arccos(np.dot(vector1, vector2))

    # MATRIX MULTIPLICATION
    P_rot = np.zeros((len(points), 3))
    for i in range(len(points)):
        P_rot[i] = (
            points[i] * np.cos(theta)
            + np.cross(k, points[i]) * np.sin(theta)
            + k * np.dot(k, points[i]) * (1 - np.cos(theta))
        )
    return P_rot


def line_plane_intersection(plane_point, plane_normal, line_point, line_direciton, epsilon=1e-6):
    """Compute the intersection of a line with a plane."""
    ndotu = plane_normal.dot(line_direciton)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = line_point - plane_point
    si = -plane_normal.dot(w) / ndotu
    intersection_point = w + si * line_direciton + plane_point

    return intersection_point
