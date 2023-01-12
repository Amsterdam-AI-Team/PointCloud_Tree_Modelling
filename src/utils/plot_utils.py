# Urban_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

"""Visualisation utilities."""

import numpy as np
import pandas as pd
import open3d as o3d
import os
import subprocess
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon


def plot_stem_cylinders(stem_cylinders, resolution=15, cloud=None):
    
        geometries = []
        for result in stem_cylinders:
            c = result[:3]
            r = result[3]

            line_points = np.zeros((0,3))
            for i in range(resolution):
                phi = i*np.pi/(resolution/2)
                line_points = np.vstack((line_points, (c[0] + r*np.cos(phi), c[1] + r*np.sin(phi), c[2])))

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector([(i,i+1) for i in range(resolution-1)])
            line_set.colors = o3d.utility.Vector3dVector(np.zeros((resolution-1,3)))
            geometries.append(line_set)

        if cloud is not None:
            geometries.append(cloud)

        o3d.visualization.draw_geometries(geometries)
