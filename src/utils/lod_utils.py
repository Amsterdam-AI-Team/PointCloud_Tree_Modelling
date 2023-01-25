#
# PROJECT NAME by Amsterdam Intelligence, GPL-3.0 license
#
import math
import trimesh
from alphashape import alphashape
from descartes import PolygonPatch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from misc.smallestenclosingcircle import make_circle
import utils.o3d_utils as o3d_utils
import utils.tree_utils as tree_utils


def lod_2(tree_cloud, stem_radius, tree_base, crown_base,
                   crown_height, resolution=6):
    """Function to generate LOD2 mesh."""
    # Load stats
    # stem_radius = tree_stats['DBH']/2
    # tree_base = tree_stats['stem_startpoint']
    # crown_base = tree_stats['stem_endpoint']
    # crown_height = tree_stats['crown_height']
    # crown_mesh = tree_stats['crown_mesh']

    tree_top = crown_base + np.array([0,0, crown_height])

    # construct stem rims
    angles = [2*math.pi*i/float(resolution) for i in range(resolution)]
    stem_bottom_rim = np.array([
        np.array([math.cos(theta), math.sin(theta), 0.0]) * stem_radius + tree_base
        for theta in angles], dtype=float).reshape((-1,3))
    stem_top_rim = np.array([
        np.array([math.cos(theta), math.sin(theta), 0.0]) * stem_radius + crown_base
        for theta in angles], dtype=float).reshape((-1,3))

    # construct crown rims
    points = np.array(tree_cloud.points)
    points -= np.hstack([crown_base[:2], 0])
    z_bins = np.linspace(crown_base[2], tree_top[2], 20, endpoint=False)
    digi = np.digitize(points[:,2], z_bins)

    cyl_arrays = []
    for i in range(1, 20):
        if np.sum(digi==i) > 0:
            r = np.max(np.abs(np.linalg.norm(points[digi==i][:,:2], axis=1)))
            center = np.hstack([crown_base[:2], (z_bins[i]+z_bins[i-1])/2])
            cyl_arrays.append((center, r))

    periphery = np.argmax([r for _, r in cyl_arrays])
    lower_periphery = int(periphery/2)
    higher_periphery = int(periphery + (len(cyl_arrays)-periphery)/2)
    cyl_arrays = [cyl_arrays[lower_periphery], cyl_arrays[periphery], cyl_arrays[higher_periphery]]

    crown_rims = np.zeros((0,3))
    for c, r in cyl_arrays:
        crown_rim = np.array([
            np.array([math.cos(theta), math.sin(theta), 0.0]) * r + c
            for theta in angles], dtype=float).reshape((-1,3))
        crown_rims = np.vstack([crown_rims, crown_rim])

    vertices = np.vstack([[tree_base, tree_top],
                            stem_bottom_rim,
                            stem_top_rim,
                            crown_rims])

    # create faces
    num_slices = 4
    bottom_fan = np.array([
        [0, (i+1)%resolution+2, i+2]
        for i in range(resolution) ], dtype=int)

    top_fan = np.array([
        [1, i+2+resolution*num_slices, (i+1)%resolution+2+resolution*num_slices]
        for i in range(resolution) ], dtype=int)

    rim_fan = np.array([
        [[2+i, (i+1)%resolution+2, i+resolution+2],
            [i+resolution+2, (i+1)%resolution+2, (i+1)%resolution+resolution+2]]
        for i in range(resolution) ], dtype=int)
    rim_fan = rim_fan.reshape((-1, 3), order="C")

    side_fan = np.array([
        rim_fan + resolution*i 
        for i in range(num_slices)], dtype=int)
    side_fan = side_fan.reshape((-1, 3), order="C")

    faces = np.vstack([bottom_fan, top_fan, side_fan])

    # create mesh
    lod = trimesh.base.Trimesh(vertices, faces).as_open3d
    lod.paint_uniform_color(tree_utils.tree_colors['stem'])
    lod.compute_vertex_normals()
    mesh_colors = np.full((len(lod.vertices),3), tree_utils.tree_colors['foliage'])
    mesh_colors[0] = tree_utils.tree_colors['stem']
    mesh_colors[2:resolution*2+2] = tree_utils.tree_colors['stem']

    lod.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

    return lod


def lod_3(stem_radius, tree_base, crown_base,
                   crown_mesh, resolution=10, crown_steps=1.5):
    """Function to generate LOD3 mesh."""
    
    # Fit cylinder to stem
    # cyl_radius = tree_stats['DBH']/2
    # stem_center_min = tree_stats['stem_startpoint']
    # stem_center_max = tree_stats['stem_endpoint']
    # crown_mesh = tree_stats['crown_mesh']

    # crown top point (TODO: compare alternatives...)
    hull_mesh = crown_mesh.compute_convex_hull()[0]
    crown_trimesh = o3d_utils.to_trimesh(hull_mesh)
    ray_origin = np.hstack([crown_base[:2], hull_mesh.get_center()[2]])
    ray_direction = np.array([[0,0,1]]) # TODO: or stem_axis ??
    crown_center_max = crown_trimesh.ray.intersects_location([ray_origin], ray_direction)[0][0]

    # construct stem rims
    angles = [2*math.pi*i/float(resolution) for i in range(resolution)]
    stem_bottom_rim = np.array([
        np.array([math.cos(theta), math.sin(theta), 0.0]) * stem_radius + tree_base
        for theta in angles], dtype=float).reshape((-1,3))

    stem_top_rim = np.array([
        np.array([math.cos(theta), math.sin(theta), 0.0]) * stem_radius + crown_base
        for theta in angles], dtype=float).reshape((-1,3))

    # construct crown rims (ray projection)
    crown_rims = np.zeros((0,3))
    lenght = np.linalg.norm(crown_center_max - crown_base) + crown_steps
    crown_ray_origins = np.linspace(crown_base, crown_center_max, int(lenght/crown_steps))[1:]
    for ray_origin in crown_ray_origins:
        ray_origin = ray_origin.reshape(1,3)
        for theta in angles:
            ray_direction = np.array([[math.cos(theta), math.sin(theta), 0.0]])
            locations = crown_trimesh.ray.intersects_location(ray_origin, ray_direction)[0]
            idx = np.argmax(np.linalg.norm(locations - ray_origin, axis=1))
            crown_rims = np.vstack([crown_rims, locations[idx]])

    vertices = np.vstack([[tree_base, crown_center_max],
                            stem_bottom_rim,
                            stem_top_rim,
                            crown_rims])

    # create faces
    num_slices = len(crown_ray_origins) + 1
    bottom_fan = np.array([
        [0, (i+1)%resolution+2, i+2]
        for i in range(resolution) ], dtype=int)

    top_fan = np.array([
        [1, i+2+resolution*(num_slices-1), (i+1)%resolution+2+resolution*(num_slices-1)]
        for i in range(resolution) ], dtype=int)

    rim_fan = np.array([
        [[2+i, (i+1)%resolution+2, i+resolution+2],
            [i+resolution+2, (i+1)%resolution+2, (i+1)%resolution+resolution+2]]
        for i in range(resolution) ], dtype=int)
    rim_fan = rim_fan.reshape((-1, 3), order="C")

    side_fan = np.array([
        rim_fan + resolution*i 
        for i in range(num_slices-1)], dtype=int)
    side_fan = side_fan.reshape((-1, 3), order="C")

    faces = np.vstack([bottom_fan, top_fan, side_fan])

    # create mesh
    lod = trimesh.base.Trimesh(vertices, faces).as_open3d
    lod.paint_uniform_color(tree_utils.tree_colors['stem'])
    lod.compute_vertex_normals()
    mesh_colors = np.full((len(lod.vertices),3), tree_utils.tree_colors['foliage'])
    mesh_colors[0] = tree_utils.tree_colors['stem']
    mesh_colors[2:resolution*2+2] = tree_utils.tree_colors['stem']

    lod.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

    return lod


def lod_31(stem_mesh, crown_mesh):
    """Function to generate LoD mesh."""

    # merge
    lod = stem_mesh + crown_mesh

    # increase_stem
    t = np.asarray(lod.triangles)
    for idx in np.unique(t[np.isin(t, 1).any(axis=1)])[1:]:
        lod.vertices[idx][2] += .5

    return lod
