# 
# tree utils - Library (Python)
# 
# Copyright (c) 2023 Project AI
# 

import os
import math
import string
import random
import trimesh
import shapely
import pymeshfix
import subprocess
import numpy as np
import open3d as o3d
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from descartes import PolygonPatch
from alphashape import alphashape

import utils.o3d_utils as o3d_utils
import utils.graph_utils as graph_utils
import utils.math_utils as math_utils
import utils.plot_utils as plot_utils
import utils.clip_utils as clip_utils
from misc.smallestenclosingcircle import make_circle
from misc.fitcyclinders import fit_cylinders_to_stem, fit_vertical_cylinder_3D

import logging

logger = logging.getLogger()

from labels import Labels

# class TreeAnaylsis:
#     """
#     Convenience class for point cloud tree analysis.
#     """

breastheight = 1.3
tree_colors = {
            'stem': [0.36,0.25, 0.2],
            'foliage': [0,0.48,0],
            'wood': [0.45, 0.23, 0.07]
}

# -----------------
# - Tree Analysis -
# -----------------
def leafwood_classificiation(tree_cloud, method):
    """Leaf-wood classification."""

    labels = np.full(len(tree_cloud.points), Labels.LEAF, dtype=int)

    # outlier removal
    pcd_, _, trace = tree_cloud.voxel_down_sample_and_trace(0.02,
                                     tree_cloud.get_min_bound(),
                                     tree_cloud.get_max_bound())
    pcd_, ind_ = pcd_.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)
    ind_ = np.asarray(ind_)

    # classify
    if method == 'curvature':
        mask = o3d_utils.curvature_filter(pcd_, .075, min1=20, min2=35)
        ind = np.hstack([trace[i] for i in ind_[mask]])
    else:
        mask = o3d_utils.surface_variation_filter(pcd_, .1, .15)
        ind = np.hstack([trace[i] for i in ind_[mask]])

    labels[ind] = Labels.WOOD

    return labels


def reconstruct_skeleton(tree_cloud, exe_path, filename=''):
    """Function to reconstruct tree skeleton from o3d point cloud using adTree."""

    if len(filename) == '':
        filename = ''.join(random.choice(string.ascii_letters) for i in range(8))

    # create input file system
    tmp_folder = './tmp'
    in_file = os.path.join(tmp_folder, filename + '.xyz')
    out_file = os.path.join(tmp_folder, filename + '_skeleton.ply')
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    try:
        tree_cloud_sampled = tree_cloud.voxel_down_sample(0.02)
        o3d.io.write_point_cloud(in_file, tree_cloud_sampled) # write input file
        subprocess.run(
                [exe_path, in_file, out_file],
                capture_output=True,
                check=True
                )

        # read output graph
        graph, vertices, edges = graph_utils.read_ply(out_file)

    except subprocess.CalledProcessError as error_msg:
        logger.info(f"Failed reconstructing tree:\n {error_msg.stderr.decode('utf-8')}")
    except Exception as error_msg:
        logger.info(f"Failed:\n {error_msg}")

    # clean filesystem
    if os.path.exists(in_file):
        os.remove(in_file)
    if os.path.exists(out_file):
        os.remove(out_file)

    skeleton = {
        'graph': graph,
        'vertices': vertices,
        'edges': edges
    }

    return skeleton


def skeleton_split(tree_cloud, skeleton_graph):
    """Function to split the stem from the crown using the reconstructed tree skeleton."""
    try:
        # get start node and retrieve path
        z_values = nx.get_node_attributes(skeleton_graph, 'z')
        start_node = min(z_values, key=z_values.get)
        path = graph_utils.path_till_split(skeleton_graph, start_node)
        skeleton_pts = np.array([list(skeleton_graph.nodes[node].values()) for node in path])

        # Filter cloud for stem points
        tree_points = np.array(tree_cloud.points)
        labels = np.zeros(len(tree_points), dtype=bool)
        mask_idx = np.where(tree_points[:,2] < skeleton_pts[:,2].max())[0]
        
        # TODO Filter tree points
        tree = KDTree(tree_points[mask_idx])
        selection = set()
        num_ = int(np.linalg.norm(skeleton_pts[1]-skeleton_pts[0]) / 0.05)
        skeleton_pts = np.linspace(start=skeleton_pts[0], stop=skeleton_pts[-1], num=num_)
        for result in tree.query_ball_point(skeleton_pts, .75):
            selection.update(result)
        selection = mask_idx[list(selection)]
        labels[selection] = True

        return labels

    except Exception as e:
        logger.error('Error at %s', 'tree_utils error', exc_info=e)
        return None


def tree_separate(tree_cloud, adTree_exe, filter_leaves=None):
    """Function to split stem from o3d tree point cloud."""

    # 1. Classify and filter leaves (optional)
    labels = np.ones(len(tree_cloud.points), dtype=int)
    wood_cloud = tree_cloud
    if filter_leaves:
        logger.info(f"Leaf-wood classification using `{filter_leaves}` method...")
        labels = leafwood_classificiation(tree_cloud, method=filter_leaves)
        wood_cloud = tree_cloud.select_by_index(np.where(labels==Labels.WOOD)[0])
        logger.info(f"Done. {np.sum(labels==Labels.WOOD)}/{len(labels)} points wood.")

    # 2. Skeleton reconstruction
    logger.info("Reconstructing tree skeleton...")
    skeleton = reconstruct_skeleton(wood_cloud, adTree_exe)
    logger.info(f"Done. Skeleton constructed containing {len(skeleton['vertices'])} nodes")

    # 3. Stem-crow splitting
    logger.info("Splitting stem form crown...")
    mask = skeleton_split(tree_cloud, skeleton['graph'])
    labels[mask] = Labels.STEM
    logger.info(f"Done. {np.sum(mask)}/{len(labels)} points labeled as stem.")

    stem_cloud = tree_cloud.select_by_index(np.where(mask)[0])
    crown_cloud = tree_cloud.select_by_index(np.where(mask)[0], invert=True)

    return stem_cloud, crown_cloud


# ------------------
# - Crown Analysis -
# ------------------
def crown_to_mesh(crown_cloud, method, alpha=.8):
    """Function to convert to o3d crown point cloud to a mesh."""

    try:
        

        if method == 'alphashape':
            crown_cloud_sampled = crown_cloud.voxel_down_sample(0.4)
            pts = np.asarray(crown_cloud_sampled.points)
            mesh = alphashape(pts, alpha)
            clean_points, clean_faces = pymeshfix.clean_from_arrays(mesh.vertices,  mesh.faces)
            mesh = trimesh.base.Trimesh(clean_points, clean_faces)
            mesh.fix_normals()
            o3d_mesh = mesh.as_open3d
        else:
            crown_cloud_sampled = crown_cloud.voxel_down_sample(0.2)
            o3d_mesh, _ = crown_cloud_sampled.compute_convex_hull()

        o3d_mesh.compute_vertex_normals()
        o3d_mesh.paint_uniform_color(tree_colors['foliage'])
        return o3d_mesh, o3d_mesh.get_volume()

    except Exception as e:
        logger.error('Error at %s', 'tree_utils error', exc_info=e)
        return None, None


def crown_diameter(crown_cloud):
    """Function to compute crown diameter from o3d crown point cloud."""

    try:
        proj_pts = o3d_utils.project(crown_cloud, 2, .2)
        radius = make_circle(proj_pts)[2]

        # Visualize
        # fig, ax = plt.subplots(figsize=(6, 6))
        # circle = Circle((x,y), r, facecolor='none',
        #                 edgecolor=(.8, .2, .1), linewidth=3, alpha=0.5)
        # ax.add_patch(circle)
        # ax.scatter(proj_pts[:,0],proj_pts[:,1], color=(0,0.5,0), s=.3)
        # ax.plot(x,y, marker='x', c='k', markersize=5)
        # plt.show()

        return radius*2
    except Exception as e:
        logger.error('Error at %s', 'tree_utils error', exc_info=e)
        return None


def crown_shape(crown_cloud):
    '''
    Method to define the shape of the crown. The crown can either be:
    conical, inverse conical, spherical, or cylidrical.

    Defined based on the relation between radius at a, b, c.
        |<- a ->|
        |       |
        |<- b ->|
        |       |
        |<- c ->|
    '''

    try:
        # Estimate shape parameters
        crown_sampled = crown_cloud.voxel_down_sample(0.05)
        pts = np.asarray(crown_sampled.points)[:,:3]
        min_z, max_z = pts[:, 2].min(), pts[:, 2].max()
        step_size = (max_z - min_z) / 100
        bins = np.arange(min_z, max_z, step_size)
        slice_ind = np.digitize(pts[:,2], bins, right=True)

        slice_pts = pts[slice_ind >= 95,:2]
        a_rd = make_circle(slice_pts)[2]
        slice_pts = pts[(slice_ind >= 45) & (slice_ind <= 55),:2]
        b_rd = make_circle(slice_pts)[2]
        slice_pts = pts[slice_ind <= 5,:2]
        c_rd = make_circle(slice_pts)[2]

        # Classify shape
        shape = Labels.CYLINDRICAL
        if abs(a_rd - b_rd) < .2 and abs(b_rd - c_rd) < .2 and abs(a_rd - c_rd) < .2:
            shape = Labels.CYLINDRICAL
        elif a_rd > b_rd > c_rd:
            shape = Labels.CONICAL
        elif a_rd < b_rd < c_rd:
            shape = Labels.INVERSE_CONICAL
        elif a_rd < b_rd > c_rd:
            shape = Labels.SPHERICAL

        return Labels.get_str(shape)

    except Exception as e:
        logger.error('Error at %s', 'tree_utils error', exc_info=e)
        return None


def crown_height(crown_cloud):
    """Function to get the crown height."""
    try:
        return o3d_utils.cloud_height(crown_cloud)
    except Exception as e:
        logger.error('Error at %s', 'tree_utils error', exc_info=e)
        return None


def crown_base_height(crown_cloud, ahn_z):
    """Function to get base height of tree."""
    try:
        height = crown_cloud.get_min_bound()[2] - ahn_z
        return height
    except Exception as e:
        logger.error('Error at %s', 'tree_utils error', exc_info=e)
        return None


def project_tree(crown_cloud, stem_cloud):
    """TODO"""

    # Stem shape
    pts = o3d_utils.project(stem_cloud, 2, 0.02)
    stem_alphashape = alphashape.alphashape(pts, 0.5)
    stem_alphashape = stem_alphashape.buffer(0.01)
    tree_center = np.hstack(stem_alphashape.centroid.coords)

    # Crown shape
    pts = o3d_utils.project(crown_cloud, 2, 0.3)
    crown_alphashape = alphashape.alphashape(pts, 2)
    crown_alphashape = crown_alphashape.buffer(0.1)

    # Center
    pts_shifted = pts-[tree_center]
    stem_alphashape = shapely.affinity.translate(stem_alphashape, *(-1 * tree_center))
    crown_alphashape = shapely.affinity.translate(crown_alphashape, *(-1 * tree_center))

    # Initialize plot
    _, axes = plt.subplots()
    axes.plot(0, marker='x', c='k', markersize=5)
    axes.add_patch(PolygonPatch(crown_alphashape, alpha=.15, color='green', label='Crown'))
    axes.add_patch(PolygonPatch(stem_alphashape, alpha=.8, color='brown', label='Stem'))
    lim = np.max(np.abs(pts_shifted))
    axes.set_xlim(-lim-.5,lim+.5)
    axes.set_ylim(-lim-.5,lim+.5)
    axes.legend()
    axes.set_title('Tree Projection')
    plt.show()


def crown_analysis(crown_cloud, method):
    """Function to analyse tree crown o3d point cloud."""

    stats = {}
    # crown analysis
    mesh, volume = crown_to_mesh(crown_cloud, method)
    stats['crown_height'] = crown_height(crown_cloud)
    stats['crown_baseheight'] = crown_base_height(crown_cloud, stats['stem_startpoint'][2])
    stats['crown_diameter'] = crown_diameter(crown_cloud)
    stats['crown_shape'] = Labels.get_str(crown_shape(crown_cloud))
    stats['crown_volume'] = volume
    stats['crown_mesh'] = mesh

    return stats


# -----------------
# - Stem Analysis -
# -----------------

# TODO: aanpassen
def stem_to_mesh(stem_cloud):
    """Function to covert stem point cloud to mesh."""
    try:
        cyl_array = fit_cylinders_to_stem(stem_cloud, .3)
        return o3d_utils.mesh_from_cylinders(cyl_array, tree_colors['stem'])
    except Exception as e:
        logger.error('Error at %s', 'tree_utils error', exc_info=e)
        return None


def stem_height(stem_cloud, ahn_z):
    """Function to get the stem height."""
    try:
        height = stem_cloud.get_max_bound()[2] - ahn_z
        return height
    except Exception as e:
        logger.error('Error at %s', 'tree_utils error', exc_info=e)
        return None


def stem_angle(stem_cylinders):
    """Function to estimate stem angle given fitted cylinders"""
    try:
        return math_utils.vector_angle(stem_cylinders[-1,:3] - stem_cylinders[0,:3])
    except Exception as e:
        logger.error('Error at %s', 'tree_utils error', exc_info=e)
        return None


def stem_bearing(stem_cylinders):
    """Function to estimate sten angle bearing"""
    try:
        return math_utils.vector_bearing(stem_cylinders[-1,:2] - stem_cylinders[0,:2])
    except Exception as e:
        logger.error('Error at %s', 'tree_utils error', exc_info=e)
        return None


def diameter_at_breastheight(stem_cloud):
    """Function to estimate diameter at breastheight."""
    try:
        stem_points = np.asarray(stem_cloud.points)
        z = stem_points[:,2].min() + breastheight

        # clip slice
        mask = clip_utils.axis_clip(stem_points, 2, z-.15, z+.15)
        stem_slice = stem_points[mask]
        if len(stem_slice) < 20:
            return None

        # fit cylinder
        radius = fit_vertical_cylinder_3D(stem_slice, .04)[2]

        return 2*radius
    except Exception as e:
        logger.error('Error at %s', 'tree_utils error', exc_info=e)
        return None


def get_stem_endpoints(stem_cloud, ground_cloud):
    """Function to get stem endpoints."""
    try:
        # fit cylinder to stem
        stem_cloud_voxeld = stem_cloud.voxel_down_sample(0.04)
        stem_points = np.array(stem_cloud_voxeld.points)
        cyl_center, cyl_axis, cyl_radius = fit_vertical_cylinder_3D(stem_points, .05)[:3]

        # stem start point
        ground_mesh = o3d_utils.surface_mesh_creation(ground_cloud)
        ground_trimesh = o3d_utils.to_trimesh(ground_mesh)
        ray_direction = [-np.sign(cyl_axis[2]) * cyl_axis]
        locations, _, _ = ground_trimesh.ray.intersects_location([cyl_center], ray_direction)
        start_point = locations[np.argmax(locations[:,2])]

        # stem endpoint
        end_point = math_utils.line_plane_intersection(
                            np.array([0,0,stem_cloud.get_max_bound()[2]]),
                            np.array([0,0,1]),
                            cyl_center,
                            cyl_axis)

        return start_point, end_point
    except Exception as e:
        logger.error('Error at %s', 'tree_utils error', exc_info=e)
        return stem_cloud.get_min_bound(), stem_cloud.get_max_bound()


def stem_analysis(stem_cloud, ground_cloud, stats):
    """Function to analyse tree crown o3d point cloud."""

    
    # stem stats
    stats['stem_startpoint'], stats['stem_endpoint'] = get_stem_endpoints(stem_cloud, ground_cloud)
    stats['stem_height'] = stats['stem_endpoint'][2] - stats['stem_startpoint'][2]
    stats['stem_angle'] = math_utils.vector_angle(stats['stem_endpoint'] - stats['stem_startpoint'])

    # diameter at breastheight
    dbh = diameter_at_breastheight(stem_cloud)
    stats['DBH'] = dbh
    stats['circumference_BH'] = dbh * np.pi

    # stem analysis
    cyl_array = fit_cylinders_to_stem(stem_cloud, .25)
    stats['stem_CCI'] = (np.min(cyl_array[:,4]), np.max(cyl_array[:,4]))
    stats['stem_mesh'] = o3d_utils.mesh_from_cylinders(cyl_array, tree_colors['stem'])

    return stats


# -----------------
# - Visualization -
# -----------------
def show_tree(cloud, labels, skeleton=None):
    """Show point cloud with coloured inliers and outliers."""

    # Leafs
    leafs_cloud = cloud.select_by_index(np.where(labels==Labels.LEAF)[0])
    leafs_cloud.paint_uniform_color(tree_colors['foliage'])

    # Wood
    wood_cloud = cloud.select_by_index(np.where(labels==Labels.WOOD)[0])
    wood_cloud.paint_uniform_color(tree_colors['wood'])

    # Stem
    stem_cloud = cloud.select_by_index(np.where(labels==Labels.STEM)[0])
    stem_cloud.paint_uniform_color(tree_colors['stem'])

    o3d_geometries = [leafs_cloud, wood_cloud, stem_cloud]

    # Skeleton
    if skeleton:
        colors = [[.8, 0.35, 0] for i in range(len(skeleton['edges']))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(skeleton['vertices'])
        line_set.lines = o3d.utility.Vector2iVector(skeleton['edges'])
        line_set.colors = o3d.utility.Vector3dVector(colors)

        skeleton_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(skeleton['vertices']))
        skeleton_cloud = skeleton_cloud.paint_uniform_color([0,0,0])

        o3d_geometries.extend([line_set, skeleton_cloud])

    o3d.visualization.draw_geometries(o3d_geometries)


# --------------
# - Processing -
# --------------
def process_tree(tree_cloud, ground_cloud, adTree_exe, filter_leaves=None):
    """Function to analyse o3d point cloud tree."""

    tree_data = {}

    # 1. Classify and filter leaves (optional)
    labels = np.ones(len(tree_cloud.points), dtype=int)
    wood_cloud = tree_cloud
    if filter_leaves:
        logger.info(f"Leaf-wood classification using `{filter_leaves}` method...")
        labels = leafwood_classificiation(tree_cloud, method=filter_leaves)
        wood_cloud = tree_cloud.select_by_index(np.where(labels==Labels.WOOD)[0])
        logger.info(f"Done. {np.sum(labels==Labels.WOOD)}/{len(labels)} points wood.")

    # 2. Skeleton reconstruction
    logger.info("Reconstructing tree skeleton...")
    tree_data['skeleton'] = reconstruct_skeleton(wood_cloud, adTree_exe)
    logger.info(f"Done. Skeleton constructed containing {len(tree_data['skeleton']['vertices'])} nodes")

    # 3. Stem-crow splitting
    logger.info("Splitting stem form crown...")
    mask = skeleton_split(tree_cloud, tree_data['skeleton']['graph'])
    stem_cloud = tree_cloud.select_by_index(np.where(mask)[0])
    crown_cloud = tree_cloud.select_by_index(np.where(mask)[0], invert=True)
    labels[mask] = Labels.STEM
    logger.info(f"Done. {np.sum(mask)}/{len(labels)} points labeled as stem.")

    # 4. Analysis
    logger.info("Stem Analysis...")
    tree_data['stem_basepoint'], tree_data['crown_basepoint'] = get_stem_endpoints(stem_cloud, ground_cloud)
    tree_data['stem_height'] = tree_data['crown_basepoint'][2] - tree_data['stem_basepoint'][2]
    tree_data['stem_angle'] = math_utils.vector_angle(tree_data['crown_basepoint'] - tree_data['stem_basepoint'])
    tree_data['DBH'] = diameter_at_breastheight(stem_cloud)
    tree_data['CBH'] = tree_data['DBH'] * np.pi if tree_data['DBH'] is not None else None
    tree_data['stem_mesh'] = stem_to_mesh(stem_cloud)
    logger.info("Done.")

    # 5. Crown analysis
    logger.info("Crown Analysis...")
    tree_data['crown_height'] = crown_height(crown_cloud)
    tree_data['crown_baseheight'] = crown_base_height(crown_cloud, tree_data['stem_basepoint'][2])
    tree_data['crown_diameter'] = crown_diameter(crown_cloud)
    tree_data['crown_shape'] = crown_shape(crown_cloud)
    tree_data['crown_mesh-convex'], tree_data['crown_volume-convex'] = crown_to_mesh(crown_cloud, 'convex_hull')
    tree_data['crown_mesh-alpha'], tree_data['crown_volume-alpha'] = crown_to_mesh(crown_cloud, 'alphashape')
    tree_data['tree_height'] = tree_data['crown_baseheight'] + tree_data['crown_height']
    logger.info("Done.")

    # LoD generation
    # logger.info("LoD generation...")
    # tree_data['LoD'] = generate_lod(tree_data)
    # logger.info("Done.")

    return tree_data, labels
