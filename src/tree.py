import os
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

from misc.smallestenclosingcircle import make_circle
from misc.fitcyclinders import fit_cylinders_to_stem
import utils.o3d_utils as o3d_utils
import utils.graph_utils as graph_utils
import utils.math_utils as math_utils


from labels import Labels

# class TreeAnaylsis:
#     """
#     Convenience class for point cloud tree analysis.
#     """

tree_colors = {
            'stem': [0.36,0.25, 0.2],
            'foliage': [0,0.48,0],
            'wood': [0.45, 0.23, 0.07]
}

# TREE PROCESSING
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


def reconstruct_skeleton(tree_cloud, exe_path):
    """Function to reconstruct tree skeleton from o3d point cloud using adTree."""

    # create input file system
    tmp_folder = './tmp'
    in_file = os.path.join(tmp_folder, 'tree.xyz')
    out_file = os.path.join(tmp_folder, 'tree_skeleton.ply')
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    try:
        tree_cloud_sampled = tree_cloud.voxel_down_sample(0.04)
        o3d.io.write_point_cloud(in_file, tree_cloud_sampled) # write input file
        subprocess.run(
                [exe_path, in_file, out_file],
                capture_output=True,
                check=True
                )

        # read output graph
        graph, vertices, edges = graph_utils.read_ply(out_file)

    except subprocess.CalledProcessError as error_msg:
        print(f"Failed reconstructing tree:\n {error_msg.stderr.decode('utf-8')}")
    except Exception as error_msg:
        print(f"Failed:\n {error_msg}")

    # clean filesystem
    if os.path.exists(in_file):
        os.remove(in_file)
    if os.path.exists(out_file):
        os.remove(out_file)
    if os.path.isdir(tmp_folder):
        os.rmdir(tmp_folder)

    skeleton = {
        'graph': graph,
        'vertices': vertices,
        'edges': edges
    }

    return skeleton


def skeleton_split(tree_cloud, skeleton_graph):
    """Function to split the stem from the crown using the reconstructed tree skeleton."""

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


# CROWN ANALYSIS
def crown_to_mesh(crown_cloud, method, alpha=.8):
    """Function to convert to o3d crown point cloud to a mesh."""

    crown_cloud_sampled = crown_cloud.voxel_down_sample(0.25)

    if method == 'alphashape':
        pts = np.asarray(crown_cloud_sampled.points)
        mesh = alphashape(pts, alpha)
        clean_points, clean_faces = pymeshfix.clean_from_arrays(mesh.vertices,  mesh.faces)
        mesh = trimesh.base.Trimesh(clean_points, clean_faces)
        mesh.fix_normals()
        o3d_mesh = mesh.as_open3d
    else:
        o3d_mesh, _ = crown_cloud_sampled.compute_convex_hull()

    o3d_mesh.compute_vertex_normals()
    o3d_mesh.paint_uniform_color(tree_colors['foliage'])
    return o3d_mesh, o3d_mesh.get_volume()


def crown_diameter(crown_cloud):
    """Function to compute crown diameter from o3d crown point cloud."""

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

    return shape


def crown_height(crown_cloud):
    """Function to get the crown height."""
    return o3d_utils.cloud_height(crown_cloud)


def crown_base_height(crown_cloud, ahn_z=0):
    """Function to get base height of tree."""
    height = crown_cloud.get_min_bound()[2] - ahn_z
    return height


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
    stem_alphashape = shapely.affinity.translate(stem_alphashape, *(-tree_center))
    crown_alphashape = shapely.affinity.translate(crown_alphashape, *(-tree_center))

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

    mesh, volume = crown_to_mesh(crown_cloud, method)

    crown = {
        'crown_height': crown_height(crown_cloud),
        'crown_base_height': crown_base_height(crown_cloud),
        'crown_diameter': crown_diameter(crown_cloud),
        'crown_shape': Labels.get_str(crown_shape(crown_cloud)),
        'crown_volume': volume,
        'crown_mesh': mesh
    }

    return crown


# STEM ANALYSIS
def stem_height(stem_cloud):
    """Function to get the stem height."""
    return o3d_utils.cloud_height(stem_cloud)


def stem_angle(stem_cylinders):
    """Function to estimate stem angle given fitted cylinders"""
    return math_utils.vector_angle(stem_cylinders[-1,:3] - stem_cylinders[0,:3])

def stem_analysis(stem_cloud):
    """Function to analyse tree crown o3d point cloud."""

    cyl_array = fit_cylinders_to_stem(stem_cloud, .25)
    angle = stem_angle(cyl_array)
    mean_radius = cyl_array[:,3].mean()

    stem = {
        'stem_height': stem_height(stem_cloud),
        'stem_angle': angle,
        'stem_radius': mean_radius,
        'stem_circumference':  2*mean_radius * np.pi,
        'stem_CCI': (np.min(cyl_array[:,4]), np.max(cyl_array[:,4])),
        'stem_mesh': o3d_utils.mesh_from_cylinders(cyl_array, tree_colors['stem'])
    }

    return stem


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


def tree_separate(tree_cloud, adTree_exe, filter_leaves=None):
    """Function to split stem from o3d tree point cloud."""

    # 1. Classify and filter leaves (optional)
    labels = np.ones(len(tree_cloud.points), dtype=int)
    wood_cloud = tree_cloud
    if filter_leaves:
        print(f"Leaf-wood classification using `{filter_leaves}` method...")
        labels = leafwood_classificiation(tree_cloud, method=filter_leaves)
        wood_cloud = tree_cloud.select_by_index(np.where(labels==Labels.WOOD)[0])
        print(f"Done. {np.sum(labels==Labels.WOOD)}/{len(labels)} points wood.")

    # 2. Skeleton reconstruction
    print("Reconstructing tree skeleton...")
    skeleton = reconstruct_skeleton(wood_cloud, adTree_exe)
    print(f"Done. Skeleton constructed containing {len(skeleton['vertices'])} nodes")

    # 3. Stem-crow splitting
    print("Splitting stem form crown...")
    mask = skeleton_split(tree_cloud, skeleton['graph'])
    labels[mask] = Labels.STEM
    print(f"Done. {np.sum(mask)}/{len(labels)} points labeled as stem.")

    stem_cloud = tree_cloud.select_by_index(np.where(mask)[0])
    crown_cloud = tree_cloud.select_by_index(np.where(mask)[0], invert=True)

    return stem_cloud, crown_cloud


def analyse_tree(tree_cloud, adTree_exe, filter_leaves=None,
                 crown_model_method='alphashape', show_result=False):
    """Function to analyse o3d point cloud tree."""

    # 1. Classify and filter leaves (optional)
    labels = np.ones(len(tree_cloud.points), dtype=int)
    wood_cloud = tree_cloud
    if filter_leaves:
        print(f"Leaf-wood classification using `{filter_leaves}` method...")
        labels = leafwood_classificiation(tree_cloud, method=filter_leaves)
        wood_cloud = tree_cloud.select_by_index(np.where(labels==Labels.WOOD)[0])
        print(f"Done. {np.sum(labels==Labels.WOOD)}/{len(labels)} points wood.")

    # 2. Skeleton reconstruction
    print("Reconstructing tree skeleton...")
    skeleton = reconstruct_skeleton(wood_cloud, adTree_exe)
    print(f"Done. Skeleton constructed containing {len(skeleton['vertices'])} nodes")

    # 3. Stem-crow splitting
    print("Splitting stem form crown...")
    mask = skeleton_split(tree_cloud, skeleton['graph'])
    labels[mask] = Labels.STEM
    print(f"Done. {np.sum(mask)}/{len(labels)} points labeled as stem.")

    # 4. Analysis
    print("Crown & Stem Analysis...")
    stem_cloud = tree_cloud.select_by_index(np.where(mask)[0])
    crown_cloud = tree_cloud.select_by_index(np.where(mask)[0], invert=True)
    stem_stats = stem_analysis(stem_cloud)
    crown_stats = crown_analysis(crown_cloud, crown_model_method)
    tree_stats = {**stem_stats, **crown_stats}
    print("Done.")

    # Show tree
    if show_result:
        show_tree(tree_cloud, labels, skeleton)

    return tree_stats
