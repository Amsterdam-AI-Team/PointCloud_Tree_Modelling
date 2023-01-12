# Urban_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license
import math
import numpy as np
import laspy
import open3d as o3d
import trimesh
from misc.quaternion import Quaternion
import matplotlib.pyplot as plt
from alphashape import alphashape
from descartes import PolygonPatch


######## PointCloud utils ##########

def point_density(pcd):
    """Compute the average nearest neighbor distance for the point cloud."""
    dist = pcd.compute_nearest_neighbor_distance()  
    return np.round(np.mean(dist), 3)


def statistics(pcd):
    """Prints statistics on the point cloud."""
    density = point_density(pcd)
    n = len(pcd.points)
    x, y, z = np.round(pcd.get_max_bound() - pcd.get_min_bound(), 1)

    print(f"Point cloud of {n} points, ({x}x{y}x{z}), and {density} point density.")


def read_las(las_file, output_stats=True):
    """Read a las file and return the o3d.geometry.PointCloud object."""
    las = laspy.read(las_file)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(las.xyz))
    colors = np.vstack([las.red,las.green,las.blue]).T
    colors -= colors.min()
    colors = colors / colors.max()
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if output_stats:
        statistics(pcd)

    return pcd


def save_las(pcd, outfile): #, labels=[]):
    """Save a o3d.geometry.PointCloud as las file."""
    
    points = np.asarray(pcd.points)

    las = laspy.create(file_version="1.2", point_format=3)
    las.header.offsets = np.min(points, axis=0)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    # if len(labels) > 0:
    #     for name, values in labels:
    #         las.add_extra_dim(laspy.ExtraBytesParams(name=name, type="uint8",description=name))
    #         las[name] = values

    las.write(outfile)


def surface_variation_filter(pcd, radius, threshold):
    """Compute surface variation of point cloud."""
    pcd.estimate_covariances(
        search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))
    eig_val, _ = np.linalg.eig(np.asarray(pcd.covariances))
    eig_val = np.sort(eig_val, axis=1)
    sv = eig_val[:,0] / eig_val.sum(axis=1)
    mask = sv < threshold
    return mask


def curvature_filter(pcd, radius, min1=0, max1=100,
                 min2=0, max2=100, min3=0, max3=100):
    """Compute surface variation of point cloud."""

    # estimate eigenvalues
    pcd.estimate_covariances(
        search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))
    eig_val, _ = np.linalg.eig(np.asarray(pcd.covariances))
    eig_val = np.sort(eig_val, axis=1)
    eig_val[eig_val[:,2]==1] = np.zeros(3)
    L1, L2, L3 = eig_val[:,2], eig_val[:,1], eig_val[:,0]
    L1 = (L1 - L1.min()) / ((L1.max()-L1.min()) / 100)
    L2 = (L2 - L2.min()) / ((L2.max()-L2.min()) / 100)
    L3 = (L3 - L3.min()) / ((L3.max()-L3.min()) / 100)

    mask = (L1 > min1) & (L1 < max1) & \
             (L2 > min2) & (L2 < max2) & \
             (L3 > min3) & (L3 < max3)

    return mask

def display_inlier_outlier(cloud, ind, inlier_color=[.4,.4,.4], outlier_color=[1,0,0]):
    """Show point cloud with coloured inliers and outliers."""
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    outlier_cloud.paint_uniform_color(outlier_color)
    inlier_cloud.paint_uniform_color(inlier_color)
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def project(pcd, axis, voxel_size=None):
    """Project point cloud wrt axis and voxelize if wanted."""
    pts = np.array(pcd.points)
    pts[:,axis] = 0
    pcd_ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if voxel_size:
        pcd_ = pcd_.voxel_down_sample(voxel_size)
    pts = np.asarray(pcd_.points)[:,:2]
    return pts


def trace_back(trace, ind):
    trace_ind = np.hstack([trace[i] for i in ind])
    return trace_ind


def cloud_height(cloud):
    """Function to get cloud height."""
    height = cloud.get_max_bound()[2] - cloud.get_min_bound()[2]
    return height


######## MESH utils ##########

def simplify_mesh(mesh, num_triangles):
    return mesh.simplify_quadric_decimation(target_number_of_triangles=num_triangles)


def project_mesh(mesh):

    # make shape
    pts = np.array(mesh.vertices)[:,:2]
    shape = alphashape(pts, 0.8)
    mesh_center = np.hstack(shape.centroid.coords)

    # Initialize plot
    _, axes = plt.subplots()
    axes.plot(*mesh_center, marker='x', c='k')
    axes.add_patch(PolygonPatch(shape, alpha=.8, color='green', label='Stem'))
    axes.legend()
    axes.set_title('Tree Projection')
    plt.show()


def plot_mesh(mesh):
    colors = np.asarray(mesh.vertex_colors)
    if len(colors) > 0:
        color = colors[0]
    else:
        color = [.7,.7,.7]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(*zip(*mesh.vertices), triangles=mesh.triangles, color=color)
    ax.axis('equal')
    plt.show()


def show_mesh(mesh, color=None):
    if color:
        mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])


def show_mesh_cloud(mesh, cloud):

    lines = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    lines.paint_uniform_color([0.8, .2, 0])
    o3d.visualization.draw_geometries([cloud, lines])


def to_trimesh(mesh):
    return trimesh.base.Trimesh(mesh.vertices, mesh.triangles)


def mesh_from_cylinders(cyl_array, color=[0.7,0.7,0.7], resolution=15):
    """Function to construct o3d mesh from cylindrical sections."""
    circle_fits = [(rim[:3], rim[3]) for rim in cyl_array]
    num_slices = len(circle_fits)

    # compute directional axis
    Z = np.array([0, 0, 1], dtype=float)
    centers = np.array([c for c, r in circle_fits])
    axis = centers[-1] - centers[0]
    l = np.linalg.norm(axis)
    if l <= 1e-12:
        axis=Z

    rot = Quaternion.fromData(Z, axis).to_matrix()

    # create vertices
    angles = [2*math.pi*i/float(resolution) for i in range(resolution)]
    rim = np.array([[math.cos(theta), math.sin(theta), 0.0]
        for theta in angles])
    rim = np.dot(rot, rim.T).T

    rims = np.array([
        rim * r + c
        for c, r in circle_fits], dtype=float)
    rims = rims.reshape((-1,3))

    vertices = np.vstack([[centers[0], centers[-1]], rims ])

    # create faces
    bottom_fan = np.array([
        [0, (i+1)%resolution+2, i+2]
        for i in range(resolution) ], dtype=int)

    top_fan = np.array([
        [1, i+2+resolution*(num_slices-1), (i+1)%resolution+2+resolution*(num_slices-1)]
        for i in range(resolution) ], dtype=int)

    slice_fan = np.array([
        [[2+i, (i+1)%resolution+2, i+resolution+2],
            [i+resolution+2, (i+1)%resolution+2, (i+1)%resolution+resolution+2]]
        for i in range(resolution) ], dtype=int)
    slice_fan = slice_fan.reshape((-1, 3), order="C")

    side_fan = np.array([
        slice_fan + resolution*i 
        for i in range(num_slices-1)], dtype=int)
    side_fan = side_fan.reshape((-1, 3), order="C")

    faces = np.vstack([bottom_fan, top_fan, side_fan])

    # create mesh
    mesh = trimesh.base.Trimesh(vertices, faces).as_open3d
    mesh.paint_uniform_color(color)

    return mesh