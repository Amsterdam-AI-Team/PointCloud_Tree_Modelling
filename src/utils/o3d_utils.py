# Urban_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

import numpy as np
import laspy
import open3d as o3d

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

def trace_back(trace, ind):
    trace_ind = np.hstack([trace[i] for i in ind])
    return trace_ind
