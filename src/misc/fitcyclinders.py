# 
# Fit cylinders - Library (Python)
# 
import warnings
import open3d as o3d
import trimesh
import numpy as np
import utils.math_utils as math_utils
from scipy.optimize import leastsq
from scipy.spatial.transform import Rotation as R


def fit_vertical_cylinder_3D(xyz, th):
        """
        This is a fitting for a vertical cylinder fitting
        Reference:
        http://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XXXIX-B5/169/2012/isprsarchives-XXXIX-B5-169-2012.pdf

        xyz is a matrix contain at least 5 rows, and each row stores x y z of a cylindrical surface
        p is initial values of the parameter;
        p[0] = Xc, x coordinate of the cylinder centre
        P[1] = Yc, y coordinate of the cylinder centre
        P[2] = alpha, rotation angle (radian) about the x-axis
        P[3] = beta, rotation angle (radian) about the y-axis
        P[4] = r, radius of the cylinder

        th, threshold for the convergence of the least squares

        """
        xyz_mean = np.mean(xyz, axis=0)
        xyz_centered = xyz - xyz_mean
        x = xyz_centered[:,0]
        y = xyz_centered[:,1]
        z = xyz_centered[:,2]

        # init parameters
        p = [0, 0, 0, 0, max(np.abs(y).max(), np.abs(x).max())]

        # fit
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            fitfunc = lambda p, x, y, z: (- np.cos(p[3])*(p[0] - x) - z*np.cos(p[2])*np.sin(p[3]) - np.sin(p[2])*np.sin(p[3])*(p[1] - y))**2 + (z*np.sin(p[2]) - np.cos(p[2])*(p[1] - y))**2 #fit function
            errfunc = lambda p, x, y, z: fitfunc(p, x, y, z) - p[4]**2 #error function 
            est_p = leastsq(errfunc, p, args=(x, y, z), maxfev=1000)[0]
            inliers = np.where(errfunc(est_p,x,y,z)<th)[0]
        
        # convert
        center = np.array([est_p[0],est_p[1],0]) + xyz_mean
        radius = est_p[4]
        
        rotation = R.from_rotvec([est_p[2], 0, 0])
        axis = rotation.apply([0,0,1])
        rotation = R.from_rotvec([0, est_p[3], 0])
        axis = rotation.apply(axis)

        # circumferential completeness index (CCI)
        P_xy = math_utils.rodrigues_rot(xyz_centered, axis, [0, 0, 1])
        CCI = math_utils.circumferential_completeness_index([est_p[0], est_p[1]], radius, P_xy)
        
        # visualize
        # voxel_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        # mesh = trimesh.creation.cylinder(radius=radius,
        #                  sections=20, 
        #                  segment=(center+axis*z.min(),center+axis*z.max())).as_open3d
        # mesh_lines = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        # mesh_lines.paint_uniform_color((0, 0, 0))

        # inliers_pcd = voxel_cloud.select_by_index(inliers)
        # inliers_pcd.paint_uniform_color([0,1,0])
        # outlier_pcd = voxel_cloud.select_by_index(inliers, invert=True)
        # outlier_pcd.paint_uniform_color([1,0,0])

        # o3d.visualization.draw_geometries([inliers_pcd, outlier_pcd, mesh_lines])

        return center, axis, radius, inliers, CCI


def fit_cylinders_to_stem(stem_cloud, slice_thickness):
    """
    Fits a 3D line to the skeleton points cluster provided.
    Uses this line as the major axis/axial vector of the cylinder to be fitted.
    Fits a series of circles perpendicular to this axis
    to the point cloud of this particular stem segment.
    Args:
        pcd: The cluster of points belonging to the segment of the branch.
    Returns:
        cyl_array: a numpy array based representation of the fitted circles/cylinders.
    """

    # fit 3D cylinder
    stem_cloud_sampled = stem_cloud.voxel_down_sample(0.01)
    points = np.asarray(stem_cloud_sampled.points)[:,:3]
    min_z, max_z = points[:,2].min(), points[:,2].max()
    center, axis, _, _, _ = fit_vertical_cylinder_3D(points, .1)

    # slice cylinder
    b_center = center + axis * (min_z - center[2] + slice_thickness/2),
    t_center = center + axis * (max_z - center[2] - slice_thickness/2)
    length = np.linalg.norm(t_center - b_center)
    line_centers = np.linspace(b_center, t_center, int(length / slice_thickness))

    # fit cylinders per slice
    cyl_array = np.zeros((0, 5))
    for line_center in line_centers:
        plane_slice = points[
            np.linalg.norm(abs(axis * (points - line_center)), axis=1) < (slice_thickness / 2)
        ]
        if plane_slice.shape[0] > 0:
            cyl_center, _, cyl_radius, _, cyl_cci = fit_vertical_cylinder_3D(plane_slice, .03)
            cyl_array = np.vstack((cyl_array, np.array([*cyl_center,cyl_radius,cyl_cci])))

    return cyl_array
