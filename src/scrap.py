import numpy as np
from skimage.measure import LineModelND, CircleModel, ransac
from sklearn.neighbors import NearestNeighbors


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


def circumferential_completeness_index(fitted_circle_centre, estimated_radius, slice_points):
    """
    Computes the Circumferential Completeness Index (CCI) of a fitted circle.
    Args:
        fitted_circle_centre: x, y coords of the circle centre
        estimated_radius: circle radius
        slice_points: the points the circle was fitted to
    Returns:
        CCI
    """

    sector_angle = 4.5  # degrees
    num_sections = int(np.ceil(360 / sector_angle))
    sectors = np.linspace(-180, 180, num=num_sections, endpoint=False)

    centre_vectors = slice_points[:, :2] - fitted_circle_centre
    norms = np.linalg.norm(centre_vectors, axis=1)

    centre_vectors = centre_vectors / np.atleast_2d(norms).T
    centre_vectors = centre_vectors[
        np.logical_and(norms >= 0.8 * estimated_radius, norms <= 1.2 * estimated_radius)
    ]

    sector_vectors = np.vstack((np.cos(sectors), np.sin(sectors))).T
    CCI = (
        np.sum(
            [
                np.any(
                    np.degrees(
                        np.arccos(
                            np.clip(np.einsum("ij,ij->i", np.atleast_2d(sector_vector), centre_vectors), -1, 1)
                        )
                    )
                    < sector_angle / 2
                )
                for sector_vector in sector_vectors
            ]
        )
        / num_sections
    )

    return CCI


def fit_circle_3D(points, V):
    """
    Fits a circle using Random Sample Consensus (RANSAC) to a set of points in a plane perpendicular to vector V.
    Args:
        points: Set of points to fit a circle to using RANSAC.
        V: Axial vector of the cylinder you're fitting.
    Returns:
        cyl_output: numpy array of the format [[x, y, z, x_norm, y_norm, z_norm, radius, CCI, 0, 0, 0, 0, 0, 0]]
    """

    CCI = 0
    r = 0
    P = points[:, :3]
    P_mean = np.mean(P, axis=0)
    P_centered = P - P_mean
    normal = V / np.linalg.norm(V)
    if normal[2] < 0:  # if normal vector is pointing down, flip it around the other way.
        normal = normal * -1

    # Project points to coords X-Y in 2D plane
    P_xy = rodrigues_rot(P_centered, normal, [0, 0, 1])

    # Fit circle in new 2D coords with RANSAC
    if P_xy.shape[0] >= 20:

        model_robust, inliers = ransac(
            P_xy[:, :2],
            CircleModel,
            min_samples=int(P_xy.shape[0] * 0.3),
            residual_threshold=0.05,
            max_trials=10000,
        )
        xc, yc = model_robust.params[0:2]
        r = model_robust.params[2]
        CCI = circumferential_completeness_index([xc, yc], r, P_xy[:, :2])

    if CCI < 0.3:
        r = 0
        xc, yc = np.mean(P_xy[:, :2], axis=0)
        CCI = 0

    # Transform circle center back to 3D coords
    cyl_centre = rodrigues_rot(np.array([[xc, yc, 0]]), [0, 0, 1], normal) + P_mean
    cyl_output = np.array(
        [
            [
                cyl_centre[0, 0],
                cyl_centre[0, 1],
                cyl_centre[0, 2],
                normal[0],
                normal[1],
                normal[2],
                r,
                CCI,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ]
    )
    return cyl_output


def fit_cylinder(skeleton_points, point_cloud, num_neighbours):
    """
    Fits a 3D line to the skeleton points cluster provided.
    Uses this line as the major axis/axial vector of the cylinder to be fitted.
    Fits a series of circles perpendicular to this axis to the point cloud of this particular stem segment.
    Args:
        skeleton_points: A single cluster of skeleton points which should represent a segment of a tree/branch.
        point_cloud: The cluster of points belonging to the segment of the branch.
        num_neighbours: The number of skeleton points to use for fitting each circle in the segment. lower numbers
                        have fewer points to fit a circle to, but higher numbers are negatively affected by curved
                        branches. Recommend leaving this as it is.
    Returns:
        cyl_array: a numpy array based representation of the fitted circles/cylinders.
    """

    point_cloud = point_cloud[:, :3]
    skeleton_points = skeleton_points[:, :3]
    cyl_array = np.zeros((0, 14))
    line_centre = np.mean(skeleton_points[:, :3], axis=0)
    _, _, vh = np.linalg.svd(line_centre - skeleton_points)
    line_v_hat = vh[0] / np.linalg.norm(vh[0])

    if skeleton_points.shape[0] <= num_neighbours:
        group = skeleton_points
        line_centre = np.mean([np.min(group[:, :3], axis=0), np.max(group[:, :3], axis=0)], axis=0)
        length = np.linalg.norm(np.max(group, axis=0) - np.min(group, axis=0))
        plane_slice = point_cloud[
            np.linalg.norm(abs(line_v_hat * (point_cloud - line_centre)), axis=1) < (length / 2)
        ]  # calculate distances to plane at centre of line.
        if plane_slice.shape[0] > 0:
            cylinder = fit_circle_3D(plane_slice, line_v_hat)
            cyl_array = np.vstack((cyl_array, cylinder))
    else:
        while skeleton_points.shape[0] > num_neighbours:
            nn = NearestNeighbors()
            nn.fit(skeleton_points)
            starting_point = np.atleast_2d(skeleton_points[np.argmin(skeleton_points[:, 2])])
            group = skeleton_points[nn.kneighbors(starting_point, n_neighbors=num_neighbours)[1][0]]
            line_centre = np.mean(group[:, :3], axis=0)
            length = np.linalg.norm(np.max(group, axis=0) - np.min(group, axis=0))
            plane_slice = point_cloud[
                np.linalg.norm(abs(line_v_hat * (point_cloud - line_centre)), axis=1) < (length / 2)
            ]  # calculate distances to plane at centre of line.
            if plane_slice.shape[0] > 0:
                cylinder = fit_circle_3D(plane_slice, line_v_hat)
                cyl_array = np.vstack((cyl_array, cylinder))
            skeleton_points = np.delete(skeleton_points, np.argmin(skeleton_points[:, 2]), axis=0)
    return cyl_array