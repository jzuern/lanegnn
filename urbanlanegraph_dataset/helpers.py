import numpy as np


# Coordinate transformations for Kabsch-Umeyama algorithm
coordinates_dict = {
    'Pittsburgh': {
        'points_shapefile':
            np.array([
                [909.27, -119.73, 0],
                [4867.8, 2476.4, 0],
                [5667.6, -568.1, 0],
            ]),
        'points_image': np.array([
            [5763, 30661, 0],
            [32138, 13330, 0],
            [37450, 33660, 0],
        ])
    },
    'Miami': {
        'points_shapefile':
            np.array([
                [2074.90, 1764.30, 0],
                [5994.12, -570.43, 0],
                [-3778.6, -261.6, 0],
            ]),
        'points_image': np.array([
            [65969, 10647, 0],
            [92068, 26208, 0],
            [26924, 24162, 0],
        ]),
    },
    'Detroit': {
        'points_shapefile':
            np.array([
                [10715.6, 3905.1, 0],
                [10429.7, 5141.65, 0],
                [11752.9, 5233.52, 0],
            ]),
        'points_image': np.array([
            [31901., 25657., 0],
            [29997., 17421., 0],
            [19405. * 2, 8405. * 2, 0],
        ])
    },
    'PaloAlto': {
        'points_shapefile':
            np.array([
                [154.25, -1849.03, 0],
                [972.4, 1468, 0],
                [-3057.08, 3133.05, 0],
            ]),
        'points_image': np.array([
            [17430 * 2, 23024 * 2, 0],
            [20170 * 2, 11967 * 2, 0],
            [6735 * 2, 6423 * 2, 0],
        ])
    },
    'Washington': {
        'points_shapefile':
            np.array([
                [3415.5, 24.32, 0],
                [-1579.6, 1635.7, 0],
                [1704, 8484.4, 0],
            ]),
        'points_image': np.array([
            [40799,    69508, 0],
            [3754 * 2, 29380 * 2, 0],
            [14692 * 2, 6556 * 2, 0],
        ])
    },
    'Austin': {
        'points_shapefile':
            np.array([
                [-734.3, 2558.3, 0],
                [-753.7, -3418.3, 0],
                [2302.6, -1396.2, 0]
            ]),
        'points_image': np.array([
            [7038 * 2, 8747 * 2, 0],
            [6962 * 2., 28660 * 2, 0],
            [17155.5 * 2, 21925 * 2, 0]
        ])
    },
}


def kabsch_umeyama(A, B):

    '''
    Calculate the optimal rigid transformation matrix between 2 sets of N x 3 corresponding points using Kabsch Umeyama algorithm.
    '''

    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    # crop last dimension
    R = R[0:2, 0:2]
    t = t[0:2]

    return R, c, t


def get_transform_params(city_name):
    """

    Args:
        city_name:

    Returns:
        The transformation parameters for the city.
    """

    if city_name not in coordinates_dict.keys():
        raise NotImplementedError("Can't find satellite alignment for city {}. Please check the capitalization".format(city_name))

    points_shapefile = coordinates_dict[city_name]["points_shapefile"]
    points_image = coordinates_dict[city_name]["points_image"]

    # Perform the coordinate transformation
    R, c, t = kabsch_umeyama(points_image, points_shapefile)

    """
    Using these parameters, we can transform the coordinates of the argoverse2 annotations to the coordinates of the satellite image:
    
    >> coordinates_av2 = np.array([x, y])
    >> coordinates_image = t + c * R @ coordinates_av2
    
    >> coordinates_image   # np.array([u, v])
    """


    return [R, c, t]