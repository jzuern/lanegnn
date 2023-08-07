import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist


def get_delaunay_triangulation(points):
    """
    DEPRECATED
    Used for triangular edge proposal method instead of get_random_edges().
    """
    tri = Delaunay(points)

    # convert simplices to index pairs
    simplices = []
    for s in tri.simplices:
        simplices.append([s[0], s[1]])
        simplices.append([s[1], s[0]])
        simplices.append([s[0], s[2]])
        simplices.append([s[2], s[0]])
        simplices.append([s[1], s[2]])
        simplices.append([s[2], s[1]])

    return np.array(simplices)



def get_random_edges(point_coords, min_point_dist=10, max_point_dist=50):
    """
    Based on node coordinates and a minimum/maximum edge length, return edges
    """
    # compute pairwise distances
    dist_mat = cdist(point_coords, point_coords)
    # print(dist_mat.shape)

    # find all pairs of points with a distance less than max_point_dist
    filter_matrix = (dist_mat < max_point_dist) * (dist_mat > min_point_dist)

    valid_edges = np.where(filter_matrix)
    valid_edges = np.array(list(zip(valid_edges[0], valid_edges[1])))
    valid_edges = np.unique(valid_edges, axis=0)

    return valid_edges


def primes_from_2_to(n):
    """Prime number from 2 to n.

    From `StackOverflow <https://stackoverflow.com/questions/2068372>`_.

    :param int n: sup bound with ``n >= 6``.
    :return: primes in 2 <= p < n.
    :rtype: list
    """
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool)
    for i in range(1, int(n ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3::2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


def van_der_corput(n_sample, base=2):
    """Van der Corput sequence.

    :param int n_sample: number of element of the sequence.
    :param int base: base of the sequence.
    :return: sequence of Van der Corput.
    :rtype: list (n_samples,)
    """
    sequence = []
    for i in range(n_sample):
        n_th_number, denom = 0., 1.
        while i > 0:
            i, remainder = divmod(i, base)
            denom *= base
            n_th_number += remainder / denom
        sequence.append(n_th_number)

    return sequence


def halton(dim, n_sample):
    """Halton sequence.

    :param int dim: dimension
    :param int n_sample: number of samples.
    :return: sequence of Halton.
    :rtype: array_like (n_samples, n_features)
    """
    big_number = 10
    while 'Not enought primes':
        base = primes_from_2_to(big_number)[:dim]
        if len(base) == dim:
            break
        big_number += 1000

    # Generate a sample using a Van der Corput sequence per dimension.
    sample = [van_der_corput(n_sample + 1, dim) for dim in base]
    sample = np.stack(sample, axis=-1)[1:]

    return sample
