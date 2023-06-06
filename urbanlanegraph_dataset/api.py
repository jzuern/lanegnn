import numpy as np
import glob
import os
import pickle
import cv2


class UrbanLaneGraphMetadata:
    """
    Metadata class that provides city, tile, offset info as well as GT graphs and images.
    """
    def __init__(self, dataset_path):
        """
        Contains val and test tile info but retrieves train-split information from the dataset location.
        :param dataset_path: path to main dataset folder that contains tiles, successor data etc.
        """
        # Test and Val tiles for each city

        self.dataset_path = dataset_path
        self.splits = ["test", "val", "train"]
        self.cities = ["austin", "detroit", "miami", "paloalto", "pittsburgh", "washington"]

        self.tile_dict = {
            "test": {
                "austin": [41, 72],
                "detroit": [135, 204],
                "miami": [143, 94],
                "paloalto": [24, 49],
                "pittsburgh": [19, 67],
                "washington": [48],
            },
            "val": {
                "austin": [83, 40],
                "detroit": [136, 165],
                "miami": [185, 194],
                "paloalto": [62, 43],
                "pittsburgh": [5, 36],
                "washington": [46, 55],
            },
            "train": {
                "austin": list(),
                "detroit": list(),
                "miami": list(),
                "paloalto": list(),
                "pittsburgh": list(),
                "washington": list(),
            }
        }

        # Construct dictionary of all tile offsets
        self.tile_offset_dict = dict()
        for split in self.splits:
            self.tile_offset_dict[split] = dict()
            for city in self.cities:
                self.tile_offset_dict[split][city] = dict()

        for split in self.splits:
            for city in self.cities:
                tile_list = glob.glob(os.path.join(dataset_path, city, 'tiles', split, '*.gpickle'))
                for tile_path in tile_list:
                    tile_name = tile_path.split('/')[-1].split('.')[0]
                    tile_no = tile_name.split('_')[1]
                    tile_offset1 = int(tile_name.split('_')[-2])
                    tile_offset2 = int(tile_name.split('_')[-1])
                    self.tile_offset_dict[split][city][tile_no] = (tile_offset1, tile_offset2)

                    if split == "train":
                        self.tile_dict[split][city].append(tile_no)

        print("Initialized metadata for the UrbanLaneGraph dataset from ", dataset_path)

    def get_cities(self):
        return self.cities

    def get_splits(self):
        return self.splits

    def get_tiles(self, split, city):
        return self.tile_dict[split][city]

    def get_tile_offset(self, split, city, tile_no=None):
        """
        Provides aerial image offsets to retrievve a certain tile.
        :param split: dataset split
        :param city: city string, lowercase (see class init)
        :param tile_no: number of the tile
        :return: either dictionary mapping from tile-no -> offsets or a tuple for a single tile
        """
        if tile_no is None:
            return self.tile_offset_dict[split][city]

        elif tile_no is not None:
            if isinstance(tile_no, int) or isinstance(tile_no, str):
                return self.tile_offset_dict[split][city][str(tile_no)]

            if isinstance(tile_no, list):
                offset_dict = dict()
                for t in tile_no:
                    offset_dict[int(t)] = self.tile_offset_dict[split][city][str(t)]
                return offset_dict

    def get_tile_graph(self, split, city, tile_no):
        """
        Provides GT graph as nx.DiGraph of a certain tile.
        :param split: dataset split
        :param city: city string, lowercase (see class init)
        :param tile_no: number of tile
        :return: nx.DiGraph GT graph
        """
        gt_graph_dir = os.path.join(self.dataset_path, city, 'tiles', split)
        fname = city + '_' + str(tile_no) + '_' + "_".join([str(offset) for offset in self.get_tile_offset(split, city, tile_no)]) + '.gpickle'
        with open(os.path.join(gt_graph_dir, fname), 'rb') as f:
            gt_graph = pickle.load(f)
        return gt_graph

    def get_tile_image(self, split, city, tile_no, path=False):
        """
        Loads aerial image of a tile.
        :param split: dataset split
        :param city: city string, lowercase (see class init)
        :param tile_no: number of tile
        :return either image matrix oder image + path to image
        """
        image_dir = os.path.join(self.dataset_path, city, 'tiles', split)
        fname = city + '_' + str(tile_no) + '_' + "_".join([str(offset) for offset in self.get_tile_offset(split, city, tile_no)]) + '.png'
        image = cv2.cvtColor(cv2.imread(os.path.join(image_dir, fname)), cv2.COLOR_BGR2RGB)
        if path:
            return image, os.path.join(image_dir, fname)
        return image


# Coordinate transformations for Kabsch-Umeyama algorithm
coordinates_dict = {
    'pittsburgh': {
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
    'miami': {
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
    'detroit': {
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
    'paloalto': {
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
    'washington': {
        'points_shapefile':
            np.array([
                [3415.5, 24.32, 0],
                [-1579.6, 1635.7, 0],
                [1704, 8484.4, 0],
            ]),
        'points_image': np.array([
            [40799, 69508, 0],
            [3754 * 2, 29380 * 2, 0],
            [14692 * 2, 6556 * 2, 0],
        ])
    },
    'austin': {
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
        raise NotImplementedError(
            "Can't find satellite alignment for city {}. Please check the capitalization".format(city_name))

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
