import networkx as nx
import numpy as np
import os
import cv2
import json
from glob import glob
import scipy
import pickle
import argparse

from lanegnn.utils.params import ParamLib


def extract_pose(line, sat_image, min_line_length=20, offset_x=0, offset_y=0, node_mask=None):
    """
    Extracts a pose from a line
    :param line: List of coordinates to extract pose from
    :param min_line_length: Minimum length of line to extract pose from
    :param offset_x: img offset in x direction
    :param offset_y: img offset in y direction

    """
    poses_to_save = list()
    # Change coordinate system (x, y) -> (y, x) and apply crop offset
    start_node_old_coords = line[0]
    end_node_old_coords = line[1]

    # start_node_new_coords = [int(start_node_old_coords[0]) + offset_x,
    #                          int(start_node_old_coords[1]) + offset_y]
    # end_node_new_coords = [int(end_node_old_coords[0]) + offset_x,
    #                        int(end_node_old_coords[1]) + offset_y]

    start_node_new_coords = [int(start_node_old_coords[0]),
                             int(start_node_old_coords[1])]
    end_node_new_coords = [int(end_node_old_coords[0]),
                           int(end_node_old_coords[1])]

    # Infer line length
    line_length = scipy.spatial.distance.euclidean(start_node_new_coords, end_node_new_coords)

    # Get yaw angle of line
    yaw = np.arctan2(end_node_new_coords[0] - start_node_new_coords[0],
                     end_node_new_coords[1] - start_node_new_coords[1])

    if line_length > min_line_length:
        if 0 < start_node_new_coords[0] < sat_image.shape[1] and 0 < start_node_new_coords[1] < sat_image.shape[0]:
            if node_mask is not None and node_mask[end_node_new_coords[0], end_node_new_coords[1]] != 255:
                # Mask is defined and node is outside of mask -> Do not add the node
                pass
            else:
                poses_to_save.append({"x": start_node_new_coords[1],
                                      "y": start_node_new_coords[0],
                                      "yaw": yaw,
                                      "len": line_length,
                                      "type": "start",
                                      "offset_x": offset_x,
                                      "offset_y": offset_y
                                      })
        if line_length > 200:
            if 0 < end_node_new_coords[0] < sat_image.shape[1] and 0 < end_node_new_coords[1] < sat_image.shape[0]:
                if node_mask is not None and node_mask[end_node_new_coords[0], end_node_new_coords[1]] != 255:
                    # Mask is defined and node is outside of mask -> Do not add the node
                    pass
                else:
                    poses_to_save.append({"x": end_node_new_coords[1],
                                          "y": end_node_new_coords[0],
                                          "yaw": yaw,
                                          "len": line_length,
                                          "type": "end",
                                          "offset_x": offset_x,
                                          "offset_y": offset_y
                                          })
    return poses_to_save


def plot_extracted_poses(sat_image, init_poses, node_mask=None):
    """
    Plots extracted poses on satellite image
    :param params: parameter config
    :param init_poses: List of extracted poses
    :param node_mask: Masking based on GT graph
    """

    for pose in init_poses:
        # plot init pose (x, y, yaw
        y = pose["x"] #- pose["offset_y"]
        x = pose["y"] #- pose["offset_x"]
        yaw = pose["yaw"] - np.pi / 2

        xx = int(x + 25 * np.cos(yaw))
        yy = int(y - 25 * np.sin(yaw))

        cv2.circle(sat_image, (y, x), 5, (0, 0, 255), -1)
        cv2.arrowedLine(sat_image, (y, x), (yy, xx), (0, 255, 255), 2)

    return sat_image


def generate_gt_mask(params, mask_offset, sat_image):
    """
    Generates a mask based on the GT graph with some padding offset around the nodes
    :param params: parameter config
    :param mask_offset: Offset around the nodes
    :param sat_image: Satellite image
    """
    gt_graph_file = glob(os.path.join(params.driving.data_dir,
                                      params.driving.city,
                                      "*_{}_*_g_gt.gpickle".format(params.driving.tile_no)))[0]
    # Load GT graph
    with open(gt_graph_file, 'rb') as f:
        gt_graph = pickle.load(f)

    node_mask = np.zeros((sat_image.shape[0], sat_image.shape[1]))
    for n in gt_graph.nodes:
        try:
            print(gt_graph.nodes[n]["pos"][1], gt_graph.nodes[n]["pos"][1])
            node_mask[int(gt_graph.nodes[n]["pos"][1]), int(gt_graph.nodes[n]["pos"][1])] = 255

            # Mask out a 50px radius around the node
            node_mask[int(gt_graph.nodes[n]["pos"][1]) - mask_offset:int(gt_graph.nodes[n]["pos"][1]) + mask_offset,
            int(gt_graph.nodes[n]["pos"][0]) - mask_offset:int(gt_graph.nodes[n]["pos"][0]) + mask_offset] = 255
        except:
            print("maybe outside bounds?")
    print(np.sum(node_mask))
    return node_mask


def extract_init_poses(params, sat_image, node_mask=None):
    """
    Extracts init poses based on LaneExtraction outputs.
    """
    city = params.driving.city
    tile_no = params.driving.tile_no
    dir_path = params.driving.data_dir

    ways_params = list()
    ways_params.extend(glob(os.path.join(dir_path, city) + '/*_{}_*-ways.json'.format(tile_no)))

    init_poses = list()
    for ways_param in ways_params:
        ways_param_sep = ways_param.split('/')[-1].split('-')[0].split('_')

        img_offset_x = int(ways_param_sep[-2])
        img_offset_y = int(ways_param_sep[-1])

        # open ways.json
        with open(ways_param, 'r') as f:
            ways = json.load(f)

        for multiline in ways:
            if len(multiline) == 2:
                # multiline is just a line
                add_poses = extract_pose(multiline,
                                         min_line_length=40,
                                         offset_x=img_offset_x,
                                         offset_y=img_offset_y,
                                         sat_image=sat_image,
                                         node_mask=node_mask)
            elif len(multiline) > 2:
                print(multiline)
                # multiline is a list of lines
                # iterate through each line segment
                for i in range(0, len(multiline) - 1):
                    line = [multiline[i], multiline[i + 1]]
                    add_poses = extract_pose(line,
                                             min_line_length=20,
                                             offset_x=img_offset_x,
                                             offset_y=img_offset_y,
                                             sat_image=sat_image,
                                             node_mask=node_mask)
            init_poses.extend(add_poses)
    return init_poses, img_offset_x, img_offset_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRIVE")
    # General parameters (namespace: main)
    parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)

    # Namespace-specific arguments (namespace: driving)
    parser.add_argument("--city", type=str)
    parser.add_argument("--tile_no", type=int)

    opt = parser.parse_args()

    params = ParamLib(opt.config)
    params.main.overwrite(opt)
    params.driving.overwrite(opt)

    sat_image_path = glob(os.path.join(params.driving.data_dir,
                                       params.driving.city,
                                       "*_{}_*-raw.png".format(params.driving.tile_no)))[0]
    sat_image = cv2.imread(sat_image_path)

    # node_mask = generate_gt_mask(params, mask_offset=100, sat_image=sat_image)
    extracted_init_poses, img_offset_x, img_offset_y = extract_init_poses(params, sat_image, node_mask=None)
    plot_image = plot_extracted_poses(sat_image, extracted_init_poses)

    cv2.imwrite(os.path.join(params.driving.data_dir,
                             params.driving.city,
                             "{}_{}_{}_{}_debug.png".format(params.driving.city,
                                                            params.driving.tile_no,
                                                            img_offset_x,
                                                            img_offset_y)), sat_image)

    # Save init poses
    init_poses_path = os.path.join(params.driving.data_dir,
                                   params.driving.city,
                                   "{}_{}_{}_{}_init_poses.json".format(params.driving.city,
                                                                        params.driving.tile_no,
                                                                        img_offset_x,
                                                                        img_offset_y))
    with open(init_poses_path, 'w') as f:
        json.dump(extracted_init_poses, f)

    print(len(extracted_init_poses))
