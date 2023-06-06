import numpy as np
import random
import torch
import cv2
import time
import json
import copy
import argparse
import torch_geometric
import networkx as nx
import pynvml
import yaml
import os
import glob
import pickle
import ray
from ray.util.multiprocessing import Pool
from PIL import Image
Image.MAX_IMAGE_PIXELS = pow(2, 35).__int__()

from lanegnn.utils.params import ParamLib
from lanegnn.utils.graph import unbatch_edge_index, laplacian_smoothing, transform_graph_to_pos_indexing
from lanegnn.utils.misc import chunks, mean_angle_abs_diff
from lanegnn.utils.continuous import is_in_roi, transform_keypoint_to_world
from lanegnn.utils.inference import generate_data_live, render_current_graph, get_init_poses, init_model, \
    create_drive_experiment_folders, load_sat_img

from lanegnn.inference.traverse_endpoint import preprocess_predictions, predict_lanegraph
from lanegnn.inference.step import step_forward
from lanegnn.inference.aggregate import aggregate, fuse
from lanegnn.inference.postprocess_drive_multiple import drive_agg_synth


# numpy random seed
np.random.seed(0)
torch.manual_seed(0)


def drive_agg(params, model, context_regressor, ego_regressor, satellite_image,
              G_agg_prev=None, visited_edges=None, ego_positions=None, split_edge_list=None, pose_from_list=None, drive_idx=None, chunk_idx=None):

    params.model.dataparallel = False

    # single mode
    if pose_from_list is None:
        x_start, y_start, angle_start = params.driving.starting_pose
        x_start = eval(x_start) if type(x_start) == str else x_start
        y_start = eval(y_start) if type(y_start) == str else y_start
        angle_start = eval(angle_start) if type(angle_start) == str else angle_start

    # multiple mode
    else:
        x_start, y_start, angle_start = pose_from_list
        x_start = eval(x_start) if type(x_start) == str else x_start
        y_start = eval(y_start) if type(y_start) == str else y_start
        angle_start = eval(angle_start) if type(angle_start) == str else angle_start


    if "LaneExtraction" in params.driving.data_dir or "code_release" in params.driving.data_dir:
        print(satellite_image.shape)

        mrg = 512
        satellite_image = np.pad(satellite_image, ((mrg, mrg), (mrg, mrg), (0, 0)), 'symmetric')

        print(satellite_image.shape)

        x_start += mrg
        y_start += mrg

        roi_area_xxyy = np.array([0,  # horizontal_min
                                  satellite_image.shape[0],  # horizontal_max
                                  0,  # vertical_min
                                  satellite_image.shape[1]])  # vertical_max

    else:
        roi_area_xxyy = np.array([x_start - 1000,   # horizontal_min
                                  x_start + 1000,   # horizontal_max
                                  y_start - 1000,   # vertical_min
                                  y_start + 1000])  # vertical_max
        satellite_image = satellite_image[roi_area_xxyy[2]:roi_area_xxyy[3], roi_area_xxyy[0]:roi_area_xxyy[1]]

    satellite_image = np.ascontiguousarray(satellite_image)

    print(satellite_image.shape)
    satellite_image_global = copy.deepcopy(satellite_image)
    print("Satellite image loaded")

    graphs_pred = []

    # Initial ego position and orientation
    initial_pose = np.array([x_start,     # x horizontal
                             y_start,     # y vertical
                             angle_start])  # yaw
    print(initial_pose)
    branch_id = 0
    step_counter = 0

    G_agg = nx.DiGraph()
    if G_agg_prev is None:
        G_agg_prev = nx.DiGraph()
    if ego_positions is None:
        ego_positions = list()
    else:
        if type(ego_positions) == np.ndarray:
            ego_positions = ego_positions.tolist()
    #if visited_edges is None:
    visited_edges = list()

    #if split_edge_list is None:
    split_edge_list = list()

    ego_x_y_yaw = initial_pose

    branch_alive = True
    num_fut_branches = 1
    branch_age = 0
    remove_smp = True

    # split_edge_queue = PriorityQueue(maxsize=10000)

    while num_fut_branches or branch_alive:

        if step_counter > params.driving.max_steps:
            print("     Max steps reached. Exiting loop.")
            break

        if branch_id > params.driving.max_branch_id:
            print("     Max branch ID reached. Exiting loop.")
            break

        visited_nodes = list()
        for edge in visited_edges:
            visited_nodes.append(edge[0])
            visited_nodes.append(edge[1])

        start_time = time.time()


        # Normal check
        if branch_alive:
            branch_age += 1
            # Proceed with exploring current branch
            # print("     Branch {} is alive.".format(branch_id))
            pass
        else:
            branch_age = 0
            print(ego_x_y_yaw)
            # Branch ended somehow, start looking for new branches
            print("     Branch is dead. Continuing with next promising edge.")

            # split_edge_queue = PriorityQueue(maxsize=10000)
            # Go through all nodes contained in G_agg and select nodes with a branching factor of 2 and higher
            # These represent all (directional) split points in the graph
            agg_out_degree = dict(G_agg.out_degree(list(G_agg.nodes())))

            for n in G_agg.nodes():
                if agg_out_degree[n] >= 2 and n in visited_nodes:
                    # Gather all untraversed edges leaving this node
                    unvisited_edges_starting_at_n = [e for e in G_agg.out_edges(n) if e not in visited_edges]
                    if len(unvisited_edges_starting_at_n) > 0:

                        # Add all untraversed nodes to priority queue that is min-sorted -> inv. DFS weight necessary
                        node_tree = list(nx.dfs_tree(G_agg, source=n, depth_limit=10))
                        inv_tree_weight = -np.sum([G_agg.nodes[k]['weight'] for k in node_tree])

                        # Add all edges that leave node n to queue with node + edge succ DFS tree weight
                        for unvisited_edge in unvisited_edges_starting_at_n:
                            edge_tree = list(nx.dfs_tree(G_agg, source=unvisited_edge[1], depth_limit=10))
                            inv_edge_tree_weight = -np.sum([G_agg.nodes[k]['weight'] for k in edge_tree])

                            inv_weight = inv_tree_weight + inv_edge_tree_weight
                            # if True:
                            if inv_weight < -100:
                                # checking whether edge is already in list
                                is_edge_in_list = len([elem for elem in split_edge_list if elem[1] == unvisited_edge]) > 0
                                if not is_edge_in_list:
                                    # ego_positions_arr = np.array(ego_positions)
                                    # pot_ego_x_y_yaw = np.array([G_agg.nodes[unvisited_edge[1]]['pos'][0],
                                    #                             G_agg.nodes[unvisited_edge[1]]['pos'][1],
                                    #                             np.arctan2(G_agg.nodes[unvisited_edge[1]]['pos'][1] -
                                    #                                        G_agg.nodes[unvisited_edge[0]]['pos'][1],
                                    #                                        G_agg.nodes[unvisited_edge[1]]['pos'][0] -
                                    #                                        G_agg.nodes[unvisited_edge[0]]['pos'][0]),
                                    #                             ])
                                    # # Get pairwise distance between nodes in G_agg and G_new
                                    # ego_pos_arr = ego_positions_arr[:, 0:2].reshape(-1, 2)
                                    # node_pos_ego = np.array([pot_ego_x_y_yaw[0], pot_ego_x_y_yaw[1]]).reshape(-1, 2)
                                    # node_distances = cdist(ego_pos_arr, node_pos_ego, metric='euclidean')
                                    #
                                    # # Get pairwise angle difference between nodes in G_agg and G_new
                                    # ego_yaw_arr = ego_positions_arr[:, 2].reshape(-1, 1)
                                    # node_mean_ang_ego = np.array([pot_ego_x_y_yaw[2]]).reshape(-1, 1)
                                    # node_mean_ang_distances = cdist(ego_yaw_arr, node_mean_ang_ego, lambda u, v: mean_angle_abs_diff(u, v))
                                    #
                                    # position_criterium = node_distances < 10
                                    # angle_criterium = node_mean_ang_distances < 0.3
                                    # criterium = position_criterium & angle_criterium
                                    #
                                    # # Check criterium sum
                                    # if np.sum(criterium) > 0:
                                    #     print("Pose has already been visited")
                                    #     continue

                                    split_edge_list.append((inv_weight, unvisited_edge))
                                    print("     Added unvisited_edge {} with weight {} to split_edge_list.".format(
                                        unvisited_edge, inv_weight))
                                    # split_edge_queue.put((inv_weight, e))

            # If there exist future unexplored edges, obtain the next one with highest DFS weight => branch alive again
            # num_fut_branches = split_edge_queue.qsize()
            num_fut_branches = len(split_edge_list)
            if num_fut_branches > 0:
                while True:
                    # inv_weight, successor_split_edge = split_edge_queue.get()

                    # select random edge
                    #inv_weight, successor_split_edge = random.choice(split_edge_list)

                    # select edge with smallest weight
                    if len(split_edge_list) > 0:
                        inv_weight, successor_split_edge = min(split_edge_list, key=lambda x: x[0])
                    else:
                        break

                    split_edge_list.remove((inv_weight, successor_split_edge))

                    try:
                        visited_edges.append(successor_split_edge)
                        # Get node positions of successor_split_edge
                        ego_x_y_yaw = np.array([G_agg.nodes[successor_split_edge[1]]['pos'][0],
                                                G_agg.nodes[successor_split_edge[1]]['pos'][1],
                                                np.arctan2(G_agg.nodes[successor_split_edge[1]]['pos'][1] - G_agg.nodes[successor_split_edge[0]]['pos'][1],
                                                           G_agg.nodes[successor_split_edge[1]]['pos'][0] - G_agg.nodes[successor_split_edge[0]]['pos'][0]),
                                               ])
                        ego_positions.append(ego_x_y_yaw)
                        break

                    except:
                        print("     Warning: Successor split edge not found in graph, skipping.")
                        continue

                print("     Found new ego pose: {}".format(ego_x_y_yaw))
                branch_id += 1
                branch_alive = True
                continue
            else:
                # branch_alive = False
                break

        # Check if next ego_pose will be out of bounds wrt to defined ROI
        if ego_x_y_yaw is not None:
            if not is_in_roi(ego_x_y_yaw, roi_area_xxyy, margin=600):
                print("     Pose out of ROI. Killing branch.")
                branch_alive = False
                continue
        else:
            print("     Pose is None. Killing branch.")
            branch_alive = False
            continue

        if branch_age > params.driving.max_branch_age:
            print("     Branch age exceeded. Killing branch.")
            branch_alive = False
            continue

        print("Step: {}, Branch: {}, Branch age: {}, edge queue length: {} | Ego position: [{} {}], Heading: {:.2f} deg".format(step_counter,
                  branch_id, branch_age, num_fut_branches, int(ego_x_y_yaw[0]), int(ego_x_y_yaw[1]), ego_x_y_yaw[2]/np.pi*180))


        # Generate data object for the current frame
        data, ego_regr_smooth, context_regr_smooth = \
            generate_data_live(ego_regressor, context_regressor, params, satellite_image, ego_x_y_yaw,
                               roi_area_xxyy, step_counter)

        # Check if we have any nodes/edges in data object
        if data.edge_index.shape[0] == 0:
            print("     No edges found in current crop. Killing branch.")
            branch_alive = False
            continue

        # Predict on sample and get UCS traversal from predictions
        _, node_scores_pred, endpoint_scores_pred, _, img_rgb, node_pos, fut_nodes, _, startpoint_idx = preprocess_predictions(params, model, data, gt_available=False)
        pred_graph = predict_lanegraph(fut_nodes, startpoint_idx, node_scores_pred, endpoint_scores_pred, node_pos, debug=False)

        # Update graph node positions from local ego-crop to global coordinate frame
        for n in pred_graph.nodes():
            pred_graph.nodes[n]['pos'] = transform_keypoint_to_world(pred_graph.nodes[n]['pos'], ego_x_y_yaw)

        # Rephrase local node indexing to int-pixel-based global coordinates based on just obtained global pos coords
        pred_graph = transform_graph_to_pos_indexing(pred_graph)

        # smooth graph
        if params.driving.smooth_pred_graph:
            if len(pred_graph.nodes()) > 0 and len(pred_graph.edges()) > 0:
                pred_graph = laplacian_smoothing(copy.deepcopy(pred_graph), gamma=0.25, iterations=2)

        G_agg_before = copy.deepcopy(G_agg)

        G_agg, merging_map = aggregate(G_agg,
                                       pred_graph,
                                       visited_edges,
                                       threshold_px=params.driving.threshold_px,
                                       threshold_rad=params.driving.threshold_rad,
                                       closest_lat_thresh=params.driving.closest_lat_dist_succ,
                                       w_decay=params.driving.w_decay,
                                       remove=params.driving.remove_smp)

        # Check if graph has changed
        # Update ego position and orientation based on maximum-weight path
        traversed_edge, ego_x_y_yaw, step_vector, branch_alive, visited_edges = step_forward(ego_x_y_yaw, G_agg, visited_edges)

        # visualization
        # For plotting purposes only
        if branch_alive:
            start = np.array(ego_x_y_yaw[:2])
            end = np.array([ego_x_y_yaw[0] + 30 * np.cos(ego_x_y_yaw[2]), ego_x_y_yaw[0] + 30 * np.sin(ego_x_y_yaw[2])])
            plot_step_vector = end - start
            graph_canvas = render_current_graph(params, G_agg, G_agg_before, pred_graph, visited_edges,
                                                    satellite_image_global, ego_x_y_yaw, plot_step_vector, roi_area_xxyy,
                                                    split_edge_list, step_counter)

        step_counter += 1

        if branch_alive:
            graphs_pred.append(pred_graph)
            ego_positions.append(ego_x_y_yaw)
            #if traversed_edge[1] in visited_nodes:
            #    print("     Warning: Node already visited, evaluate frontier next.")
            #    branch_alive = False
            #    continue

        print("     Step took {:.2f} seconds.".format(time.time() - start_time))

    print("Finished drive.")

    G_agg, merging_map = aggregate(G_agg_prev,
                                   G_agg,
                                   visited_edges,
                                   threshold_px=params.driving.threshold_px,
                                   threshold_rad=params.driving.threshold_rad,
                                   closest_lat_thresh=params.driving.closest_lat_dist_agg,
                                   w_decay=params.driving.w_decay,
                                   remove=params.driving.remove_smp)

    # Store graph up to here
    config_str = "{}-{}-chunk{}-drive{}".format(params.driving.city, params.driving.tile_no, chunk_idx, drive_idx)
    tmp_data_dict = {
        "G_agg": G_agg,
        "visited_edges": visited_edges,
        "ego_positions": np.array(ego_positions),
        "graphs_pred": graphs_pred,
    }

    with open(os.path.join(params.driving.experiment_dir, "single_graphs/graph_{}.pkl".format(config_str)), "wb") as f:
        pickle.dump(tmp_data_dict, f)

    # For plotting purposes only
    try:
        start = np.array(ego_x_y_yaw[:2])
        end = np.array([ego_x_y_yaw[0]+30*np.cos(ego_x_y_yaw[2]), ego_x_y_yaw[0]+30*np.sin(ego_x_y_yaw[2])])
        plot_step_vector = end - start

        graph_canvas = render_current_graph(params, G_agg, G_agg, G_agg, visited_edges,
                                            satellite_image_global, ego_x_y_yaw, plot_step_vector, roi_area_xxyy,
                                            split_edge_list, step_counter)
    except:
        print("     Warning: Could not plot graph.")

    return G_agg, graphs_pred, visited_edges, np.array(ego_positions), split_edge_list, roi_area_xxyy, satellite_image


@ray.remote(num_gpus=0.3)
def process_chunk(data):
    poses, gpu_idx, params, chunk_idx = data
    model, context_regressor, ego_regressor, sat_img = init_model(params)

    print("USED GPU: {}".format(ray.get_gpu_ids()))
    graphs_pred = None
    roi_area_xxyy = None
    visited_edges = None
    ego_positions = None
    evaluated_poses = list()

    # wait 20 seconds
    # time.sleep(20)
    pose = None

    G_agg = nx.DiGraph()
    for drive_idx, pose in enumerate(poses):
        evaluated_poses.append(pose)
        if pose["type"] == "end" and pose["len"] > 80:
            # Go backward using yaw angle
            pose["x"] = pose["x"] - 80 * np.cos(pose["yaw"])
            pose["y"] = pose["y"] - 80 * np.sin(pose["yaw"])
        pose = [pose["x"], pose["y"], pose["yaw"]]
        print(pose)
        if drive_idx == 0:
            G_agg, graphs_pred, visited_edges, ego_positions, split_edge_list, roi_area_xxyy, satellite_image = \
                drive_agg(params, model, context_regressor, ego_regressor, sat_img,
                          G_agg_prev=G_agg, visited_edges=None, ego_positions=None,
                          split_edge_list=None, pose_from_list=pose, drive_idx=drive_idx, chunk_idx=chunk_idx)
        else:
            try:
                G_agg, graphs_pred, visited_edges, ego_positions, split_edge_list, roi_area_xxyy, satellite_image = \
                    drive_agg(params, model, context_regressor, ego_regressor, sat_img,
                              G_agg_prev=G_agg, visited_edges=visited_edges, ego_positions=ego_positions,
                              split_edge_list=split_edge_list,
                              pose_from_list=pose, drive_idx=drive_idx, chunk_idx=chunk_idx)
            except:
                print("skipped one init pose because it failed")
                continue

    data_dict = {
        "G_agg": G_agg,
        "visited_edges": visited_edges,
        "graphs_pred": graphs_pred,
        "ego_positions": ego_positions,
        "roi_area_xxyy": roi_area_xxyy,
        "visited_init_poses": evaluated_poses,
    }

    # define configuration string
    config_str = "gpu{}_drive{}_px{}-rad{}-w_d{}".format(chunk_idx, drive_idx, params.driving.threshold_px, params.driving.threshold_rad, params.driving.w_decay)

    with open(os.path.join(params.driving.experiment_dir,
                           "full_graph_chunks/data_dict_{}.pkl".format(config_str)), "wb") as f:
        pickle.dump(data_dict, f)

    return data_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DRIVE")
    # General parameters (namespace: main)
    parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)

    # Namespace-specific arguments (namespace: driving)
    parser.add_argument('--city', type=str, help="city name")
    parser.add_argument('--tile_no', type=int, help="tile number")
    parser.add_argument('--threshold_px', type=float, help='')
    parser.add_argument('--threshold_rad', type=float, help="dataset path")
    parser.add_argument('--non_drivable_thresh', type=float, help="non drivable threshold for masking")
    parser.add_argument('--closest_lat_dist_succ', type=float, help="in-drive lateral aggregation pixel threshold")
    parser.add_argument('--closest_lat_dist_agg', type=float, help="among-drive lateral aggregation pixel threshold")
    parser.add_argument('--w_decay', type=bool, help="weight decay")
    parser.add_argument('--max_steps', type=bool, help="maximum number of among-branch steps")
    parser.add_argument('--max_branch_age', type=bool, help="maximum number of in-branch steps")
    parser.add_argument('--smooth_pred_graph', type=bool, help="should graph be laplace-smoothed before aggregating?")
    parser.add_argument('--remove_smp', type=bool, help="remove parallel paths and unvalited splits and merges")
    parser.add_argument('--max_branch_id', type=int, help="drive index")

    opt = parser.parse_args()

    params = ParamLib(opt.config)
    params.main.overwrite(opt)
    params.driving.overwrite(opt)

    if params.driving.mode == "single" or params.driving.mode == "multiple":
        raise NotImplementedError

    elif params.driving.mode == "distributed":

        # Create directory structure and print drive parameters
        created_paths = create_drive_experiment_folders(params)

        poses = get_init_poses(params)
        random.shuffle(poses)

        # Write params to file
        with open(os.path.join(params.driving.experiment_dir, "params.yaml"), "w") as f:
            yaml.dump(params, f)

        # Get all available GPUs
        available_gpus = list()
        all_device_idcs = [i for i in range(torch.cuda.device_count())]
        pynvml.nvmlInit()
        print("Driver Version:", pynvml.nvmlSystemGetDriverVersion())
        for gpu_idx in all_device_idcs:
            used_mem = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)).used
            # Check whether the GPU util is below 10000 MiB
            if used_mem < 10000*10**6:
                available_gpus.append(gpu_idx)

        # export all available gpus as environment variable before initializing ray
        print("Available GPUs: {}".format(available_gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu_idx) for gpu_idx in available_gpus])

        ray.init(num_gpus=params.driving.ray_num_gpus)

        # Split poses into chunks that can be processed by the available GPU, respectively
        chunk_len = int(np.ceil(len(poses) / len(available_gpus)))
        pose_chunks = list(chunks(poses, chunk_len))
        print(len(pose_chunks))

        chunk_idx = 0
        chunk_data = list()
        for pose_chunk, gpu in zip(pose_chunks, available_gpus):
            chunk_data.append((pose_chunk, gpu, params, chunk_idx))
            chunk_idx += 1

        # Please leave here for debugging
        # process_chunk(chunk_data[1])

        if params.driving.agg_single_drives:
            drive_file_path = os.path.join(params.driving.experiment_dir,
                                           "single_graphs/")
            data_dict_key = "graphs_pred"
        else:
            drive_file_path = os.path.join(params.driving.experiment_dir,
                                           "full_graph_chunks/")
            data_dict_key = "G_agg"


        # COMMENT THIS TO ONLY AGGREGATE SINGLE DRIVES / CHUNKS
        lazy_values = [process_chunk.remote(data) for data in chunk_data]
        values = ray.get(lazy_values)
        ray.shutdown()

        drive_files = list()
        drive_files.extend(glob.glob(drive_file_path + '*.pkl'))
        drive_files = sorted(drive_files)

        # Output folder
        # Create full_graph output folder if it does not exist yet
        synth_file_path = os.path.join(params.driving.experiment_dir, "full_graph/")

        G_agg = nx.DiGraph()
        for drive_idx, drive_file in enumerate(drive_files):
            drive_data_dict = pickle.load(open(drive_file, 'rb'))

            # Aggregates single drives
            if params.driving.agg_single_drives:
                if drive_idx == 0:
                    pred_graphs = drive_data_dict[data_dict_key]
                    visited_edges = drive_data_dict['visited_edges']
                    for pred_idx, pred_graph in enumerate(pred_graphs):
                        G_agg, merging_map = aggregate(G_agg,
                                                       pred_graph,
                                                       visited_edges,
                                                       threshold_px=params.driving.threshold_px,
                                                       threshold_rad=params.driving.threshold_rad,
                                                       closest_lat_thresh=params.driving.closest_lat_dist_succ,
                                                       w_decay=params.driving.w_decay,
                                                       remove=params.driving.remove_smp)
                else:
                    pred_graphs = drive_data_dict[data_dict_key]
                    visited_edges = drive_data_dict['visited_edges']
                    for pred_idx, pred_graph in enumerate(pred_graphs):
                        G_agg, merging_map = aggregate(G_agg,
                                                       pred_graph,
                                                       visited_edges,
                                                       threshold_px=params.driving.threshold_px,
                                                       threshold_rad=params.driving.threshold_rad,
                                                       closest_lat_thresh=params.driving.closest_lat_dist_succ,
                                                       w_decay=params.driving.w_decay,
                                                       remove=params.driving.remove_smp)
            # Aggregates chunks
            else:
                chunk_graph = drive_data_dict[data_dict_key]
                visited_edges = drive_data_dict['visited_edges']
                G_agg, merging_map = aggregate(G_agg,
                                               chunk_graph,
                                               visited_edges,
                                               threshold_px=params.driving.threshold_px,
                                               threshold_rad=params.driving.threshold_rad,
                                               closest_lat_thresh=params.driving.closest_lat_dist_succ,
                                               w_decay=params.driving.w_decay,
                                               remove=params.driving.remove_smp)

            # Plot graphs when using aggregating chunks
            if params.driving.agg_single_drives is False:
                # Define some dummy values for ego pose
                ego_x_y_yaw = np.array([2000, 2000, 0.0])
                start = np.array(ego_x_y_yaw[:2])
                end = np.array([ego_x_y_yaw[0] + 30 * np.cos(ego_x_y_yaw[2]), ego_x_y_yaw[0] + 30 * np.sin(ego_x_y_yaw[2])])
                plot_step_vector = end - start

                # Load and pad satellite image
                sat_img = np.pad(load_sat_img(params), ((512, 512), (512, 512), (0, 0)), 'symmetric')
                roi_area_xxyy = np.array([0, sat_img.shape[0],  0, sat_img.shape[1]])

                # Render and plot full graph
                graph_canvas = render_current_graph(params, G_agg, G_agg, G_agg, visited_edges,
                                                    np.ascontiguousarray(sat_img), ego_x_y_yaw, plot_step_vector,
                                                    roi_area_xxyy, [], 0)
                combined_plot = cv2.addWeighted(sat_img, 0.3, graph_canvas, 0.7, 0.0)
                write_path = os.path.join(params.driving.experiment_dir,
                                          "full_graph/full_print_{:03d}.png".format(drive_idx))
                cv2.imwrite(write_path, cv2.cvtColor(combined_plot, cv2.COLOR_BGR2RGB))

            # Store intermediate/final agg results
            data_dict = {"G_agg": G_agg, "config": params.driving}
            with open(os.path.join(params.driving.experiment_dir,
                                   "full_graph/full_data_dict_{}.pkl".format(drive_idx)), "wb") as f:
                pickle.dump(data_dict, f)
            print("Finished with drive {}!".format(drive_idx))

        print("Finished with aggregating over all drives!")
