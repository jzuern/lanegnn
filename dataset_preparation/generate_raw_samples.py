import os
import networkx as nx
import matplotlib.pyplot as plt
import json
import cv2
import glob
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
import sys
import skfmm
import pickle

from urbanlanegraph_dataset.api import *
from methods.lanegnn.utils.graph import get_supernodes
from helper import filter_roi, filter_subgraph


# Settings
max_samples = 100000  # max number of samples generated for training (only approximately)
gpickle_tile_width = 5000  # in dm
walk_length_max = 5000
skip_last_n_nodes = 50  # Stop walk along trajectory so many edges before terminal node
n_noise_iterations = 1  # Per crop, how many noise alterations should be applied to the crop
crop_size = 256  # This is the actual crop size
crop_size_large = 2 * crop_size  # This is twice the actual crop size afterwards
# End settings



# hardcode numpy seed for reproducibility
np.random.seed(0)

input_dir = sys.argv[1]
output_dir = sys.argv[2]
city = sys.argv[3]
split = sys.argv[4]

print("input_dir: ", input_dir)
print("output_dir: ", output_dir)
print("city: ", city)
print("split: ", split)


assert city in ["paloalto", "pittsburgh", "austin", "miami", "detroit", "washington"]

imname_rgb = "{}/{}/{}.png".format(input_dir, city, city)
imname_drivable = "{}/{}/{}_drivable.png".format(input_dir, city, city)
imname_centerlines = "{}/{}/{}_centerlines.png".format(input_dir, city, city)
imname_directions = "{}/{}/{}_direction.png".format(input_dir, city, city)
tile_files = "{}/{}/tiles/{}/*.gpickle".format(input_dir, city, split)


output_dir = "{}/{}".format(output_dir, city)


# SETTINGS END


if __name__ == '__main__':

    os.makedirs(output_dir + "/{}/".format(split), exist_ok=True)

    tile_files = sorted(glob.glob(tile_files))

    print("Loading images...")

    rgb_global = Image.open(imname_rgb)
    rgb_global = np.array(rgb_global)
    rgb_global = np.ascontiguousarray(cv2.cvtColor(rgb_global, cv2.COLOR_BGR2RGB))

    drivable_global = Image.open(imname_drivable)
    drivable_global = np.array(drivable_global)

    sdf_global = Image.open(imname_centerlines)
    sdf_global = np.array(sdf_global)

    directions_global = Image.open(imname_directions)
    directions_global = np.array(directions_global)

    print("Loading images...Done!")

    # Calculate how many samples we could create from this city with the settings we have:
    N_POSSIBLE_SAMPLES = 0

    for gpickle_file in tile_files:
        # G = nx.read_gpickle(gpickle_file)
        with open(gpickle_file, 'rb') as f:
            G = pickle.load(f)
        G = nx.DiGraph(G)

        G_contr = G.copy(as_view=False)
        supernodes = get_supernodes(G_contr, max_distance=1.0)
        for supernode in supernodes:
            nodes = sorted(list(supernode))
            for node in nodes[1:]:
                G_contr = nx.contracted_nodes(G_contr, nodes[0], node, self_loops=False)
        G_contr.remove_edges_from(nx.selfloop_edges(G_contr))
        starting_nodes = [n for n in G.nodes() if len(list(G.predecessors(n))) == 0]

        for walk_no, start_node in enumerate(starting_nodes):

            successor_subgraph = nx.bfs_tree(G, start_node)
            graph_successor_edges = nx.dfs_edges(successor_subgraph, start_node)

            # Generate Agent Trajectory
            agent_trajectory = [start_node]
            curr_node = start_node
            for i in range(0, walk_length_max):
                successors = [n for n in G.successors(curr_node)]
                if len(successors) == 0:
                    break
                curr_node = successors[np.random.randint(0, len(successors))]
                agent_trajectory.append(curr_node)

            agent_trajectory = agent_trajectory[0:-skip_last_n_nodes]
            if len(agent_trajectory) == 0:
                continue

            N_POSSIBLE_SAMPLES += len(agent_trajectory) - 1

    # Create list of indices that we will use to sample from the city
    sample_indices = np.round(np.linspace(0, N_POSSIBLE_SAMPLES - 1, int(max_samples/ n_noise_iterations))).astype(int)

    sample_counter = 0

    for iii, gpickle_file in enumerate(tile_files):

        print(iii+1, "/", len(tile_files), ":", gpickle_file)
        tile_id = gpickle_file.split("/")[-1].split(".")[0]

        with open(gpickle_file, 'rb') as f:
            G = pickle.load(f)
        G = nx.DiGraph(G)

        starting_nodes = [n for n in G.nodes() if len(list(G.predecessors(n))) == 0]

        print("     Found {} starting nodes.".format(len(starting_nodes)))

        # Walk the graph along the edges from the start nodes which have no incoming edges
        for walk_no, start_node in enumerate(starting_nodes):

            successor_subgraph = nx.bfs_tree(G, start_node)
            graph_successor_edges = nx.dfs_edges(successor_subgraph, start_node)

            # Generate Agent Trajectory
            agent_trajectory = [start_node]
            curr_node = start_node
            for i in range(0, walk_length_max):
                successors = [n for n in G.successors(curr_node)]
                if len(successors) == 0:
                    break
                curr_node = successors[np.random.randint(0, len(successors))]
                agent_trajectory.append(curr_node)

            # Visualize the graph for debugging
            # visualize_graph(G, successor_subgraph, agent_visited_nodes)

            # leave out the last nodes cause otherwise future trajectory is ending in image
            agent_trajectory = agent_trajectory[0:-skip_last_n_nodes]
            if len(agent_trajectory) == 0:
                continue

            step = 0
            print("     Starting walk from node: {}, found agent trajectory of length: ".format(start_node), len(agent_trajectory))

            # Iterate over agent trajectory:
            for t in range(0, len(agent_trajectory)-1, 2):
                if sample_counter not in sample_indices:
                    sample_counter += 1
                    continue

                curr_node = agent_trajectory[t]
                next_node = agent_trajectory[t+1]

                # Get the lane segment between the current and next node
                pos = G.nodes[curr_node]["pos"]
                next_pos = G.nodes[next_node]["pos"]
                curr_lane_segment = next_pos - pos

                # Get angle
                yaw = np.arctan2(curr_lane_segment[1], curr_lane_segment[0]) + np.pi / 2
                R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

                ego_x_y_yaw = np.array([pos[0], pos[1], yaw])

                x_true = ego_x_y_yaw[0]
                y_true = ego_x_y_yaw[1]
                yaw_true = ego_x_y_yaw[2]

                xs_noise = np.random.default_rng().normal(loc=x_true, scale=5, size=n_noise_iterations)
                ys_noise = np.random.default_rng().normal(loc=y_true, scale=5, size=n_noise_iterations)
                yaws_noise = np.random.default_rng().normal(loc=yaw_true, scale=0.3, size=n_noise_iterations)

                # iterate over noise samples
                for x_noise, y_noise, yaw_noise in zip(xs_noise, ys_noise, yaws_noise):

                    pos_noise = np.array([x_noise, y_noise])
                    R_noise = np.array([[np.cos(yaw_noise), -np.sin(yaw_noise)],
                                        [np.sin(yaw_noise),  np.cos(yaw_noise)]])

                    # Crop source and dest points
                    rgb_crop_ = rgb_global[int(y_noise - crop_size_large):int(y_noise + crop_size_large),
                                int(x_noise - crop_size_large):int(x_noise + crop_size_large)].copy()
                    sdf_crop_ = sdf_global[int(y_noise - crop_size_large):int(y_noise + crop_size_large),
                                int(x_noise - crop_size_large):int(x_noise + crop_size_large)].copy()
                    drivable_crop_ = drivable_global[int(y_noise - crop_size_large):int(y_noise + crop_size_large),
                                     int(x_noise - crop_size_large):int(x_noise + crop_size_large)].copy()
                    directions_crop_ = directions_global[int(y_noise - crop_size_large):int(y_noise + crop_size_large),
                                       int(x_noise - crop_size_large):int(x_noise + crop_size_large)].copy()

                    # Source points are around center in satellite image crop
                    center = np.array([crop_size_large, crop_size_large])

                    # For bottom centered
                    src_pts = np.array([[-128,  0],
                                        [-128, -255],
                                        [ 127, -255],
                                        [ 127,  0]])

                    src_pts_context = np.array([[-256,  128],
                                                [-256, -383],
                                                [ 255, -383],
                                                [ 255,  128]])

                    # Rotate source points
                    src_pts = (np.matmul(R_noise, src_pts.T).T + center).astype(np.float32)
                    src_pts_context = (np.matmul(R_noise, src_pts_context.T).T + center).astype(np.float32)

                    # Destination points are simply the corner points
                    dst_pts = np.array([[0, crop_size - 1],
                                                [0, 0],
                                                [crop_size - 1, 0],
                                                [crop_size - 1, crop_size - 1]],
                                               dtype="float32")

                    dst_pts_context = np.array([[0, crop_size_large - 1],
                                                        [0, 0],
                                                        [crop_size_large - 1, 0],
                                                        [crop_size_large - 1, crop_size_large - 1]],
                                                       dtype="float32")

                    # the perspective transformation matrix
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    M_context = cv2.getPerspectiveTransform(src_pts_context, dst_pts_context)

                    # directly warp the rotated rectangle to get the straightened rectangle
                    try:
                        rgb_crop = cv2.warpPerspective(rgb_crop_, M, (crop_size, crop_size), cv2.INTER_LINEAR)
                        drivable_crop = cv2.warpPerspective(drivable_crop_, M, (crop_size, crop_size), cv2.INTER_NEAREST)
                        sdf_crop = cv2.warpPerspective(sdf_crop_, M, (crop_size, crop_size), cv2.INTER_LINEAR)
                        rgb_crop_context = cv2.warpPerspective(rgb_crop_, M_context, (crop_size_large, crop_size_large), cv2.INTER_LINEAR)
                        sdf_crop_context = cv2.warpPerspective(sdf_crop_, M_context, (crop_size_large, crop_size_large), cv2.INTER_LINEAR)
                        directions_crop_context = cv2.warpPerspective(directions_crop_, M_context, (crop_size_large, crop_size_large), cv2.INTER_LINEAR)
                        rgb_crop_viz = rgb_crop.copy()
                    except Exception as e:
                        continue
                    # fig, axarr = plt.subplots(1, 2)
                    # axarr[0].imshow(rgb_crop)
                    # axarr[1].imshow(rgb_crop_context)
                    # plt.show()

                    successor_subgraph_visible = filter_subgraph(G, successor_subgraph, curr_node, max_distance=300)
                    successor_subgraph_visible_context = filter_subgraph(G, successor_subgraph, curr_node, max_distance=500)

                    if successor_subgraph_visible.number_of_edges() == 0:
                        continue

                    max_out_edges = max(successor_subgraph_visible.out_degree(node) for node in successor_subgraph_visible.nodes())
                    num_nodes = successor_subgraph_visible.number_of_nodes()

                    # we need large enough successor graphs
                    if num_nodes < 10:
                        continue

                    # only take few samples where there is only one out edge
                    if max_out_edges == 1:
                        continue

                    for n in successor_subgraph_visible.nodes():
                        node_pos = G.nodes[n]["pos"]
                        node_pos = node_pos - pos_noise + center
                        node_pos = cv2.perspectiveTransform(node_pos[None, None, :], M)
                        node_pos = node_pos[0, 0, :].astype(np.int32)
                        cv2.circle(rgb_crop_viz, (node_pos[0], node_pos[1]), 3, (255, 0, 255), -1)

                    # We also need to render the centerline regression lines
                    centerlines_ego_binary = np.zeros_like(rgb_crop)
                    centerlines_context_ego_binary = np.zeros_like(rgb_crop_context)
                    crop_viz = rgb_crop.copy()

                    for e in successor_subgraph_visible.edges():

                        # Render ego centerline
                        start = G.nodes[e[0]]["pos"]
                        end = G.nodes[e[1]]["pos"]
                        start = start - pos_noise + center
                        start = cv2.perspectiveTransform(start[None, None, :], M)
                        start = start[0, 0, :].astype(np.int32)
                        end = end - pos_noise + center
                        end = cv2.perspectiveTransform(end[None, None, :], M)
                        end = end[0, 0, :].astype(np.int32)
                        cv2.line(centerlines_ego_binary, (start[0], start[1]), (end[0], end[1]), (255, 255, 255), 3)

                        # Render Viz
                        cv2.arrowedLine(crop_viz, (start[0], start[1]), (end[0], end[1]), (142, 255, 0), thickness=1,
                                        line_type=cv2.LINE_AA, tipLength=0.2)

                    # Render ego context centerline
                    for e in successor_subgraph_visible_context.edges():
                        start = G.nodes[e[0]]["pos"]
                        end = G.nodes[e[1]]["pos"]
                        start = start - pos_noise + center
                        start = cv2.perspectiveTransform(start[None, None, :], M_context)
                        start = start[0, 0, :].astype(np.int32)
                        end = end - pos_noise + center
                        end = cv2.perspectiveTransform(end[None, None, :], M_context)
                        end = end[0, 0, :].astype(np.int32)
                        cv2.line(centerlines_context_ego_binary, (start[0], start[1]), (end[0], end[1]), (255, 255, 255), 3)

                    centerlines_ego_binary = (centerlines_ego_binary[:, :, 0] > 0).astype(np.uint8)
                    centerlines_context_ego_binary = (centerlines_context_ego_binary[:, :, 0] > 0).astype(np.uint8)

                    successful_sdf_render = False
                    try:
                        f = 20  # distance function scale
                        sdf_ego = skfmm.distance(1 - centerlines_ego_binary)
                        sdf_ego[sdf_ego > f] = f
                        sdf_ego = sdf_ego / f
                        sdf_ego = 1 - sdf_ego
                        sdf_ego = (255 * sdf_ego).astype(np.uint8)

                        sdf_context_ego = skfmm.distance(1 - centerlines_context_ego_binary)
                        sdf_context_ego[sdf_context_ego > f] = f
                        sdf_context_ego = sdf_context_ego / f
                        sdf_context_ego = 1 - sdf_context_ego
                        sdf_context_ego = (255 * sdf_context_ego).astype(np.uint8)
                        successful_sdf_render = True
                    except Exception as e:
                        sdf_context_ego = np.zeros_like(centerlines_context_ego_binary).astype(np.uint8)
                        sdf_ego = np.zeros_like(centerlines_ego_binary).astype(np.uint8)
                        successful_sdf_render = False

                    # Generate obj_relation_triplets and obj_boxes
                    obj_boxes = []
                    for n in G.nodes():
                        p = G.nodes[n]["pos"]
                        p = p - pos_noise + center
                        p = cv2.perspectiveTransform(p[None, None, :], M)
                        p = p[0, 0, :]
                        obj_boxes.append([p[0], p[1]])
                    obj_boxes = np.array(obj_boxes)

                    obj_relation_triplets = []
                    for e in successor_subgraph_visible.edges():
                        obj_relation_triplets.append([e[0], e[1]])

                    obj_relation_triplets = np.array(obj_relation_triplets)

                    obj_boxes, obj_relation_triplets = filter_roi(obj_boxes, obj_relation_triplets, 0, 0, 256, 256)

                    # We also need to filter away all isolated anchors not connected to the graph
                    valid_box_indices = np.zeros(len(obj_boxes), dtype=bool)
                    for i in range(len(obj_boxes)):
                        if len(np.where(obj_relation_triplets[:, 0] == i)[0]) > 0 or len(
                                np.where(obj_relation_triplets[:, 1] == i)[0]) > 0:
                            valid_box_indices[i] = True

                    # Perform index remapping
                    index_map = np.cumsum(valid_box_indices)
                    obj_relation_triplets_ = obj_relation_triplets.copy()
                    for k in range(len(obj_relation_triplets)):
                        idx = obj_relation_triplets_[k, 0]
                        obj_relation_triplets[k, 0] = index_map[idx] - 1
                        idx = obj_relation_triplets_[k, 1]
                        obj_relation_triplets[k, 1] = index_map[idx] - 1

                    # Filter out all invalid boxes
                    obj_boxes = obj_boxes[valid_box_indices]

                    obj_labels = np.ones(obj_boxes.shape[0], dtype=np.int32)
                    obj_relations = np.zeros([len(obj_boxes), len(obj_boxes)], dtype=np.uint8)
                    for triplet in obj_relation_triplets:
                        obj_relations[triplet[0], triplet[1]] = 1

                    # Only keep a few of boring samples
                    if np.sum(obj_relations) == 0:
                        continue

                    max_node_degree = np.max(np.sum(obj_relations, axis=0))
                    num_connections = len(obj_relation_triplets)


                    # Serialize everything
                    cv2.imwrite("{}/{}/{}_{:03d}_{:03d}-rgb.png".format(output_dir, split, tile_id, walk_no, step), rgb_crop)
                    cv2.imwrite("{}/{}/{}_{:03d}_{:03d}-rgb-context.png".format(output_dir, split, tile_id, walk_no, step), rgb_crop_context)
                    cv2.imwrite("{}/{}/{}_{:03d}_{:03d}-centerlines-ego.png".format(output_dir, split, tile_id, walk_no, step), centerlines_ego_binary*255)
                    cv2.imwrite("{}/{}/{}_{:03d}_{:03d}-centerlines-sdf-ego.png".format(output_dir, split, tile_id, walk_no, step), sdf_ego)
                    cv2.imwrite("{}/{}/{}_{:03d}_{:03d}-centerlines-sdf-ego-context.png".format(output_dir, split, tile_id, walk_no, step), sdf_context_ego)
                    cv2.imwrite("{}/{}/{}_{:03d}_{:03d}-centerlines-sdf.png".format(output_dir, split, tile_id, walk_no, step), sdf_crop)
                    cv2.imwrite("{}/{}/{}_{:03d}_{:03d}-centerlines-sdf-context.png".format(output_dir, split, tile_id, walk_no, step), sdf_crop_context)
                    cv2.imwrite("{}/{}/{}_{:03d}_{:03d}-drivable.png".format(output_dir, split, tile_id, walk_no, step), drivable_crop)
                    cv2.imwrite("{}/{}/{}_{:03d}_{:03d}-directions-context.png".format(output_dir, split, tile_id, walk_no, step), directions_crop_context)
                    cv2.imwrite("{}/{}/{}_{:03d}_{:03d}-viz.png".format(output_dir, split, tile_id, walk_no, step), crop_viz)

                    targets_dict = {}
                    targets_dict["bboxes"] = obj_boxes.tolist()
                    targets_dict["labels"] = obj_labels.tolist()
                    targets_dict["pred_labels"] = obj_relations.tolist()
                    targets_dict["relation_labels"] = obj_relation_triplets.tolist()
                    targets_dict["ego_x_y_yaw"] = [x_noise, y_noise, yaw_noise]

                    with open("{}/{}/{}_{:03d}_{:03d}-graph.json".format(output_dir, split, tile_id, walk_no, step), 'w', encoding='utf-8') as f:
                        json.dump(targets_dict, f, ensure_ascii=False, indent=4, separators=(',', ':'), sort_keys=True)


                    # serialize successor graph

                    for n in successor_subgraph_visible.nodes():
                        pos = G.nodes[n]["pos"]
                        pos = pos - pos_noise + center
                        pos = cv2.perspectiveTransform(pos[None, None, :], M)
                        pos = pos[0, 0, :]
                        successor_subgraph_visible.nodes[n]["pos"] = pos

                    # relabel nodes from 0 to len(successor_subgraph_visible.nodes())
                    mapping = {n: i for i, n in enumerate(successor_subgraph_visible.nodes())}
                    successor_subgraph_visible = nx.relabel_nodes(successor_subgraph_visible, mapping)

                    # delete all nodes outside of the crop
                    successor_subgraph_visible_ = successor_subgraph_visible.copy()
                    for n in successor_subgraph_visible_.nodes():
                        pos = successor_subgraph_visible.nodes[n]["pos"]
                        if pos[0] < 0 or pos[0] > 256 or pos[1] < 0 or pos[1] > 256:
                            successor_subgraph_visible.remove_node(n)

                    # fig, ax = plt.subplots(figsize=(10, 10))
                    # ax.aspect = 'equal'
                    # ax.imshow(rgb_crop)
                    # nx.draw_networkx(successor_subgraph_visible,
                    #                  ax=ax,
                    #                  pos=nx.get_node_attributes(successor_subgraph_visible, 'pos'),
                    #                  with_labels=False,
                    #                  width=2,
                    #                  arrowsize=4,
                    #                  node_size=20)
                    # plt.show()

                    # nx.write_gpickle(successor_subgraph_visible, "{}/{}/{}_{:03d}_{:03d}-graph.gpickle".format(output_dir, split, tile_id, walk_no, step))
                    gpickle_fname = "{}/{}/{}_{:03d}_{:03d}-graph.gpickle".format(output_dir, split, tile_id, walk_no, step)
                    with open(gpickle_fname, 'wb') as f:
                        pickle.dump(successor_subgraph_visible, f, pickle.HIGHEST_PROTOCOL)

                    sample_counter += 1
                    step += 1