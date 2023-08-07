import copy
import networkx as nx
import numpy as np
import torch
import os
import argparse
from glob import glob
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import codecs
import json
import torchvision.transforms as T

from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import nearest_points

import ray
from ray.util.multiprocessing import Pool

from methods.lanegnn.utils.params import ParamLib
from methods.lanegnn.utils.continuous import get_cropped_edge_img, is_in_mask_loop, get_gt_sdf_with_direction, get_pointwise_edge_gt
from methods.lanegnn.utils.sampling import get_delaunay_triangulation, halton, get_random_edges
from methods.regressors import build_net



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_context_centerline_regressor(ckpt=None, use_cuda=False):
    model = build_net.build_network(snapshot=ckpt, backend='resnet152', use_cuda=use_cuda, n_classes=1)
    return model

def get_ego_centerline_regressor(ckpt=None, use_cuda=False, num_channels=3):
    model = build_net.build_network(snapshot=ckpt, backend='resnet152', use_cuda=use_cuda, n_classes=1, num_channels=num_channels)
    return model


def process_chunk(data):

    # Load the data for this chunk.
    rgb_files, rgb_context_files, sdf_ego_files, sdf_context_files, drivable_files, json_files, params = data

    print("Using context regressor checkpoint: ", params.preprocessing.context_regressor_ckpt)
    print("Using ego regressor checkpoint:     ", params.preprocessing.ego_regressor_ckpt)

    context_regressor = get_context_centerline_regressor(ckpt=params.preprocessing.context_regressor_ckpt,
                                                         use_cuda=False)
    ego_regressor = get_ego_centerline_regressor(ckpt=params.preprocessing.ego_regressor_ckpt,
                                                 num_channels=params.preprocessing.ego_regressor_num_channels,
                                                 use_cuda=False)

    context_regressor.eval()
    ego_regressor.eval()
    chunk_it_idx = 0

    for rgb_file, rgb_context_file, sdf_ego_file, sdf_context_file, drivable_file, json_file in zip(rgb_files, rgb_context_files, sdf_ego_files, sdf_context_files, drivable_files, json_files):

        # try:
        rgb_context = np.asarray(Image.open(rgb_context_file)) # int [0, 255]
        rgb = np.asarray(Image.open(rgb_file)) # int [0, 255]
        drivable = np.array(Image.open(drivable_file))
        graph = json.loads(codecs.open(json_file, 'r', encoding='utf-8').read())

        scene_tag = rgb_file.split('/')[-1].split('-')[0]
        city_name = rgb_file.split('/')[-3]

        # Skip existing samples
        maybe_existing = os.path.join(params.paths.export_dir, scene_tag) + '_{}-rgb-context.pth'.format(city_name)
        if os.path.exists(maybe_existing):
            print('Skipping existing sample: {}'.format(maybe_existing))
            continue

        # GT graph representation
        waypoints = np.array(graph["bboxes"])
        relation_labels = np.array(graph["relation_labels"])

        # Get 1 graph start node and N graph end nodes
        G_gt_nx = nx.DiGraph()
        for e in relation_labels:
            if not G_gt_nx.has_node(e[0]):
                G_gt_nx.add_node(e[0], pos=waypoints[e[0]])
            if not G_gt_nx.has_node(e[1]):
                G_gt_nx.add_node(e[1], pos=waypoints[e[1]])
            G_gt_nx.add_edge(e[0], e[1])

        # Throw out all easy samples
        max_node_degree = max([G_gt_nx.degree(x) for x in G_gt_nx.nodes()])
        if max_node_degree >= 3 or chunk_it_idx % 10 == 0:
             print("[{}/{}] Step counter fulfilled or max degree high enough: {}".format(chunk_it_idx, len(rgb_files), max_node_degree))
             pass
        else:
            print("[{}/{}] No intersection scenario -> skipping".format(chunk_it_idx, len(rgb_files)))
            chunk_it_idx += 1
            continue

        start_node = [x for x in G_gt_nx.nodes() if G_gt_nx.in_degree(x) == 0 and G_gt_nx.out_degree(x) > 0][0]
        start_node_pos = G_gt_nx.nodes[start_node]['pos']
        end_nodes = [x for x in G_gt_nx.nodes() if G_gt_nx.out_degree(x) == 0 and G_gt_nx.in_degree(x) > 0]
        end_node_pos_list = [G_gt_nx.nodes[x]['pos'] for x in end_nodes]

        gt_lines = []
        gt_multilines = []
        gt_graph_edge_index = list()
        for l in relation_labels:
            line = [waypoints[l[0], 0], waypoints[l[0], 1], waypoints[l[1], 0], waypoints[l[1], 1]]
            gt_multilines.append(((waypoints[l[0], 0], waypoints[l[0], 1]), (waypoints[l[1], 0], waypoints[l[1], 1])))
            gt_lines.append(line)
            gt_graph_edge_index.append((l[0], l[1]))

        gt_lines = np.array(gt_lines)

        gt_lines_shapely = []
        for l in gt_lines:
            x1 = l[0]
            y1 = l[1]
            x2 = l[2]
            y2 = l[3]
            gt_lines_shapely.append(LineString([(x1, y1), (x2, y2)]))

        gt_multiline_shapely = MultiLineString(gt_multilines)

        # Remove park areas and set max-value for drivable surface
        # park-area 255, drivable 128, non-drivable 0

        num_park_pixels = np.sum(drivable == 255)
        num_lane_pixels = np.sum(drivable == 128)

        if float(num_park_pixels) / (float(num_lane_pixels) + 1) > 0.2:
            drivable[drivable > 128] = 255.0
            drivable[drivable < 128] = 0.0
            drivable[drivable == 128] = 255.0
        else:
            drivable[drivable > 128] = 0.0
            drivable[drivable < 128] = 0.0
            drivable[drivable == 128] = 255.0

        # Mask of non-drivable surface for violating edge rejection
        # [depr] non_drivable_mask = drivable < 255
        # [depr] sdf_ego = cv2.GaussianBlur(sdf_ego, (31, 31), cv2.BORDER_DEFAULT)

        # Feed ego-RGB / context-RGB to regressors and produce SDF approximations / drivable surface (used for node sampling)
        # RGB2BGR is necessary because regressors are trained with cv2-color-order images
        rgbcontext2sdf = torch.from_numpy(cv2.cvtColor(rgb_context, cv2.COLOR_RGB2BGR)).permute(2, 0, 1).float() / 255.0
        context_regr = torch.nn.Sigmoid()(context_regressor(rgbcontext2sdf.unsqueeze(0)))
        context_regr_ = context_regr.detach().cpu().numpy()[0, 0]
        context_regr_smooth = cv2.GaussianBlur(context_regr_, (1, 1), cv2.BORDER_CONSTANT)


        if params.preprocessing.ego_regressor_num_channels == 4:
            # print("Using 4-channel ego-regressor")
            rgb_for_cat = torch.from_numpy(cv2.cvtColor(rgb_context, cv2.COLOR_RGB2BGR)).permute(2, 0, 1).float() / 255.0
            rgb2sdf = torch.cat((rgb_for_cat, context_regr[0]), dim=0)
        else:
            rgb2sdf = torch.from_numpy(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)).permute(2, 0, 1).float() / 255.0

        ego_regr = torch.nn.Sigmoid()(ego_regressor(rgb2sdf.unsqueeze(0)))
        ego_regr = ego_regr.detach().cpu().numpy()[0, 0]
        ego_regr_smooth = cv2.GaussianBlur(ego_regr, (1, 1), cv2.BORDER_CONSTANT)

        if params.preprocessing.ego_regressor_num_channels == 4:
            # We need to cut this down to proper size again (remove context)
            ego_regr_smooth = ego_regr_smooth[128:128 + 256, 128:128 + 256]

        # USE ego_regr_smooth < 0.08 normally
        # context_regr_smooth_sampling = context_regr_smooth[128:128 + 256, 128:128 + 256]
        # non_drivable_mask = context_regr_smooth_sampling < 0.65
        non_drivable_mask = ego_regr_smooth < 0.08

        # Normalize drivable surface to create a uniform distribution
        drivable_distrib = drivable/np.sum(drivable)


        if params.preprocessing.gt_pointwise:
            # ----- DOES NOT WORK RIGHT NOW: Coordinates greater than 255 are created
            if params.preprocessing.sampling_method == "uniform":
                # Node sampling
                # Create a flat copy of the array & sample index from 1D array
                # with prob of the original array
                flat = drivable_distrib.flatten()
                sample_index = np.random.choice(a=flat.size, p=flat, size=params.preprocessing.num_node_samples)
                adjusted_index = np.unravel_index(sample_index, drivable_distrib.shape)
                point_coords = list(zip(*adjusted_index))
                # Append starting point as a node
                point_coords.append((255, 128))

            elif params.preprocessing.sampling_method == "halton":

                # Single-level halton sampling
                point_coords = halton(2, params.preprocessing.num_node_samples-1) * 255
                halton_points = point_coords.astype(np.int32)

                # filter all points where non_drivable_mask is True and add initial node
                point_coords = halton_points[np.logical_not(non_drivable_mask[halton_points[:, 0], halton_points[:, 1]])]
                point_coords = np.concatenate((point_coords, np.array([[255, 128]])), axis=0)
        else:
            print("SDF-wise edge GT not implemented")


        # Construct edges based on obstacle rejection
        if params.preprocessing.edge_proposal_method == 'triangular':
            edge_proposal_pairs = get_delaunay_triangulation(point_coords)
        elif params.preprocessing.edge_proposal_method == 'random':
            edge_proposal_pairs = get_random_edges(point_coords)

        edge_proposal_pairs = np.unique(edge_proposal_pairs, axis=0)
        edge_proposal_pairs = edge_proposal_pairs.tolist()

        # Triangulation based edge proposal generation
        edges = list()
        edges_locs = list()
        node_gt_list = list()
        node_feats_list = list()

        for i, anchor in enumerate(point_coords):
            node_tensor = torch.tensor([anchor[0], anchor[1]]).reshape(1, -1)
            node_feats_list.append(node_tensor)
            shapely_point = Point([(anchor[1], anchor[0])])
            node_gt_score = shapely_point.distance(gt_multiline_shapely)
            node_gt_list.append(node_gt_score)

        if len(node_feats_list) == 0:
            print("No nodes found. Skipping sample")
            continue

        node_feats = torch.cat(node_feats_list, dim=0)

        for [i, j] in edge_proposal_pairs:
            anchor = point_coords[i]
            point = point_coords[j]

            if is_in_mask_loop(non_drivable_mask, anchor[1], anchor[0], point[1], point[0], params.preprocessing.N_interp):
                edges_locs.append((anchor, point))
                edges.append((i, j))

        if len(edges) == 0:
            print("No edges found. Skipping sample")
            continue

        #print("--normalize node gt")

        # Min-max scaling of node_scores
        node_gt_score = torch.FloatTensor(node_gt_list)
        node_gt_score -= node_gt_score.min()
        node_gt_score /= node_gt_score.max()
        node_gt_score = 1 - node_gt_score
        node_gt_score = node_gt_score**8

        # Scales edge img feature to VGG16 input size
        transform2vgg = T.Compose([
            T.ToPILImage(),
            T.Resize(32),
            T.ToTensor()])

        # Crop edge img feats and infer edge GT from SDF
        # print("len(edges)", len(edges))
        gt_sdf, angles_gt_dense = get_gt_sdf_with_direction(gt_lines_shapely)

        edge_attr_list = list()
        edge_img_feats_list = list()
        edge_idx_list = list()

        if params.preprocessing.gt_pointwise:
            cum_edge_dist_list = list()
            angle_penalty_list = list()

        #print("--edge feat constr")

        for edge_idx, edge in enumerate(edges):
            i, j = edge
            s_x, s_y = point_coords[i][1], point_coords[i][0]
            e_x, e_y = point_coords[j][1], point_coords[j][0]

            # if params.preprocessing.visualize:
            #     plt.arrow(s_x, s_y, e_x-s_x, e_y-s_y, color="red", width=0.5, head_width=5)

            delta_x, delta_y = e_x - s_x, e_y - s_y
            mid_x, mid_y = s_x + delta_x/2, s_y + delta_y/2

            edge_len = np.sqrt(delta_x**2 + delta_y**2)
            edge_angle = np.arctan(delta_y/(delta_x + 1e-6))

            edge_tensor = torch.tensor([edge_angle, edge_len, mid_x, mid_y]).reshape(1, -1)
            edge_attr_list.append(edge_tensor)

            # Crop edge images:
            crop_img_rgb = get_cropped_edge_img(edge_angle, mid_x, mid_y, rgb_context)
            crop_img_rgb_resized = transform2vgg(crop_img_rgb).unsqueeze(0)
            crop_img_sdf = get_cropped_edge_img(edge_angle, mid_x, mid_y, context_regr_smooth)
            crop_img_sdf_resized = transform2vgg(crop_img_sdf).unsqueeze(0)
            # RGB and SDF in range [0.0, 1.0] float32

            if params.preprocessing.gt_pointwise:
                cum_edge_distance, angle_penalty = get_pointwise_edge_gt(s_x, s_y, e_x, e_y, params.preprocessing.N_interp, gt_multiline_shapely, angles_gt_dense)

                cum_edge_dist_list.append(cum_edge_distance)
                angle_penalty_list.append(angle_penalty)
                edge_idx_list.append((i, j))

            edge_img_feats_list.append(torch.cat([crop_img_rgb_resized, crop_img_sdf_resized], dim=1))

        edge_img_feats = torch.cat(edge_img_feats_list, dim=0)

        edge_attr = torch.cat(edge_attr_list, dim=0)

        if params.preprocessing.visualize:
            plt.show()

        # Pointwise edge score normalization
        if params.preprocessing.gt_pointwise:
            try:
                cum_edge_dist_gt = np.array(cum_edge_dist_list)
                cum_edge_dist_gt -= cum_edge_dist_gt.min()
                cum_edge_dist_gt /= cum_edge_dist_gt.max()
                cum_edge_dist_gt = 1 - cum_edge_dist_gt
                edge_gt_score = cum_edge_dist_gt * np.array(angle_penalty_list)
                edge_gt_score = cum_edge_dist_gt**8
            except:
                pass

        # Now we correct the edge weights according to dijsktra path
        G_proposal_nx = nx.DiGraph()
        for edge_idx, e in enumerate(edge_idx_list):
            if not G_proposal_nx.has_node(e[0]):
                G_proposal_nx.add_node(e[0], pos=point_coords[e[0]])
            if not G_proposal_nx.has_node(e[1]):
                G_proposal_nx.add_node(e[1], pos=point_coords[e[1]])
            G_proposal_nx.add_edge(e[0], e[1], weight=1-edge_gt_score[edge_idx])

        # Now we search for shortest path through the G_proposal_nx from start node to end nodes
        point_coords_swapped = np.array(point_coords)[:, ::-1]
        start_node_idx = np.argmin(np.linalg.norm(point_coords_swapped - start_node_pos, axis=1))

        # Get edge distances to gt graph
        d_list = list()
        for e in edge_idx_list:
            p0 = point_coords[e[0]]
            p1 = point_coords[e[1]]
            p0 = [p0[1], p0[0]]
            p1 = [p1[1], p1[0]]
            points_interpolated = np.linspace(p0, p1, 5)
            points_interpolated = [Point(p) for p in points_interpolated]
            max_dist = max([gt_multiline_shapely.distance(p) for p in points_interpolated])
            d_list.append(max_dist)
        edge_gt_score_binary = np.array(d_list)
        edge_gt_score_binary[edge_gt_score_binary < 7] = 1
        edge_gt_score_binary[edge_gt_score_binary > 7] = 0

        # Remove non-plausible edges
        G_proposal_pruned = copy.deepcopy(G_proposal_nx)
        for edge_idx, e in enumerate(edge_idx_list):
            contains_start = False
            if e[0] == start_node_idx or e[1] == start_node_idx:
                contains_start = True
            if edge_gt_score_binary[edge_idx] == 0 and not contains_start:
                G_proposal_pruned.remove_edge(e[0], e[1])
        # Remove isolated nodes
        G_proposal_pruned.remove_nodes_from(list(nx.isolates(G_proposal_pruned)))

        # Find ego-reachable end nodes
        end_nodes = [x for x in G_gt_nx.nodes() if G_gt_nx.out_degree(x) == 0 and G_gt_nx.in_degree(x) > 0]
        end_node_pos_list = [G_gt_nx.nodes[x]['pos'] for x in end_nodes]

        reachable_end_node_idx_list = list()
        for end_node in end_nodes:
            closest_node_idx = None
            closest_node_dist = 100000
            end_node_pos = G_gt_nx.nodes[end_node]['pos']
            nothing_found = True
            steps_backwards = 0
            while nothing_found and steps_backwards < 7:
                for node_idx in G_proposal_pruned.nodes:
                    node_pos = G_proposal_pruned.nodes[node_idx]['pos'][::-1]
                    dist = np.linalg.norm(end_node_pos - node_pos)
                    if dist < closest_node_dist:
                        closest_node_dist = dist
                        closest_node_idx = node_idx
                if closest_node_dist < 8:
                    reachable_end_node_idx_list.append(closest_node_idx)
                    nothing_found = False
                else:
                    nothing_found = True
                    predecessors = list(G_gt_nx.predecessors(end_node))
                    if len(predecessors) > 0:
                        end_node = predecessors[0]
                        end_node_pos = G_gt_nx.nodes[end_node]['pos']
                        steps_backwards += 1
                    else:
                        break

        if params.preprocessing.visualize:

            plt.ion()
            # # And we plot it for debugging
            fig, axarr = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
            axarr[0].imshow(rgb)
            axarr[1].imshow(rgb)
            # Overlay rgb with non_drivable_mask
            axarr[0].imshow(non_drivable_mask, alpha=0.5, cmap='gray')
            axarr[0].title.set_text('A')
            axarr[1].title.set_text('B')

            # Plot all found endpoints in pruned graph
            for endnode_idx in reachable_end_node_idx_list:
                axarr[1].scatter(G_proposal_pruned.nodes[endnode_idx]['pos'][1],
                                 G_proposal_pruned.nodes[endnode_idx]['pos'][0],
                                 color='yellow', s=50)

            for i in point_coords:
                axarr[1].scatter(i[1], i[0], c='red', s=6.0)
            for e in G_gt_nx.edges:
                s_x, s_y = G_gt_nx.nodes[e[0]]['pos'][0], G_gt_nx.nodes[e[0]]['pos'][1]
                e_x, e_y = G_gt_nx.nodes[e[1]]['pos'][0], G_gt_nx.nodes[e[1]]['pos'][1]
                axarr[0].arrow(s_x, s_y, e_x - s_x, e_y - s_y, color='g', width=0.2, head_width=1)

            for n in G_gt_nx.nodes:
                axarr[0].scatter(G_gt_nx.nodes[n]['pos'][0], G_gt_nx.nodes[n]['pos'][1], color='g', s=4)

            # Plot all edges in reduced graph
            for e in G_proposal_pruned.edges:
                s_x, s_y = G_proposal_nx.nodes[e[0]]['pos'][1], G_proposal_nx.nodes[e[0]]['pos'][0]
                e_x, e_y = G_proposal_nx.nodes[e[1]]['pos'][1], G_proposal_nx.nodes[e[1]]['pos'][0]
                axarr[1].arrow(s_x, s_y, e_x - s_x, e_y - s_y,
                               color='dodgerblue', width=0.2, head_width=1)
            for n in G_proposal_pruned.nodes:
                axarr[1].scatter(G_proposal_pruned.nodes[n]['pos'][1], G_proposal_pruned.nodes[n]['pos'][0],
                                 color='cyan', s=4)


        # Perform Dijkstra on pruned graph
        shortest_paths = list()
        for end_node_idx in reachable_end_node_idx_list:
            try:
                shortest_paths.append(nx.shortest_path(G_proposal_pruned, start_node_idx, end_node_idx, weight="weight"))
            except Exception as e:
                print(e)


        dijkstra_edge_list = list()
        for path in shortest_paths:
            path = [start_node_idx] + path
            dijkstra_edge_list += list(zip(path[:-1], path[1:]))

        # Now we correct the edge weights according to dijsktra path
        edge_gt_score_dijkstra = edge_gt_score.copy()
        for idx in range(len(edge_idx_list)):
            e = edge_idx_list[idx]
            if (e[0], e[1]) in dijkstra_edge_list:
                # print("Maximizing score for edge {}-{} because it is in dijkstra edge list".format(e[0], e[1]))
                edge_gt_score_dijkstra[idx] = 1.0
                edge_gt_score[idx] = 1.0
        # Maybe one-hot encoding of path is better?
        edge_gt_score_dijkstra[edge_gt_score_dijkstra < 0.999] = 0

        # Now find the involved nodes and compute their deltas towards the actual GT path
        involved_nodes = list(set([item for sublist in shortest_paths for item in sublist]))
        node_pos_reg_delta_gt = np.zeros((node_gt_score.shape[0], 2))
        for node_idx in involved_nodes:
            if node_idx is not start_node_idx:
                node_p = Point(point_coords_swapped[node_idx][0], point_coords_swapped[node_idx][1])
                gt_closest_point, _ = nearest_points(gt_multiline_shapely, node_p)
                delta_pos = np.array([gt_closest_point.x, gt_closest_point.y]) - point_coords_swapped[node_idx]
                node_pos_reg_delta_gt[node_idx] = delta_pos[np.newaxis, :]

        if params.preprocessing.visualize:

            # Plot solutions on pruned graph
            for e in dijkstra_edge_list:
                s_x, s_y = G_proposal_pruned.nodes[e[0]]['pos'][1], G_proposal_pruned.nodes[e[0]]['pos'][0]
                e_x, e_y = G_proposal_pruned.nodes[e[1]]['pos'][1], G_proposal_pruned.nodes[e[1]]['pos'][0]
                axarr[1].arrow(s_x, s_y, e_x - s_x, e_y - s_y,
                               color="yellow", width=0.2, head_width=1)

            # Plot reg delta and involved nodes
            for node_idx in involved_nodes:
                axarr[0].scatter(G_proposal_pruned.nodes[node_idx]['pos'][1],
                                 G_proposal_pruned.nodes[node_idx]['pos'][0], s=20)
                delta_x = node_pos_reg_delta_gt[node_idx][0]
                delta_y = node_pos_reg_delta_gt[node_idx][1]
                axarr[0].arrow(G_proposal_pruned.nodes[node_idx]['pos'][1],
                               G_proposal_pruned.nodes[node_idx]['pos'][0],
                               delta_x,
                               delta_y,
                               color="r", width=0.2, head_width=1)
            plt.savefig(os.path.join(params.paths.export_dir, scene_tag) + '_{}-viz.png'.format(city_name))
            plt.tight_layout()
            plt.show()
            plt.savefig(os.path.join(params.paths.export_dir, scene_tag) + '_{}-viz.png'.format(city_name))
            plt.close()

        node_endpoint_gt = torch.zeros(len(node_feats), dtype=torch.long)
        for end_node_idx in reachable_end_node_idx_list:
            node_endpoint_gt[end_node_idx] = 1

        edge_gt_score = torch.from_numpy(edge_gt_score).float()
        edge_gt_score_dijkstra = torch.from_numpy(edge_gt_score_dijkstra).float()
        node_pos_reg_delta_gt = torch.from_numpy(node_pos_reg_delta_gt).float()

        gt_graph = torch.tensor(gt_lines) # [num_gt_graph_edges, 4]
        edges = torch.tensor(edges)

        if not os.path.exists(params.paths.export_dir):
            os.makedirs(params.paths.export_dir, exist_ok=True)

        print("saving to", os.path.join(params.paths.export_dir, scene_tag) + '*.pth')

        torch.save(node_feats, os.path.join(params.paths.export_dir, scene_tag) + '_{}-node-feats.pth'.format(city_name))
        torch.save(node_endpoint_gt, os.path.join(params.paths.export_dir, scene_tag) + '_{}-node-endpoint-gt.pth'.format(city_name))
        torch.save(node_pos_reg_delta_gt, os.path.join(params.paths.export_dir, scene_tag) + '_{}-node-pos-reg-delta-gt.pth'.format(city_name))
        torch.save(edges, os.path.join(params.paths.export_dir, scene_tag) + '_{}-edges.pth'.format(city_name))
        torch.save(edge_attr, os.path.join(params.paths.export_dir, scene_tag) + '_{}-edge-attr.pth'.format(city_name))

        # convert edge_img_feats to float range [0, 255] before casting to uint8 [0, 255]
        edge_img_feats = edge_img_feats * 255.0
        torch.save(edge_img_feats.to(torch.uint8), os.path.join(params.paths.export_dir, scene_tag) + '_{}-edge-img-feats.pth'.format(city_name))

        torch.save(node_gt_score, os.path.join(params.paths.export_dir, scene_tag) + '_{}-node-gt.pth'.format(city_name))
        torch.save(edge_gt_score, os.path.join(params.paths.export_dir, scene_tag) + '_{}-edge-gt.pth'.format(city_name))
        torch.save(edge_gt_score_dijkstra, os.path.join(params.paths.export_dir, scene_tag) + '_{}-edge-gt-onehot.pth'.format(city_name))
        torch.save(context_regr_smooth, os.path.join(params.paths.export_dir, scene_tag) + '_{}-context-regr-smooth.pth'.format(city_name)) # [0.0, 1.0]
        torch.save(ego_regr_smooth, os.path.join(params.paths.export_dir, scene_tag) + '_{}-ego-regr-smooth.pth'.format(city_name)) # [0.0, 1.0]
        torch.save(gt_graph, os.path.join(params.paths.export_dir, scene_tag) + '_{}-gt-graph.pth'.format(city_name))
        torch.save(torch.FloatTensor(rgb), os.path.join(params.paths.export_dir, scene_tag) + '_{}-rgb.pth'.format(city_name)) # [0.0,255.0]
        torch.save(torch.FloatTensor(rgb_context), os.path.join(params.paths.export_dir, scene_tag) + '_{}-rgb-context.pth'.format(city_name)) # [0.0,255.0]

        chunk_it_idx += 1


def process_all_chunks(params: ParamLib, chunk_size: int, raw_dataset: str, processed_dataset:str):

    path = raw_dataset
    export_dir = processed_dataset

    params.paths.export_dir = export_dir

    print("Exporting to", params.paths.export_dir)

    rgb_files = sorted(glob(path + '/*-rgb.png'))
    rgb_context_files = sorted(glob(path + '/*-rgb-context.png'))
    sdf_context_files = sorted(glob(path + '/*-sdf-context.png'))
    sdf_ego_files = sorted(glob(path + '/*-centerlines-sdf-ego.png'))
    drivable_files = sorted(glob(path + '/*-drivable.png'))
    json_files = sorted(glob(path + '/*-graph.json'))

    print(len(json_files))

    assert len(rgb_files) > 0
    assert len(rgb_files) == len(rgb_context_files)
    assert len(rgb_context_files) == len(sdf_context_files)
    assert len(sdf_context_files) == len(sdf_ego_files)
    assert len(sdf_ego_files) == len(drivable_files)
    assert len(drivable_files) == len(json_files)


    # Shuffle all files with same permutation
    np.random.seed(0)
    perm = np.random.permutation(len(rgb_files))
    rgb_files = [rgb_files[i] for i in perm]
    rgb_context_files = [rgb_context_files[i] for i in perm]
    sdf_context_files = [sdf_context_files[i] for i in perm]
    sdf_ego_files = [sdf_ego_files[i] for i in perm]
    drivable_files = [drivable_files[i] for i in perm]
    json_files = [json_files[i] for i in perm]

    # Divide the set of scenes (per split) into chunks that are each processed by a different worker node.
    rgb_chunks = list(chunks(rgb_files, chunk_size))
    rgb_context_chunks = list(chunks(rgb_context_files, chunk_size))
    sdf_context_chunks = list(chunks(sdf_context_files, chunk_size))
    drivable_chunks = list(chunks(drivable_files, chunk_size))
    json_chunks = list(chunks(json_files, chunk_size))
    sdf_ego_chunks = list(chunks(sdf_ego_files, chunk_size))

    chunk_data = list()
    for rgb_chunk, rgb_context_chunk, sdf_ego_chunk, sdf_context_chunk, drivable_chunk, json_chunk in \
            zip(rgb_chunks, rgb_context_chunks, sdf_ego_chunks, sdf_context_chunks, drivable_chunks, json_chunks):
        chunk_data.append((rgb_chunk, rgb_context_chunk, sdf_ego_chunk, sdf_context_chunk, drivable_chunk, json_chunk, params))


    # Keep for debugging:
    process_chunk(chunk_data[0])  # single-threaded execution instead of parallelized (for debugging)

    # Write preprocessing log
    global_log_file = open("prepr_log.txt", "w")
    global_log_file.write('Hello World!')
    global_log_file.close()

    # Parallelized operation
    pool = Pool()
    pool.map(process_chunk, [data for data in chunk_data])


if __name__ == '__main__':

    # ----------- Parameter sourcing --------------

    parser = argparse.ArgumentParser(description="Do Preprocessing")

    # General parameters (namespace: main)
    parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)
    # parser.add_argument('--city', type=str, help='one of all,mia,pit,pao,atx', required=True)
    parser.add_argument('--raw_dataset', type=str, help='path to read raw dataset', required=True)
    parser.add_argument('--processed_dataset', type=str, help='path to write processed dataset', required=True)
    parser.add_argument('--ego_regressor_ckpt', type=str, help='path to ego regressor checkpoint', required=True)
    parser.add_argument('--context_regressor_ckpt', type=str, help='path to context regressor checkpoint', required=True)

    opt = parser.parse_args()

    params = ParamLib(opt.config)
    params.main.overwrite(opt)
    params.preprocessing.overwrite(opt)
    params.model.overwrite(opt)

    num_cpus = params.preprocessing.num_cpus

    params.preprocessing.context_regressor_ckpt = opt.context_regressor_ckpt
    params.preprocessing.ego_regressor_ckpt = opt.ego_regressor_ckpt


    num_samples = len(glob(opt.raw_dataset + '/*-rgb.png'))
    num_chunks = int(np.ceil(num_samples / num_cpus))

    ray.init(num_cpus=num_cpus,
             include_dashboard=False,
             _system_config={"automatic_object_spilling_enabled": True,
                             "object_spilling_config": json.dumps(
                                 {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}}, )}, )

    # Construct graphs using the generated scene metadata
    process_all_chunks(params, num_chunks, opt.raw_dataset, opt.processed_dataset)
