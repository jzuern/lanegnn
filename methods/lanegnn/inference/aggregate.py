import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import nearest_points
import osmnx.distance
import math
import string, random
import torch
import cv2
from collections import defaultdict
import copy
from scipy.spatial.distance import cdist
import time
from collections import deque
import argparse
import torch_geometric
import os
import json
from tqdm import tqdm
import pickle
from PIL import Image
Image.MAX_IMAGE_PIXELS = pow(2, 35).__int__()
from glob import glob
from scipy.spatial.distance import cdist
import torchvision.transforms as T
import multiprocessing
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lanegnn.learning.lane_mpnn import LaneGNN
from lanegnn.learning.data import GraphDataset, PreGraphDataset

from lanegnn.inference.traverse_endpoint import preprocess_predictions, predict_lanegraph

from lanegnn.utils.params import ParamLib
from lanegnn.utils.misc import mean_angle_abs_diff
from lanegnn.utils.graph import get_target_data, get_gt_graph, transform_graph_to_pos_indexing
from lanegnn.utils.continuous import get_cropped_edge_img, is_in_roi, transform_keypoint_to_world


def get_closest_agg_node_from_pred_start(G_succ, G_existing, ego_agg_min_dist=30):
    """
    Queries current ego position from G_succ and provides closest node in G_existing.
    :param G_succ: networkx successor graph prediction
    :param G_existing: current lane graph that is operated on
    :param ego_agg_min_dist: max distance threshold between ego node in G_succ and closest node in G_existing
    """
    new_in_degree = dict(G_succ.in_degree(list(G_succ.nodes())))

    # Check if dict is empty
    if len(new_in_degree) == 0:
        print("G_succ is empty. No splits to validate.")
        return None

    # Get key of new_in_degree dict with minimum value
    min_in_degree_key = min(new_in_degree, key=new_in_degree.get)
    assert new_in_degree[min_in_degree_key] == 0, "Minimum in degree key is not 0. Something is wrong."
    ego_pos = G_succ.nodes[min_in_degree_key]['pos']

    # Get the closest node in G_existing with respect to ego_pos
    closest_node = None
    closest_node_dist = np.inf
    for n in G_existing.nodes():
        dist = np.linalg.norm(ego_pos - G_existing.nodes[n]['pos'])
        if dist < closest_node_dist:
            closest_node_dist = dist
            closest_node = n
    if closest_node_dist < ego_agg_min_dist:
        agg_ego_node = closest_node
    else:
        agg_ego_node = None

    return agg_ego_node

def get_parallel_paths(G, cutoff=6):
    """
    Returns all parallel paths in G that are at maximum of length 6.
    :param G: networkx graph
    :param cutoff: maximum length of paths
    """
    return [list(nx.all_simple_paths(G, i, j, cutoff=cutoff)) for i in G.nodes() for j in G.nodes() if i != j and nx.has_path(G, i, j)]


def remove_parallel_paths(G_succ, G_existing, visited_edges, ego_agg_min_dist=30, fixed_eval_lag=1):
    """
    Removes parallel paths from G_existing that are not part of the visited edges in order decrease redundancy.
    :param G_succ: networkx graph of the succesor graph prediction
    :param G_existing: current lane graph that is operated on
    :param visited_edges: list of visited edges
    :param ego_agg_min_dist: max distance threshold between ego node in G_succ and closest node in G_existing
    :param fixed_eval_lag: fixed evaluation lag (only remove paths that lie 'behind' the ego position)
    :return: G_existing with removed parallel paths
    """
    agg_ego_node = get_closest_agg_node_from_pred_start(G_succ, G_existing, ego_agg_min_dist=ego_agg_min_dist)

    visited_nodes = list()
    for e in visited_edges:
        visited_nodes += [e[0], e[1]]
    visited_nodes = set(visited_nodes)

    agg_out_degree = dict(G_existing.out_degree(list(G_existing.nodes())))
    if agg_ego_node is None:
        return G_existing
    nodes_to_be_removed = list()
    parallel_paths = [connection for connection in get_parallel_paths(G_existing, cutoff=8) if len(connection) > 0]

    # sort parallel_paths by length of first element (shortest first: get just the loops without appendices)
    parallel_paths.sort(key=lambda x: len(x[0]))
    competing_paths = defaultdict(list)
    competing_weights = defaultdict(list)
    for paths in parallel_paths:
        start_node, end_node = paths[0][0], paths[0][-1]
        rev_G_existing = G_existing.reverse()
        try:
            length = nx.shortest_path_length(rev_G_existing, source=agg_ego_node, target=end_node)
            # eval-lag is measured wrt end_node
            if length > fixed_eval_lag:
                # check if path starting node and end node fulfill degree criteria
                branch_factor = G_existing.out_degree(start_node)
                merge_factor = G_existing.in_degree(end_node)
                if branch_factor > 1 and merge_factor > 1:
                    competing_paths[(start_node, end_node)] = paths
                    competing_weights[(start_node, end_node)] = [np.sum([G_existing.nodes[k]['weight'] for k in path]) for path in
                                                           paths]
                    max_weight_path = np.argmax(competing_weights[(start_node, end_node)])
                    for i, path in enumerate(competing_paths[(start_node, end_node)]):
                        if i != max_weight_path:
                            nodes_to_be_removed += path[1:-1]
        except nx.NetworkXNoPath:
            pass

    edges_to_be_removed = list()
    # Remove parallel edges of length 1
    for edge in G_existing.edges():
        if edge not in visited_edges:
            if edge[0] in visited_nodes and edge[1] in visited_nodes:
                edges_to_be_removed += [edge]

    for e in edges_to_be_removed:
        G_existing.remove_edge(e[0], e[1])


    for t in nodes_to_be_removed:
        if G_existing.has_node(t) and t not in visited_nodes:
            G_existing.remove_node(t)
    # Remove all remaining isolated nodes
    G_existing.remove_nodes_from(list(nx.isolates(G_existing)))
    return G_existing


def remove_unvalidated_splits_merges(G_succ, G_existing, visited_edges, ego_agg_min_dist=30, fixed_eval_lag=1, split_weight_thresh=6):
    """
    Remove unvalidated splits and merges from G_existing
    Only the non-highest weighted predecessor trees are removed when deleting merges.
    :param G_succ: networkx graph of the succesor graph prediction
    :param G_existing: current lane graph that is operated on
    :param visited_edges: list of visited edges
    :param ego_agg_min_dist: max distance threshold between ego node in G_succ and closest node in G_existing
    :param fixed_eval_lag: fixed evaluation lag (only remove splits/merges that lie 'behind' the ego position)
    :param split_weight_thresh: weight threshold that allows deletion of split/merge as it was not getting robustly
        predicted
    :return: G_existing with removed unvalidated splits and merges
    """
    agg_ego_node = get_closest_agg_node_from_pred_start(G_succ, G_existing, ego_agg_min_dist=ego_agg_min_dist)
    if agg_ego_node is None:
        return G_existing
    agg_out_degree = dict(G_existing.out_degree(list(G_existing.nodes())))

    nodes_to_be_removed = list()

    # Remove all splits that do not satisfy a certain length
    edges_to_be_removed_first = list()
    for n in G_existing.nodes():
        if agg_out_degree[n] >= 2:
            if G_existing.in_degree(n) == 0:
                edges_to_be_removed_first += [(n, k) for k in G_existing.successors(n)]
            # Gather all untraversed edges leaving this node
            edges_starting_at_n = [e for e in G_existing.out_edges(n)]
            if len(edges_starting_at_n) > 0:
                for unvisited_edge in edges_starting_at_n:
                    edge_tree = nx.dfs_tree(G_existing, source=unvisited_edge[1], depth_limit=5)
                    edge_tree_nodes = list(edge_tree)
                    edge_tree_weight = np.sum([1 for k in edge_tree_nodes])
                    if edge_tree_weight <= 1:
                        for e in edge_tree.edges():
                            edges_to_be_removed_first += [e]

    for e in edges_to_be_removed_first:
        G_existing.remove_edge(e[0], e[1])

    # Remove all merges that do not satisfy a certain weight
    for n in G_existing.nodes():
        if agg_out_degree[n] >= 2:
            rev_G_existing = G_existing.reverse()
            try:
                length = nx.shortest_path_length(rev_G_existing, source=agg_ego_node, target=n)
                if length > fixed_eval_lag:

                    # REMOVE UNVALIDATED SPLITS
                    successors = list(G_existing.successors(n))
                    successor_trees = [list(nx.dfs_tree(G_existing, source=s, depth_limit=10)) for s in successors]
                    successor_tree_weights = [np.sum([G_existing.nodes[k]['weight'] for k in t]) for t in successor_trees]

                    # Delete all successor trees with weights < 2
                    for i, s in enumerate(successors):
                        if successor_tree_weights[i] <= split_weight_thresh or len(successor_trees[i]) <= 2:
                            for t in successor_trees[i]:
                                nodes_to_be_removed.append(t)


                    # REMOVE UNVALIDATED MERGES EXCEPT HIGHEST WEIGHTED PREDECESSOR TREE
                    predecessors = list(G_existing.predecessors(n))
                    predecessor_trees = [
                        list(nx.dfs_tree(rev_G_existing, source=p, depth_limit=4)) for p in
                        predecessors]
                    predecessor_tree_weights = [np.sum([G_existing.nodes[k]['weight'] for k in t]) for t
                                              in predecessor_trees]
                    if len(predecessor_tree_weights):
                        predecessor_max_weight_tree_idx = np.argmax(predecessor_tree_weights)

                    # Delete all successor trees with weights < 2
                    for i, s in enumerate(predecessors):
                        if i == predecessor_max_weight_tree_idx:
                            continue
                        if predecessor_tree_weights[i] <= split_weight_thresh or len(
                                predecessor_trees[i]) <= 2:
                            for t in predecessor_trees[i]:
                                nodes_to_be_removed.append(t)

            except nx.NetworkXNoPath as e:
                pass


    for t in nodes_to_be_removed:
        if G_existing.has_node(t):
            G_existing.remove_node(t)

    return G_existing


def naive_aggregate(G_agg, G_new, threshold_px=20, closest_node_dist_thresh=30):
    """
    Naive aggregation of two graphs. This is done by finding the closest node in G_agg to each node in G_new and
    connecting them if they are close enough. If no node is close enough, a new node is created in G_agg.
    Hint: Make sure the successive additions of G_new involve disjoint node IDs
    :param G_agg: Graph to be aggregated to
    :param G_new: Graph to be aggregated
    :param threshold_px: Threshold in pixels for local agg graphs
    :param closest_node_dist_thresh: Euclidean distance threshold in pixels for merging nodes
    :return: G_agg with G_new aggregated
    """
    # Maps from agg nodes to new nodes
    merging_map = defaultdict(list)

    # Add aggregation weight to new predictions
    for n in G_new.nodes():
        G_new.nodes[n]['weight'] = 1.0

    # Add edge angles to new graph
    for e in G_new.edges():
        G_new.edges[e]['angle'] = np.arctan2(G_new.nodes[e[1]]['pos'][1] - G_new.nodes[e[0]]['pos'][1],
                                             G_new.nodes[e[1]]['pos'][0] - G_new.nodes[e[0]]['pos'][0])

    # Get mean of angles of edges connected to each node in G_new
    for n in G_new.nodes():
        edge_angles_pred = [nx.get_edge_attributes(G_new, 'angle')[(x, n)] for x in G_new.predecessors(n)]
        edge_angles_succ = [nx.get_edge_attributes(G_new, 'angle')[(n, x)] for x in G_new.successors(n)]
        edge_angles = edge_angles_pred + edge_angles_succ
        edge_angles_sin = [np.sin(angle) for angle in edge_angles]
        edge_angles_cos = [np.cos(angle) for angle in edge_angles]
        mean_angle = np.arctan2(np.mean(edge_angles_sin), np.mean(edge_angles_cos))
        if len(edge_angles_pred) == 0 and len(edge_angles_succ) == 0:
            mean_angle = 0
        G_new.nodes[n]['mean_angle'] = mean_angle

    # What if G_agg is empty? Then just return G_new, because it's the first graph and will be used as G_agg in next iteration
    if len(G_agg.nodes) == 0:
        return G_new.copy(), merging_map

    # Assign angle attribute on edges of G_agg and G_new
    for e in G_agg.edges():
        G_agg.edges[e]['angle'] = np.arctan2(G_agg.nodes[e[1]]['pos'][1] - G_agg.nodes[e[0]]['pos'][1],
                                             G_agg.nodes[e[1]]['pos'][0] - G_agg.nodes[e[0]]['pos'][0])

    # Get mean of angles of edges connected to each node in G_agg
    for n in G_agg.nodes():
        edge_angles_pred = [nx.get_edge_attributes(G_agg, 'angle')[(x, n)] for x in G_agg.predecessors(n)]
        edge_angles_succ = [nx.get_edge_attributes(G_agg, 'angle')[(n, x)] for x in G_agg.successors(n)]
        edge_angles = edge_angles_pred + edge_angles_succ
        edge_angles_sin = [np.sin(angle) for angle in edge_angles]
        edge_angles_cos = [np.cos(angle) for angle in edge_angles]
        mean_angle = np.arctan2(np.mean(edge_angles_sin), np.mean(edge_angles_cos))
        if len(edge_angles_pred) == 0 and len(edge_angles_succ) == 0:
            mean_angle = 0
        G_agg.nodes[n]['mean_angle'] = mean_angle

    # Get node name map
    node_names_agg = list(G_agg.nodes())
    node_names_new = list(G_new.nodes())

    # Get pairwise distance between nodes in G_agg and G_new
    node_pos_agg = np.array([G_agg.nodes[n]['pos'] for n in G_agg.nodes]).reshape(-1, 2)
    node_pos_new = np.array([G_new.nodes[n]['pos'] for n in G_new.nodes]).reshape(-1, 2)
    node_distances = cdist(node_pos_agg, node_pos_new, metric='euclidean') # i: agg, j: new

    # Produce a pairwise thresholding that allows the construction of ROIs in terms of Euclidean distance
    position_criterium = node_distances < threshold_px

    closest_agg_nodes = defaultdict()
    # Loop through all new nodes (columns indexed with j)
    for j in range(position_criterium.shape[1]):
        # Loop through all close agg-nodes and construct the j-specific local agg graph
        agg_j_multilines = list()

        # Get all agg-nodes that are close to new node j
        # Use orthogonal linear coordinates system to avoid problems arising from OSMnx distance calculation
        G_agg_j = nx.MultiDiGraph(crs="EPSG:3857")
        for i in range(position_criterium.shape[0]):
            if position_criterium[i, j]: # check if agg node i is close enough to new node j
                for e in G_agg.edges(node_names_agg[i]):
                    # Add edge to local agg graph

                    G_agg_j.add_node(str(e[0]), x=G_agg.nodes[e[0]]['pos'][0], y=G_agg.nodes[e[0]]['pos'][1])
                    G_agg_j.add_node(str(e[1]), x=G_agg.nodes[e[1]]['pos'][0], y=G_agg.nodes[e[1]]['pos'][1])
                    G_agg_j.add_edge(str(e[0]), str(e[1]))
                    agg_j_multilines.append(((G_agg.nodes[e[0]]['pos'][0], G_agg.nodes[e[0]]['pos'][1]),
                                           (G_agg.nodes[e[1]]['pos'][0], G_agg.nodes[e[1]]['pos'][1])))
        agg_j_shapely = MultiLineString(agg_j_multilines)
        # Find closest edge and closest_node in agg-graph to new node j
        if len(list(G_agg_j.edges)) > 0:
            closest_node = osmnx.distance.nearest_nodes(G_agg_j,
                                                           float(G_new.nodes[node_names_new[j]]['pos'][0]),
                                                           float(G_new.nodes[node_names_new[j]]['pos'][1]),
                                                           return_dist=False)
            closest_node = eval(closest_node)
            closest_node_dist = np.linalg.norm(np.array(G_agg.nodes[closest_node]['pos']) - G_new.nodes[node_names_new[j]]['pos'])

            if closest_node_dist < closest_node_dist_thresh:
                closest_agg_nodes[node_names_new[j]] = closest_node
                updtd_closest_node_pos = (G_agg.nodes[closest_node]['weight'] * np.array(G_agg.nodes[closest_node]['pos']) \
                                         + np.array(G_new.nodes[node_names_new[j]]['pos'])) / (G_agg.nodes[closest_node]['weight'] + 1)

                # Check if the updated node is not NaN
                if not math.isnan(updtd_closest_node_pos[0] * updtd_closest_node_pos[1]):
                    G_agg.nodes[closest_node]['pos'][0], G_agg.nodes[closest_node]['pos'][1] = updtd_closest_node_pos[0], updtd_closest_node_pos[1]

                # Record merging weights
                G_agg.nodes[closest_node]['weight'] += 1

                merging_map[closest_node].append(node_names_new[j])

    # What happens to all other nodes in G_new? Add them to G_agg
    mapped_new_nodes = [*merging_map.values()]
    mapped_new_nodes = [item for sublist in mapped_new_nodes for item in sublist]
    for n in G_new.nodes():
        if n not in mapped_new_nodes:
            G_agg.add_node(n, pos=G_new.nodes[n]['pos'], weight=G_new.nodes[n]['weight'], score=G_new.nodes[n]['score'])

    for e in G_new.edges():
        n = e[0]
        m = e[1]
        angle = np.arctan2(G_new.nodes[m]['pos'][1] - G_new.nodes[n]['pos'][1],
                           G_new.nodes[m]['pos'][0] - G_new.nodes[n]['pos'][0])

        # Add completely new edges
        if n not in mapped_new_nodes and m not in mapped_new_nodes:
            G_agg.add_edge(n, m, angle=G_new.edges[e]['angle'])

        # Add leading edges
        if n in mapped_new_nodes and m not in mapped_new_nodes:
            angle = np.arctan2(G_new.nodes[m]['pos'][1] - G_agg.nodes[closest_agg_nodes[n]]['pos'][1],
                               G_new.nodes[m]['pos'][0] - G_agg.nodes[closest_agg_nodes[n]]['pos'][0])
            G_agg.add_edge(closest_agg_nodes[n], m, angle=angle)

        # Add trailing edges
        if n not in mapped_new_nodes and m in mapped_new_nodes:
            angle = np.arctan2(G_agg.nodes[closest_agg_nodes[m]]['pos'][1] - G_new.nodes[n]['pos'][1],
                               G_agg.nodes[closest_agg_nodes[m]]['pos'][0] - G_new.nodes[n]['pos'][0])
            G_agg.add_edge(n, closest_agg_nodes[m], angle=angle)
    return G_agg, merging_map


def aggregate(G_agg, G_new, visited_edges, threshold_px=20, threshold_rad=0.1, closest_lat_thresh=30, w_decay=False, remove=False):
    """
    Lateral aggregation scheme that adds G_new to G_agg while neglecting longitudinal node deviations in order to not
    skew the structure of G_agg.
    Hint: Make sure the successive additions of G_new involve disjoint node IDs
    :param G_agg: Aggregated graph
    :param G_new: New (successor) graph to be added to G_agg
    :param visited_edges: Edges that have already been visited in G_agg
    :param threshold_px: Euclidean distance threshold for local-agg-graph
    :param threshold_rad: Angular distance threshold for local-agg-graph
    :param closest_lat_thresh: Threshold for lateral merging of nodes in G_new to G_agg
    :param w_decay: Boolean flag for weight decay of G_agg nodes
    :param remove: Boolean flag for removing nodes in G_agg (splits/merges and redundant paths)
    :return: G_agg: Aggregated graph with G_new added
    :return: merging_map: Mapping from G_agg nodes to G_new nodes
    """
    # Maps from agg nodes to new nodes
    merging_map = defaultdict(list)

    # Add aggregation weight to new predictions
    if w_decay:
        new_in_degree = dict(G_new.in_degree(list(G_new.nodes())))
        # Check if dict is empty
        if len(new_in_degree) > 0:
            # Get key of new_in_degree dict with minimum value
            new_ego_root_node = min(new_in_degree, key=new_in_degree.get)
            shortest_paths_from_root = nx.shortest_path_length(G_new, new_ego_root_node)
            for n in G_new.nodes():
                G_new.nodes[n]['weight'] = 1 - 0.05 * shortest_paths_from_root[n]
    else:
        for n in G_new.nodes():
            G_new.nodes[n]['weight'] = 1.0

    # Add edge angles to new graph
    for e in G_new.edges():
        G_new.edges[e]['angle'] = np.arctan2(G_new.nodes[e[1]]['pos'][1] - G_new.nodes[e[0]]['pos'][1],
                                             G_new.nodes[e[1]]['pos'][0] - G_new.nodes[e[0]]['pos'][0])

    # Get mean of angles of edges connected to each node in G_agg
    for n in G_new.nodes():
        edge_angles_pred = [nx.get_edge_attributes(G_new, 'angle')[(x, n)] for x in G_new.predecessors(n)]
        edge_angles_succ = [nx.get_edge_attributes(G_new, 'angle')[(n, x)] for x in G_new.successors(n)]
        edge_angles = edge_angles_pred + edge_angles_succ
        edge_angles_sin = [np.sin(angle) for angle in edge_angles]
        edge_angles_cos = [np.cos(angle) for angle in edge_angles]
        mean_angle = np.arctan2(np.mean(edge_angles_sin), np.mean(edge_angles_cos))
        if len(edge_angles_pred) == 0 and len(edge_angles_succ) == 0:
            mean_angle = 0
        G_new.nodes[n]['mean_angle'] = mean_angle

    # What if G_agg is empty? Then just return G_new, because it's the first graph and will be used as G_agg in next iteration
    if len(G_agg.nodes) == 0:
        return G_new.copy(), merging_map

    if remove:
        # Remove splits as soon as traveled past them by fixed_eval_lag
        G_agg = remove_unvalidated_splits_merges(G_new, G_agg, visited_edges, ego_agg_min_dist=30, fixed_eval_lag=1, split_weight_thresh=2)
        #G_agg = remove_parallel_paths(G_new, G_agg, visited_edges, ego_agg_min_dist=30, fixed_eval_lag=1)

    # Assign angle attribute on edges of G_agg and G_new
    for e in G_agg.edges():
        G_agg.edges[e]['angle'] = np.arctan2(G_agg.nodes[e[1]]['pos'][1] - G_agg.nodes[e[0]]['pos'][1],
                                             G_agg.nodes[e[1]]['pos'][0] - G_agg.nodes[e[0]]['pos'][0])

    # Get mean of angles of edges connected to each node in G_agg
    for n in G_agg.nodes():
        edge_angles_pred = [nx.get_edge_attributes(G_agg, 'angle')[(x, n)] for x in G_agg.predecessors(n)]
        edge_angles_succ = [nx.get_edge_attributes(G_agg, 'angle')[(n, x)] for x in G_agg.successors(n)]
        edge_angles = edge_angles_pred + edge_angles_succ
        edge_angles_sin = [np.sin(angle) for angle in edge_angles]
        edge_angles_cos = [np.cos(angle) for angle in edge_angles]
        mean_angle = np.arctan2(np.mean(edge_angles_sin), np.mean(edge_angles_cos))
        if len(edge_angles_pred) == 0 and len(edge_angles_succ) == 0:
            mean_angle = 0
        G_agg.nodes[n]['mean_angle'] = mean_angle

    # Get node name map
    node_names_agg = list(G_agg.nodes())
    node_names_new = list(G_new.nodes())

    # Get pairwise distance between nodes in G_agg and G_new
    node_pos_agg = np.array([G_agg.nodes[n]['pos'] for n in G_agg.nodes]).reshape(-1, 2)
    node_pos_new = np.array([G_new.nodes[n]['pos'] for n in G_new.nodes]).reshape(-1, 2)
    node_distances = cdist(node_pos_agg, node_pos_new, metric='euclidean') # i: agg, j: new

    # Get pairwise angle difference between nodes in G_agg and G_new
    node_mean_ang_agg = np.array([G_agg.nodes[n]['mean_angle'] for n in G_agg.nodes]).reshape(-1, 1)
    node_mean_ang_new = np.array([G_new.nodes[n]['mean_angle'] for n in G_new.nodes]).reshape(-1, 1)
    node_mean_ang_distances = cdist(node_mean_ang_agg, node_mean_ang_new, lambda u, v: mean_angle_abs_diff(u, v))

    # Produce a pairwise thresholding that allows the construction of ROIs in terms of Euclidean distance
    # and angle difference
    position_criterium = node_distances < threshold_px
    angle_criterium = node_mean_ang_distances < threshold_rad
    criterium = position_criterium & angle_criterium

    closest_agg_nodes = defaultdict()

    # Loop through all new nodes (columns indexed with j)
    for j in range(criterium.shape[1]):
        # Loop through all close agg-nodes and construct the j-specific local agg graph
        agg_j_multilines = list()

        # Get all agg-nodes that are close to new node j
        # Use orthogonal linear coordinates system to avoid problems arising from OSMnx distance calculation
        G_agg_j = nx.MultiDiGraph(crs="EPSG:3857")
        for i in range(criterium.shape[0]):
            if criterium[i, j]: # check if agg node i is close enough to new node j
                for e in G_agg.edges(node_names_agg[i]):
                    # Add edge to local agg graph
                    G_agg_j.add_node(str(e[0]), x=G_agg.nodes[e[0]]['pos'][0], y=G_agg.nodes[e[0]]['pos'][1])
                    G_agg_j.add_node(str(e[1]), x=G_agg.nodes[e[1]]['pos'][0], y=G_agg.nodes[e[1]]['pos'][1])
                    G_agg_j.add_edge(str(e[0]), str(e[1]))
                    agg_j_multilines.append(((G_agg.nodes[e[0]]['pos'][0], G_agg.nodes[e[0]]['pos'][1]),
                                           (G_agg.nodes[e[1]]['pos'][0], G_agg.nodes[e[1]]['pos'][1])))
        agg_j_shapely = MultiLineString(agg_j_multilines)
        # Find closest edge and closest_node in agg-graph to new node j
        if len(list(G_agg_j.edges)) > 0:
            closest_edge, closest_lat_dist = osmnx.distance.nearest_edges(G_agg_j,
                                                        float(G_new.nodes[node_names_new[j]]['pos'][0]),
                                                        float(G_new.nodes[node_names_new[j]]['pos'][1]),
                                                        return_dist=True)
            closest_node = osmnx.distance.nearest_nodes(G_agg_j,
                                                           float(G_new.nodes[node_names_new[j]]['pos'][0]),
                                                           float(G_new.nodes[node_names_new[j]]['pos'][1]),
                                                           return_dist=False)
            closest_node = eval(closest_node)
            closest_node_dist = np.linalg.norm(np.array(G_agg.nodes[closest_node]['pos']) - G_new.nodes[node_names_new[j]]['pos'])

            if closest_lat_dist < closest_lat_thresh:
                closest_i, closest_j = eval(closest_edge[0]), eval(closest_edge[1])

                # assign second-closest to closest_node not closest_i
                if closest_i == closest_node:
                    sec_closest_node = closest_j
                else:
                    sec_closest_node = closest_i

                closest_agg_nodes[node_names_new[j]] = closest_node

                sec_closest_node_dist = np.linalg.norm(np.array(G_agg.nodes[sec_closest_node]['pos']) - G_new.nodes[node_names_new[j]]['pos'])

                closest_node_dist_x = G_agg.nodes[closest_node]['pos'][0] - G_new.nodes[node_names_new[j]]['pos'][0]
                closest_node_dist_y = G_agg.nodes[closest_node]['pos'][1] - G_new.nodes[node_names_new[j]]['pos'][1]

                alpha = np.arccos(closest_lat_dist/ closest_node_dist)
                beta = np.arctan(closest_node_dist_y / closest_node_dist_x)
                gamma = np.pi/2 - alpha - beta

                sec_alpha = np.arccos(closest_lat_dist / sec_closest_node_dist)

                closest_long_dist = closest_node_dist * np.sin(alpha)
                sec_closest_long_dist = sec_closest_node_dist * np.sin(sec_alpha)

                curr_new_node = np.array(G_new.nodes[node_names_new[j]]['pos'])
                virtual_closest_lat_node = curr_new_node + closest_long_dist * np.array([-np.cos(gamma), np.sin(gamma)])
                virtual_sec_closest_lat_node = curr_new_node + sec_closest_long_dist * np.array([np.cos(gamma), -np.sin(gamma)])

                omega_closest = 1 - closest_node_dist / (closest_node_dist + sec_closest_node_dist)
                omega_sec_closest = 1 - sec_closest_node_dist / (closest_node_dist + sec_closest_node_dist)

                # Calculating the node weights for aggregation
                closest_agg_node_weight = G_agg.nodes[closest_node]['weight']/(G_agg.nodes[closest_node]['weight'] + 1)
                closest_new_node_weight = omega_closest * 1 / (G_agg.nodes[closest_node]['weight'] + 1)
                # Normalization of closest weights
                closest_weights_sum = closest_agg_node_weight + closest_new_node_weight
                closest_agg_node_weight = closest_agg_node_weight / closest_weights_sum
                closest_new_node_weight = closest_new_node_weight / closest_weights_sum

                sec_closest_agg_node_weight = G_agg.nodes[sec_closest_node]['weight'] / (G_agg.nodes[sec_closest_node]['weight'] + 1)
                sec_closest_new_node_weight = omega_sec_closest * 1 / (G_agg.nodes[sec_closest_node]['weight'] + 1)
                # Normalization of sec-closest weights
                sec_closest_weights_sum = sec_closest_agg_node_weight + sec_closest_new_node_weight
                sec_closest_agg_node_weight = sec_closest_agg_node_weight / sec_closest_weights_sum
                sec_closest_new_node_weight = sec_closest_new_node_weight / sec_closest_weights_sum

                updtd_closest_node_pos = closest_agg_node_weight * np.array(G_agg.nodes[closest_node]['pos']) + closest_new_node_weight * np.array(virtual_closest_lat_node)
                updtd_sec_closest_node_pos = sec_closest_agg_node_weight * np.array(G_agg.nodes[sec_closest_node]['pos']) + sec_closest_new_node_weight * np.array(virtual_sec_closest_lat_node)

                # Check if the updated node is not NaN
                if not math.isnan(updtd_closest_node_pos[0] * updtd_closest_node_pos[1]):
                    G_agg.nodes[closest_node]['pos'][0], G_agg.nodes[closest_node]['pos'][1] = updtd_closest_node_pos[0], updtd_closest_node_pos[1]
                if not math.isnan(updtd_sec_closest_node_pos[0] * updtd_sec_closest_node_pos[1]):
                    G_agg.nodes[sec_closest_node]['pos'][0], G_agg.nodes[sec_closest_node]['pos'][1] = updtd_sec_closest_node_pos[0], updtd_sec_closest_node_pos[1]

                # Record merging weights
                G_agg.nodes[closest_node]['weight'] += 1
                G_agg.nodes[sec_closest_node]['weight'] += 1

                merging_map[closest_node].append(node_names_new[j])
                merging_map[sec_closest_node].append(node_names_new[j])


    # What happens to all other nodes in G_new? Add them to G_agg
    mapped_new_nodes = [*merging_map.values()]
    mapped_new_nodes = [item for sublist in mapped_new_nodes for item in sublist]
    for n in G_new.nodes():
        if n not in mapped_new_nodes:
            G_agg.add_node(n, pos=G_new.nodes[n]['pos'], weight=G_new.nodes[n]['weight'], score=G_new.nodes[n]['score'])


    for e in G_new.edges():
        n = e[0]
        m = e[1]

        angle = np.arctan2(G_new.nodes[m]['pos'][1] - G_new.nodes[n]['pos'][1],
                           G_new.nodes[m]['pos'][0] - G_new.nodes[n]['pos'][0])

        # Add completely new edges
        if n not in mapped_new_nodes and m not in mapped_new_nodes:
            G_agg.add_edge(n, m, angle=G_new.edges[e]['angle'])

        # Add leading edges
        if n in mapped_new_nodes and m not in mapped_new_nodes:
            angle = np.arctan2(G_new.nodes[m]['pos'][1] - G_agg.nodes[closest_agg_nodes[n]]['pos'][1],
                               G_new.nodes[m]['pos'][0] - G_agg.nodes[closest_agg_nodes[n]]['pos'][0])
            G_agg.add_edge(closest_agg_nodes[n], m, angle=angle)

        # Add trailing edges
        if n not in mapped_new_nodes and m in mapped_new_nodes:
            angle = np.arctan2(G_agg.nodes[closest_agg_nodes[m]]['pos'][1] - G_new.nodes[n]['pos'][1],
                               G_agg.nodes[closest_agg_nodes[m]]['pos'][0] - G_new.nodes[n]['pos'][0])
            G_agg.add_edge(n, closest_agg_nodes[m], angle=angle)
    return G_agg, merging_map
