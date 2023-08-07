import networkx as nx
import numpy as np
import cv2
import os

# Please only comment out, do not delete
# os.environ['CUDA_VISIBLE_DEVICES'] = "4"
from torch_geometric.data import Batch

import networkx as nx
import argparse
from collections import defaultdict
from tqdm import tqdm
import time
import torch
import torch.utils.data
import torch_geometric.data
from PIL import Image
from shapely.geometry import LineString, MultiLineString, Point

#  Please only commment out, do not delete
import matplotlib
#matplotlib.use('Agg')

from queue import PriorityQueue
import matplotlib.pyplot as plt

from methods.lanegnn.utils.params import ParamLib
from methods.lanegnn.utils.graph import unbatch_edge_index

from methods.lanegnn.learning.lane_mpnn import LaneGNN
from methods.lanegnn.learning.data import PreGraphDataset


def preprocess_predictions(params, model, data, gt_available=True):
    """
    Performs LaneGNN forward pass, thresholds the graph prediction and create graph structure to
    traverse with UCS
    :param params: parameter config
    :param model: trained LaneGNN instance
    :param data: PyG data object fed to LaneGNN
    :param gt_available: adds ground truth for comparison
    """
    # LOAD DATA
    if params.model.dataparallel:
        data = [item.to(params.model.device) for item in data]
    else:
        data = data.to(params.model.device)

    with torch.no_grad():
        edge_scores, node_scores, endpoint_scores = model(data)

    edge_scores = torch.nn.Sigmoid()(edge_scores).squeeze()
    node_scores = torch.nn.Sigmoid()(node_scores).squeeze()
    endpoint_scores = torch.nn.Sigmoid()(endpoint_scores).squeeze()

    edge_scores_pred = edge_scores.detach().cpu().numpy()
    node_scores_pred = node_scores.detach().cpu().numpy()
    endpoint_scores_pred = endpoint_scores.detach().cpu().numpy()

    # Convert list of Data to DataBatch for post-processing
    if params.model.dataparallel:
        data = Batch.from_data_list(data)

    node_pos = data.x.cpu().numpy()
    node_pos[:, [1, 0]] = node_pos[:, [0, 1]]

    if gt_available:
        edge_scores_gt_onehot = data.edge_gt_onehot.cpu().numpy()
    else:
        edge_scores_gt_onehot = None

    img_rgb = data.rgb.cpu().numpy()[0:256, :, :]
    node_pos = data.x.cpu().numpy()
    node_pos[:, [1, 0]] = node_pos[:, [0, 1]]

    fut_nodes = defaultdict(list)
    past_nodes = defaultdict(list)

    # Get starting position
    startpoint_idx = node_pos.shape[0] - 1

    # ADD ALL EDGES ABOVE THRESHOLD TO GRAPH
    nx_selected = nx.DiGraph()
    for edge_idx, edge in enumerate(data.edge_index.T):
        i, j = edge
        i, j = i.item(), j.item()
        if edge_scores_pred[edge_idx] > 0.5:
            nx_selected.add_edge(i, j, weight=1 - edge_scores_pred[edge_idx])
            nx_selected.add_node(j, pos=node_pos[j], score=node_scores_pred[j])
            nx_selected.add_node(i, pos=node_pos[i], score=node_scores_pred[i])
        else:
            # add startpoint-connecting edges to graph in any case
            if i == startpoint_idx or j == startpoint_idx:
                nx_selected.add_edge(i, j, weight=1 - edge_scores_pred[edge_idx])
                nx_selected.add_node(j, pos=node_pos[j], score=node_scores_pred[j])
                nx_selected.add_node(i, pos=node_pos[i], score=node_scores_pred[i])

    # Write fut_nodes and past_nodes dictionaries to be used in UCS
    for local_edge_idx, edge in enumerate(nx_selected.edges()):
        i, j = edge
        fut_nodes[i].append((j, nx_selected.edges[edge]['weight']))
        past_nodes[j].append((i, nx_selected.edges[edge]['weight']))

    # Sort each list entry in fut_nodes & past_nodes by weight
    for key in fut_nodes:
        fut_nodes[key] = sorted(fut_nodes[key], key=lambda x: x[1])
    for key in past_nodes:
        past_nodes[key] = sorted(past_nodes[key], key=lambda x: x[1])


    return edge_scores_pred, node_scores_pred, endpoint_scores_pred, edge_scores_gt_onehot, img_rgb, \
           node_pos, fut_nodes, past_nodes, startpoint_idx


def get_endpoint_ranking(endpoint_scores):
    """
    Sorts endpoint nodes based on their assigned scores for UCS
    :param endpoint_scores: terminal node scores
    :return endpoint_ranking: ranked indices of terminal nodes
    """
    endpoint_ranking = endpoint_scores.argsort()[::-1]
    return endpoint_ranking


def uniform_cost_search(fut_nodes, start_idx, goal_idx, node_pos, rad_thresh, debug=False):
    """
    Takes thresholded graph and performs uniform cost search
    :param fut_nodes: graph dictionary holding edges
    :param start_idx: node index of start node
    :param goal_idx: node index of goal node
    :param node_pos: positions of graph nodes in order to check vicinity to goal nodes
    :param rad_thresh: threshold that prevents hard turns
    :param debug: flag to print debug outputs
    :returns priority_queue: Either type list (graph traversal) or type PriorityQueue (no output)
    """
    visited = set()
    priority_queue = PriorityQueue()
    priority_queue.put((0, [start_idx]))

    while priority_queue:
        if priority_queue.empty():
            if debug:
                print('distance: infinity \nroute: \nnone')
            break

        # priority queue automatically sorts by first element = cost
        cost, route = priority_queue.get()
        curr_idx = route[len(route) - 1]

        if curr_idx not in visited:
            visited.add(curr_idx)
            # Check for goal
            if np.linalg.norm(node_pos[curr_idx]-node_pos[goal_idx]) < rad_thresh:
                route.append(cost)
                if debug:
                    display_route(route)
                return route

        children_idcs = get_unvisited_reachable_nodes(fut_nodes, curr_idx, route, visited, node_pos)

        for _, child_idx in enumerate(children_idcs):
            #print(fut_nodes[child_idx])
            rel_index_child = [elt[0] for elt in fut_nodes[curr_idx]].index(child_idx)
            child_cost = cost + fut_nodes[curr_idx][rel_index_child][1]
            temp = route[:]
            temp.append(child_idx) # Add children node to route
            priority_queue.put((child_cost, temp))

    return priority_queue


def display_route(route):
    """
    Prints edges contained in graph traversal
    :param route: ordered graph traversal with traversal cost as last list item
    """
    length = len(route)
    distance = route[-1]
    print('Cost: %s' % distance)
    print('Best route: ')
    count = 0
    while count < (length - 2):
        print('%s -> %s' % (route[count], route[count + 1]))
        count += 1
    return

def filter_endpoints(unvisited_endpoints, node_pos, lanegraph_shapely, rad_thresh):
    """
    Loop through all endpoints and check whether they are close to the current state of the lanegraph
    that is represented as a Shapely multiline object
    :param unvisited_endpoints: list of unvisited terminal nodes
    :param node_pos: position of graph nodes
    :param lanegraph_shapely: so-far produced graph structure
    :param rad_thresh: radius threshold used to check vicinity to current lane graph.
    :return unvisited_endpoints: list of endpoints that are not coverd by the current lane graph
    """

    for idx, val in enumerate(unvisited_endpoints):
        if val:
            endpoint_point = Point([(node_pos[idx, 0], node_pos[idx, 1])])
            endpoint_dist = endpoint_point.distance(lanegraph_shapely)
            if endpoint_dist < rad_thresh:
                unvisited_endpoints[idx] = False
    return unvisited_endpoints


def predict_lanegraph(fut_nodes, start_idx, node_scores_pred, endpoint_scores_pred, node_pos, endpoint_thresh=0.9, rad_thresh=50, debug=False):
    """
    Takes the thresholded graph, performs uniform-cost search from the start node to multiple terminal nodes.
    Removes redundant terminal node that are already considered. Returns pruned successor lane graph.
    :param fut_nodes: graph edges used in UCS
    :param start_idx: index of starting node
    :param node_scores_pred: node scores
    :param endpoint_scores_pred: terminal node scores
    :param node_pos: position of nodes in image space
    :param endpoint_thresh: threshold that needs to be met in order to consider a terminal node
    :param rad_thresh: Euclidean radius threshold for which terminal nodes are disregarded due to vicinity to lane graph
    :param debug: flag for more debug output
    :return lanegraph: pruned lanegraph that is used in aggregate() next
    """
    # Get endpoint ranking and start with maximum first
    lanegraph = nx.DiGraph()
    multilines = []
    lanegraph_shapely = MultiLineString()
    node_ranking = get_endpoint_ranking(endpoint_scores_pred)
    goal_frontier = node_ranking[0]
    unvisited_endpoints = endpoint_scores_pred > endpoint_thresh

    while goal_frontier:
        ucs_output = uniform_cost_search(fut_nodes, start_idx, goal_frontier, node_pos, rad_thresh, debug=debug)
        if isinstance(ucs_output, PriorityQueue):
            if debug:
                print('No route found')
            unvisited_endpoints[goal_frontier] = False

        elif isinstance(ucs_output, list):
            route = ucs_output

            route_idx = 0
            while route_idx < (len(route) - 2):
                # Add edges to lanegraph object and shapely object
                # Check if edge is already contained in lanegraph
                lanegraph.add_node(route[route_idx], pos=node_pos[route[route_idx]], score=node_scores_pred[route[route_idx]])
                lanegraph.add_node(route[route_idx + 1], pos=node_pos[route[route_idx + 1]], score=node_scores_pred[route[route_idx + 1]])
                lanegraph.add_edge(route[route_idx], route[route_idx + 1])
                multilines.append(((node_pos[route[route_idx], 0], node_pos[route[route_idx], 1]), (node_pos[route[route_idx + 1], 0], node_pos[route[route_idx + 1], 1])))
                # Set edge costs of all edges contained in lanegraph to 0 to enforce future traversal of these edges
                child_list_idx = [elt[0] for elt in fut_nodes[route[route_idx]]].index(route[route_idx + 1])

                fut_nodes[route[route_idx]][child_list_idx] = (fut_nodes[route[route_idx]][child_list_idx][0], 0)
                route_idx += 1
            lanegraph_shapely = MultiLineString(multilines)

        # Filter endpoints based on already found paths & select next maximum-score endpoint and add to frontier
        unvisited_endpoints = filter_endpoints(unvisited_endpoints, node_pos, lanegraph_shapely, rad_thresh)
        node_ranking = get_endpoint_ranking(endpoint_scores_pred)

        # Look whether there are remaining valid endpoints
        if np.sum(unvisited_endpoints) > 0:
            for _, node_idx in enumerate(node_ranking):
                bool_value = unvisited_endpoints[node_idx]
                if unvisited_endpoints[node_idx]:
                    # Check for endpoints still to be traversed that are close enough to the set
                    # of thresholded nodes
                    fut_nodes[node_idx] = [(elt[0], 0) for elt in fut_nodes[node_idx]]
                    # Get all keys and values from fut_nodes
                    all_values_idcs = [item[0] for sublist in fut_nodes.values() for item in sublist]
                    node_set = set(fut_nodes.keys()).union(set(all_values_idcs))

                    cand_dists = defaultdict()
                    for cand_idx in node_set:
                        cand_dists[cand_idx] = np.linalg.norm(node_pos[cand_idx]-node_pos[node_idx])

                    if min(cand_dists) < 50:
                        closest_endpoint_cand = min(cand_dists, key=cand_dists.get)
                    else:
                        continue
                    unvisited_endpoints[node_idx] = False
                    goal_frontier = closest_endpoint_cand
                    break
        else:
            goal_frontier = None

    return lanegraph


def get_unvisited_reachable_nodes(fut_nodes, curr_idx, curr_path, visited, node_pos):
    """
    Obtains potential successor nodes that are unvisited and do not show a too-large angle difference
    compared to the currently produced successor lane graph.
    :param fut_nodes: graph edges in dictionary-format
    :param curr_idx: current frontier node
    :param curr_path: path up to frontier node
    :param visited: list of visited node indices
    :param node_pos: node position matrix
    :return unvisited_nodes: indices of potential successor nodes that were not traversed yet
    """
    unvisited_nodes = []
    for node in fut_nodes[curr_idx]:
        if node[0] not in visited:
            if len(curr_path) == 1:
                unvisited_nodes.append(node[0])
            else:
               angle_diff = node_in_corridor(node[0], curr_idx, curr_path[-2], node_pos)
               if angle_diff < np.pi / 4:
                   unvisited_nodes.append(node[0])
            #else:
            #    unvisited_nodes.append(node[0])
    return unvisited_nodes

def node_in_corridor(fut_idx, prev_idx, prevprev_idx, node_pos):
    """
    Computes angle difference between the current edge up to some node and some edge from the current
    node to a potential successor node.
    :param fut_idx: index of potential successor node
    :param prev_idx: current node index
    :param prevprev_idx: predecessor node index
    :param node_pos: position features of graph nodes
    :return edge_diff: angle difference between the current edge and the edge candidate
    """
    edge_dx_fut = node_pos[fut_idx, 0] - node_pos[prev_idx, 0]
    edge_dy_fut = node_pos[fut_idx, 1] - node_pos[prev_idx, 1]
    edge_angle_fut = np.arctan2(edge_dy_fut, edge_dx_fut)

    edge_dx_past = node_pos[prev_idx, 0] - node_pos[prevprev_idx, 0]
    edge_dy_past = node_pos[prev_idx, 1] - node_pos[prevprev_idx, 1]
    edge_angle_past = np.arctan2(edge_dy_past, edge_dx_past)

    edge_diff = np.abs(edge_angle_fut - edge_angle_past)
    return edge_diff
