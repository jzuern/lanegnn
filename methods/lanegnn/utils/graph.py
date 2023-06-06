import numpy as np
import json
from collections import defaultdict
import networkx as nx
import cv2
from scipy.spatial.distance import cdist
from shapely.geometry import LineString, Point
import torch
from torch_geometric.utils import degree


def unbatch(src, batch, dim: int = 0):
    """
    Take the src value and separate it into multiple matrices based on the provided batch association.
    :param src: feature matrix to split
    :param batch: Batch vector associating each node to a distinct batch
    """
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)


def unbatch_edge_index(edge_index, batch):
    """
    Take the batch vector and produce multiple unconnected edge indices
    :param edge_index: Sparse definition of edges
    :param batch: Batch vector associating each node to a distinct batch
    """
    deg = degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1)


def laplacian_smoothing(G, gamma=0.5, iterations=1):
    """
    Takes input graph and applies multiple iterations of Laplacian smoothing weighted by gamma.
    Note that this function produces an undirected graph when computing the Laplacian.
    :param G: networkx graph to smooth
    :param gamma: smoothing intensity
    :param iterations: number of iterations to smooth for
    :return: smoothed graph
    """
    L = nx.laplacian_matrix(nx.Graph(G)).todense()
    O = np.eye(L.shape[0]) - gamma * L

    for it in range(iterations):

        node_pos = np.array([list(G.nodes[n]['pos']) for n in G.nodes()])
        node_pos = np.dot(O, node_pos)

        # Update node positions
        for i, node in enumerate(G.nodes()):
            #if G.degree(node) == 2:
            G.nodes[node]['pos'] = np.array(node_pos[i, :]).flatten()

    return G


def assign_edge_lengths(G):
    """
    Takes an input graph G and adds the edge length
    :param G: networkx.Graph
    """
    for u, v, d in G.edges(data=True):
        d['length'] = np.linalg.norm(np.array(G.nodes[u]['pos']) - np.array(G.nodes[v]['pos']))

    return G


def merge_common(lists):
    """
    # merge function to  merge all sublist having common elements.
    :param lists:
    :return:
    """
    neigh = defaultdict(set)
    visited = set()
    for each in lists:
        for item in each:
            neigh[item].update(each)

    def comp(node, neigh=neigh, visited=visited, vis=visited.add):
        nodes = set([node])
        next_node = nodes.pop
        while nodes:
            node = next_node()
            vis(node)
            nodes |= neigh[node] - visited
            yield node

    for node in neigh:
        if node not in visited:
            yield sorted(comp(node))


def get_supernodes(G, max_distance=0.1):
    """
    Compute graph supernodes
    :param networkx.Graph G: networkx.Graph
    :param float max_distance: distance which needs to be met
    :return list:
    """
    # Max distance is in unit pixel
    print("Getting supernodes for graph with {} nodes".format(G.number_of_nodes()))

    waypoints = nx.get_node_attributes(G, 'pos')
    nodes = np.array([waypoints[i] for i in G.nodes()])
    node_name_dict = {i: node_name for i, node_name in enumerate(G.nodes())}

    distance_matrix = cdist(nodes, nodes, metric='euclidean')
    close_indices = np.argwhere(distance_matrix < max_distance)

    # convert according to node_name_dict
    close_indices_ = []
    for i, j in close_indices:
        close_indices_.append([node_name_dict[i], node_name_dict[j]])
    close_indices = close_indices_

    close_indices_sets = list(merge_common(close_indices))

    print("Getting supernodes for graph with {} nodes... done! Found {} supernodes.".format(G.number_of_nodes(), len(close_indices_sets)))

    return close_indices_sets


def get_target_data(params, data, split):
    """
    Returns the target graph data for the given sample for the successor-LGP task.
    """
    tile_no = int(data.tile_no[0].cpu().detach().numpy())
    walk_no = int(data.walk_no[0].cpu().detach().numpy())
    idx = int(data.idx[0].cpu().detach().numpy())
    city = data.city[0]


    json_fname = "{}{}/{}/{}/{:03d}_{:03d}_{:03d}-targets.json".format(params.paths.dataroot, params.paths.rel_dataset, city, split, tile_no, walk_no, idx)
    with open(json_fname, 'r') as f:
        targets = json.load(f)

    targets['tile_no'] = tile_no
    targets['walk_no'] = walk_no
    targets['idx'] = idx

    return targets



def get_gt_graph(targets):
    """
    Produces a nx.DiGraph object from the ground truth target graph.
    """
    nodes = np.array(targets['bboxes'])
    edges = np.array(targets['relation_labels'])

    graph_gt = nx.DiGraph()

    # Populate graph with nodes
    for i, n in enumerate(nodes):
        graph_gt.add_node(i, pos=n, weight=1.0)
    for e in edges:
        graph_gt.add_edge(e[0], e[1])

    graph_gt.remove_edges_from(nx.selfloop_edges(graph_gt))

    return graph_gt


def transform_graph_to_pos_indexing(G):
    """
    Relabels node names based on their pixel-wise position in the image.
    :param G: Input graph
    :return: relabeled nx.DiGraph instance
    """
    G_ = nx.DiGraph()
    for n in G.nodes():
        pos = G.nodes[n]['pos']
        pos_int = (int(pos[0]), int(pos[1]))
        if 'weight' not in G.nodes[n]:
            G.nodes[n]['weight'] = 1.0
        G_.add_node(pos_int, pos=G.nodes[n]['pos'], weight=G.nodes[n]['weight'], score=G.nodes[n]['score'])
    for e in G.edges():
        pos_start = G.nodes[e[0]]['pos']
        pos_end = G.nodes[e[1]]['pos']
        pos_start_int = (int(pos_start[0]), int(pos_start[1]))
        pos_end_int = (int(pos_end[0]), int(pos_end[1]))
        G_.add_edge(pos_start_int, pos_end_int)

    return G_
