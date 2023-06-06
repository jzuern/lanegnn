import networkx as nx
import numpy as np
import pickle
from scipy.spatial.distance import cdist
from shapely.geometry import LineString
import uuid


def load_graphs(pkl_file_path):
    with open(pkl_file_path, 'rb') as f:
        graphs = pickle.load(f)
    return graphs

def adjust_node_positions(G, x_offset=0, y_offset=0):
    for n in G.nodes:
        G.nodes[n]['pos'][0] -= x_offset
        G.nodes[n]['pos'][1] -= y_offset
    return G


def normalize_graph_pos(graph_gt, graph_pred, area_dim=512, line_area_factor=0.008):
        """
        Takes lane graphs formulated in Euclidean space and represents them in positive-valued integer image space.
        The image manifold is bounded in (0,area_dim). Based on the extent of the lanegraph, area_dim and line_area_factor
        may need tuning.
        :param graph_gt: networkx DiGraph ground truth graph with pos-attribute set.
        :param graph_pred: networkx DiGraph ground truth graph with pos-attribute set.
        :param area_dim: Size of the image manifold the Euclidean graph is transformed to.
        :param line_area_factor: Drawn line width of a lane wrt. the image size, i.e., line_width = area_dim * line_area_factor.
        """
        # obtain min (negative) x and y value of GT graph to shift all nodes to positive values
        min_gt_x = min([graph_gt.nodes[node]['pos'][0] for node in graph_gt.nodes])
        min_gt_y = min([graph_gt.nodes[node]['pos'][1] for node in graph_gt.nodes])

        line_width = line_area_factor * area_dim

        # shift all nodes to positive values
        for node in graph_gt.nodes:
            graph_gt.nodes[node]['pos'] = (graph_gt.nodes[node]['pos'][0] + abs(min_gt_x) + line_width, graph_gt.nodes[node]['pos'][1] + abs(min_gt_y) + 2 * line_width)

        for node in graph_pred.nodes:
            graph_pred.nodes[node]['pos'] = (graph_pred.nodes[node]['pos'][0] + abs(min_gt_x) + line_width, graph_pred.nodes[node]['pos'][1] + abs(min_gt_y) + 2 * line_width)

        max_gt_x = max([graph_gt.nodes[node]['pos'][0] for node in graph_gt.nodes]) + line_width
        max_gt_y = max([graph_gt.nodes[node]['pos'][1] for node in graph_gt.nodes]) + line_width

        max_size_input = max(max_gt_y, max_gt_x)
        scale_factor = area_dim / max_size_input

        # scale all nodes to area_dim
        for node in graph_gt.nodes:
            graph_gt.nodes[node]['pos'] = (graph_gt.nodes[node]['pos'][0] * scale_factor, graph_gt.nodes[node]['pos'][1] * scale_factor)

        for node in graph_pred.nodes:
            graph_pred.nodes[node]['pos'] = (graph_pred.nodes[node]['pos'][0] * scale_factor, graph_pred.nodes[node]['pos'][1] * scale_factor)

        return graph_gt, graph_pred


def generate_random_graph(area_size=[256, 256]):

    # T intersection graph
    n_interp = 5

    lane_0 = np.linspace((0.1, 0.5), (0.4, 0.5), n_interp)
    lane_1 = np.linspace((0.5, 0.9), (0.5, 0.1), n_interp)

    # add some noise
    lane_0 += np.random.normal(loc=0, scale=0.01, size=lane_0.shape)
    lane_1 += np.random.normal(loc=0, scale=0.01, size=lane_1.shape)

    lane_0 = (lane_0 * area_size[0]).astype(np.int32)
    lane_1 = (lane_1 * area_size[1]).astype(np.int32)

    graph = nx.DiGraph()
    for i, c in enumerate(lane_0):
        graph.add_node(i, pos=c)
    for i, c in enumerate(lane_1):
        graph.add_node(i+n_interp, pos=c)

    for i in range(len(lane_0) - 1):
        graph.add_edge(i, i + 1)
    for i in range(len(lane_1) - 1):
        graph.add_edge(i+n_interp, i + 1+n_interp)

    # connect two lanes
    graph.add_edge(n_interp-1, n_interp + n_interp//2)

    return graph


def assign_edge_lengths(G):
    for u, v, d in G.edges(data=True):
        d['length'] = np.linalg.norm(np.array(G.nodes[u]['pos']) - np.array(G.nodes[v]['pos']))
    return G


def scores_to_nx_graph(node_scores, edge_scores, node_threshold=0.5, edge_threshold=0.5):
    '''
    Convert node and edge scores to a networkx graph.

    Args:
        node_scores: np.array of shape (n_nodes, n_classes)
        edge_scores: np.array of shape (n_edges, n_classes)
        node_threshold: float
        edge_threshold: float

    Returns:
        nx.Graph
    '''

    n_nodes = node_scores.shape[0]
    n_edges = edge_scores.shape[0]

    G = nx.Graph()

    # add nodes
    for i in range(n_nodes):
        G.add_node(i)

    # add edges
    for i in range(n_edges):
        G.add_edge(edge_scores[i, 0], edge_scores[i, 1])

    # add node attributes
    for i in range(n_nodes):
        for j in range(node_scores.shape[1]):
            G.nodes[i][j] = node_scores[i, j]

    # add edge attributes
    for i in range(n_edges):
        for j in range(edge_scores.shape[1]):
            G.edges[edge_scores[i, 0], edge_scores[i, 1]][j] = edge_scores[i, j]

    # remove nodes with low score
    for i in range(n_nodes):
        if node_scores[i, 0] < node_threshold:
            G.remove_node(i)

    # remove edges with low score
    for i in range(n_edges):
        if edge_scores[i, 0] < edge_threshold:
            G.remove_edge(edge_scores[i, 0], edge_scores[i, 1])

    return G


def find_closest_nodes(g, pos_start, pos_end, n=1):
    node_names = list(g.nodes())

    node_pos = np.array([g.nodes[node]['pos'] for node in g.nodes])
    pos_start = np.reshape(pos_start, (1, 2))
    pos_end = np.reshape(pos_end, (1, 2))


    dist_matrix_start = cdist(node_pos, pos_start)
    dist_matrix_end = cdist(node_pos, pos_end)

    closest_start_nodes = np.argsort(dist_matrix_start, axis=0)[:n]
    closest_end_nodes = np.argsort(dist_matrix_end, axis=0)[:n]
    closest_start_nodes = np.reshape(closest_start_nodes, (-1))
    closest_end_nodes = np.reshape(closest_end_nodes, (-1))

    closest_start_nodes = [node_names[i] for i in closest_start_nodes]
    closest_end_nodes = [node_names[i] for i in closest_end_nodes]

    return closest_start_nodes, closest_end_nodes


def truncated_uuid4():
    return str(int(uuid.uuid4()))[0:6]


def prepare_graph_apls(g):

    # relabel nodes with their node position and a random string to avoid name collisions between graphs
    g = nx.relabel_nodes(g, {n: str(g.nodes[n]['pos']) + truncated_uuid4() for n in g.nodes()})

    g = nx.to_undirected(g)

    # add x,y coordinates to graph properties
    for n, d in g.nodes(data=True):
        d['x'] = d['pos'][0] * 0.15  # scale from pixels to meters
        d['y'] = d['pos'][1] * 0.15  # scale from pixels to meters

    # add length to graph properties
    for u, v, d in g.edges(data=True):
        d['geometry'] = LineString([(g.nodes[u]['x'], g.nodes[u]['y']),
                                    (g.nodes[v]['x'], g.nodes[v]['y'])])
        d['length'] = d['geometry'].length

    return g
