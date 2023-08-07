from scipy.spatial import Delaunay
import networkx as nx
import scipy
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from urbanlanegraph_dataset.api import *
from methods.lanegnn.utils.graph import merge_common


def transform_coordinates_2D(pos, R):
    """
    Transform the coordinates of a 2D point with a rotation matrix.
    :param pos: 2D point
    :param R: rotation matrix
    :return: transformed 2D point
    """
    return np.dot(R, pos)


def redistribute_vertices(geom, distance):

    if geom.geom_type == 'LineString':
        num_vert = int(round(geom.length / distance))
        if num_vert == 0:
            num_vert = 1
        return LineString(
            [geom.interpolate(float(n) / num_vert, normalized=True)
             for n in range(num_vert + 1)])
    elif geom.geom_type == 'MultiLineString':
        parts = [redistribute_vertices(part, distance)
                 for part in geom]
        return type(geom)([p for p in parts if not p.is_empty])
    else:
        raise ValueError('unhandled geometry %s', (geom.geom_type,))


def filter_nodes_outside_area(node_pos, area):

    (i, j, ih, jw) = area
    # valid_boxes = np.zeros([len(obj_boxes)]).astype(np.uint8)

    crit1 = node_pos[:,0] > i
    crit3 = node_pos[:,0] < ih
    crit5 = node_pos[:,1] > j
    crit7 = node_pos[:,1] < jw

    valid_boxes = crit1 * crit3 * crit5 * crit7

    valid_boxes = valid_boxes.astype(np.uint8)

    return valid_boxes

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

    return R, c, t



def filter_roi(obj_boxes, obj_relation_triplets, i, j, roi_width, roi_height):

    valid_boxes = filter_nodes_outside_area(obj_boxes, (i, j, i + roi_width, j + roi_height)).astype(bool)
    obj_relation_triplets = np.array(obj_relation_triplets)

    valid_triplets = np.zeros([len(obj_relation_triplets)]).astype(np.uint8)
    for k, ids in enumerate(obj_relation_triplets):
        if valid_boxes[ids[0]] and valid_boxes[ids[1]]:
            valid_triplets[k] = 1

    # Remove the relevant relationships from  obj_relation_triplets_, obj_relations, and obj_boxes
    obj_boxes_ = obj_boxes[valid_boxes].copy()
    obj_relation_triplets_ = obj_relation_triplets[valid_triplets.astype(bool)].copy()

    # Also convert the old obj indices to the new shortened obj indices:
    new_indices = np.cumsum(valid_boxes)

    for k in range(len(obj_relation_triplets_)):
        idx = obj_relation_triplets_[k, 0]
        obj_relation_triplets_[k, 0] = new_indices[idx] - 1
        idx = obj_relation_triplets_[k, 1]
        obj_relation_triplets_[k, 1] = new_indices[idx] - 1

    return obj_boxes_, obj_relation_triplets_




def visualize_graph(G, successor_subgraph, agent_visited_nodes):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('equal')
    for p in G.edges():
        start = G.nodes[p[0]]["pos"]
        end = G.nodes[p[1]]["pos"]
        # plt.text(start[0], start[1], str(p[0]), fontsize=10)
        plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], head_width=10, head_length=10)
    for p in successor_subgraph.edges():
        start = G.nodes[p[0]]["pos"]
        end = G.nodes[p[1]]["pos"]
        plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], head_width=10, head_length=10, color='r')
    for n in agent_visited_nodes:
        box = G.nodes[n]["pos"]
        plt.plot(box[0], box[1], 'bo')

    print("Agent nodes: blue")
    print("Successor nodes: red")
    plt.show()


def filter_subgraph(G, subgraph, ego_node, max_distance=256):
    """
    Filters subgraph based on spatial distance to ego-node. If a node is removed, all downstream nodes are also removed.
    :param subgraph:
    :param ego_node:
    :return:
    """

    subgraph_from_ego = nx.bfs_tree(subgraph, ego_node)

    # Remove all nodes that are too far away from the ego-node
    subgraph_from_ego_ = subgraph_from_ego.copy()
    for n in subgraph_from_ego_.nodes():
        if np.linalg.norm(G.nodes[n]["pos"] - G.nodes[ego_node]["pos"]) > max_distance:
            subgraph_from_ego.remove_node(n)

    subgraph_from_ego_ = subgraph_from_ego.copy()
    # Remove all nodes without connection to ego_node
    for n in subgraph_from_ego_.nodes():
        if not nx.has_path(subgraph_from_ego, ego_node, n):
            subgraph_from_ego.remove_node(n)

    return subgraph_from_ego