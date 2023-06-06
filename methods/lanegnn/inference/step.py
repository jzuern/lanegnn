import numpy as np
import networkx as nx

def step_forward(ego_x_y_yaw, graph, visited_edges):
    """
    Stepping function, which takes the currently predicted graph, looks for the closest node in the graph
    given the current ego pose. Then, it retrieves potential successor nodes and selects an edge to traverse.
    In case of multiple successor paths, it selects the successor with the highest successor-tree weight.
    :param ego_x_y_yaw: current ego pose
    :param graph: currently aggregated nx.DiGraph instance that is to be traversed
    :param visited_edges: list of already visited edges
    :return: next_edge: Selected edge that is traversed next
    :return next_pose: Next pose after traversal
    :return edge_vector: Vector that depicts the traversal
    :return branch_alive: provides info whether a graph died in case no successor edge is found or still alive
    :return visited_edges: list of already visited edges in the graph
    """
    # Dummy values in case we immediately return
    next_pose = None
    edge_vector = None
    next_edge = None

    curr_x = ego_x_y_yaw[0]
    curr_y = ego_x_y_yaw[1]

    # Get nearest agg-node in wrt to current ego_pose
    closest_dist = float("inf")
    closest_node = None
    for n in graph.nodes():
        dist = np.linalg.norm(np.array([curr_x, curr_y]) - np.array(graph.nodes[n]['pos']))
        if dist < closest_dist: # and yaw_dist < 0.6:
            closest_dist = dist
            closest_node = n

    # Get all successors of found agg node
    successors = list(graph.successors(closest_node))
    branch_alive = False

    if len(successors) == 0:
        print("     Could not find next edge.")
        branch_alive = False

    elif len(successors) == 1:
        print("     One single next edge found.")
        # Select the only possible edge and obtain its position and yaw angle
        next_edge = (closest_node, successors[0])
        start = np.array(graph.nodes[next_edge[0]]['pos'])
        end = np.array(graph.nodes[next_edge[1]]['pos'])
        visited_edges.append(next_edge)
        branch_alive = True

        edge_vector = end - start
        x = end[0]
        y = end[1]
        yaw = np.arctan2(edge_vector[1], edge_vector[0])
        next_pose = np.array([x, y, yaw])
        # print(len(successors))

    elif len(successors) > 1:
        print("     Multiple next edges found. Selecting edge with highest weight and prio-queuing remaining edges.")

        # Select edge with the highest successor tree weight
        successors = list(graph.successors(closest_node))
        successor_trees = [list(nx.dfs_tree(graph, source=s, depth_limit=10)) for s in successors]
        successor_tree_weights = [np.sum([graph.nodes[k]['weight'] for k in t]) for t in successor_trees]
        successor_tree_weight_order = np.argsort(successor_tree_weights)[::-1]

        # Go through all available edges that are still traversable in a high to low weight order
        next_edge = None
        for _, order_idx in enumerate(successor_tree_weight_order):
            # Take first promising edge that is not yet visited
            if (closest_node, successors[successor_tree_weight_order[order_idx]]) not in visited_edges:
                next_edge = (closest_node, successors[order_idx])
                visited_edges.append(next_edge)
                branch_alive = True
                break

        if next_edge is not None:
            start = np.array(graph.nodes[next_edge[0]]['pos'])
            end = np.array(graph.nodes[next_edge[1]]['pos'])

            edge_vector = end - start
            x = end[0]
            y = end[1]
            yaw = np.arctan2(edge_vector[1], edge_vector[0])
            next_pose = np.array([x, y, yaw])
        else:
            print("     Even though multiple options were available, did not find edge?")
            branch_alive = False

    # # Render splitting point
    # cv2.circle(satellite_image_viz, (int(start[0]), int(start[1])), 8, (0, 0, 255), -1)
    # print(next_pose, edge_vector, branch_alive)
    #except:
    #    print("     Could not find next edge.")
    #    branch_alive = False

    return next_edge, next_pose, edge_vector, branch_alive, visited_edges