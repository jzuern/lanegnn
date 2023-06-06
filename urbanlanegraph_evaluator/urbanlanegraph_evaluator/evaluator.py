import numpy as np
import json
import networkx as nx
from .utils import assign_edge_lengths, find_closest_nodes
from .metrics.geotopo import Evaluator as GeoTopoEvaluator
from .metrics.metrics import calc_sda, calc_iou, calc_apls
from .metrics.metrics import nx_to_geo_topo_format
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
import random
import warnings
from urbanlanegraph_evaluator.utils import prepare_graph_apls

random.seed(0)

class GraphEvaluator():

    """
    Evaluator for graphs
    """

    def __init__(self):
        self.metrics = {}
        self.aerial_image = None


    def reset(self):
        '''
        Reset the evaluator
        Returns: None
        '''
        self.metrics = {}
        self.aerial_image = None

    def __str__(self):
        return str(self.metrics)

    def set_aerial_image(self, img):
        '''
        Set the aerial image to visualize the graphs on top of
        Args:
            img:

        Returns: None
        '''
        self.aerial_image = img


    def evaluate_graph(self, graph_gt: nx.DiGraph, graph_pred: nx.DiGraph, area_size, lane_width):

        graph_gt = assign_edge_lengths(graph_gt)
        graph_pred = assign_edge_lengths(graph_pred)

        sda_20 = calc_sda(graph_gt, graph_pred, threshold=20)
        sda_50 = calc_sda(graph_gt, graph_pred, threshold=50)

        iou = calc_iou(graph_gt, graph_pred, area_size=area_size, lane_width=lane_width)

        graph_gt_for_apls = prepare_graph_apls(graph_gt)
        graph_pred_for_apls = prepare_graph_apls(graph_pred)

        # Try to calculate APLS metric
        try:
            apls = calc_apls(graph_gt_for_apls, graph_pred_for_apls)
        except Exception as e:
            apls = 0
            warnings.warn("Error calculating APLS metric: {}.".format(e))

        # Try to calculate GEO and TOPO metrics
        try:
            graph_gt_ = nx_to_geo_topo_format(graph_gt)
            graph_pred_ = nx_to_geo_topo_format(graph_pred)
            geo_precision, geo_recall, topo_precision, topo_recall = GeoTopoEvaluator(graph_gt_, graph_pred_).topoMetric()

        except Exception as e:
            geo_precision = 0
            geo_recall = 0
            topo_precision = 0
            topo_recall = 0
            warnings.warn("Error calculating GEO and TOPO metrics: {}.".format(e))

        metrics_dict = {
            'iou': iou,
            'apls': apls,
            'geo_precision': geo_precision,
            'geo_recall': geo_recall,
            'topo_precision': topo_precision,
            'topo_recall': topo_recall,
            'sda@20': sda_20[2],
            'sda@50': sda_50[2],
        }

        return metrics_dict


    def evaluate_paths_all(self, gt_graphs, pred_graphs, ignore_nans=True):

        self.metrics_dict = []

        for g_gt, g_pred in zip(gt_graphs, pred_graphs):
            paths_gt, paths_pred = self.generate_paths_single(g_gt, g_pred)

            metrics = self.evaluate_paths(g_gt, g_pred, paths_gt, paths_pred)
            self.metrics_dict.append(metrics)


        # Parse list of dicts to dict
        self.metrics = {}
        for key in self.metrics_dicts[0].keys():
            if ignore_nans:
                self.metrics[key] = np.nanmean([d[key] for d in self.metrics_dicts])
            else:
                self.metrics[key] = np.mean([d[key] for d in self.metrics_dicts])

        return self.metrics



    def generate_paths(self, g_gt, g_pred, num_planning_paths=100):

        """
        Generate paths for a single graph pair
        Args:
            g_gt: ground truth graph
            g_pred: predicted graph
        """

        path_counter = 0

        paths_pred = []
        paths_gt = []


        for i in range(num_planning_paths):
            path_counter += 1

            # get random start node in g_agg
            start_node = random.choice(list(g_gt.nodes()))
            walk_length_max = 100

            # Generate Agent Trajectory
            curr_node = start_node
            for j in range(0, walk_length_max):
                successors = [n for n in g_gt.successors(curr_node)]
                if len(successors) == 0:
                    break
                curr_node = successors[np.random.randint(0, len(successors))]

            if j < 50:
                continue

            pos_start = g_gt.nodes[start_node]['pos']
            pos_end = g_gt.nodes[curr_node]['pos']

            # find closest node in g_pred to start and end
            nodes_start_pred, nodes_end_pred = find_closest_nodes(g_pred, pos_start, pos_end, n=5)
            nodes_start_gt, nodes_end_gt = find_closest_nodes(g_gt, pos_start, pos_end, n=5)

            path_pred = []
            path_gt = []

            found_pred = False
            found_gt = False

            # plan path
            for ns in nodes_start_gt:
                if found_gt:
                    break
                for ne in nodes_end_gt:
                    try:
                        path_gt = nx.shortest_path(g_gt, ns, ne)
                        found_gt = True
                        break
                    except Exception as e:
                        pass
            for ns in nodes_start_pred:
                if found_pred:
                    break
                for ne in nodes_end_pred:
                    try:
                        path_pred = nx.shortest_path(g_pred, ns, ne)
                        found_pred = True
                        break
                    except Exception as e:
                        pass

            if not found_gt:
                print("No path found in GT")
                paths_gt.append(None)
                paths_pred.append(None)
                continue

            if len(path_pred) < 2:
                print("No path found in Pred")
                paths_gt.append(path_gt)
                paths_pred.append(None)
            else:
                paths_gt.append(path_gt)
                paths_pred.append(path_pred)

        return paths_gt, paths_pred


    def evaluate_paths(self, g_gt, g_pred, paths_gt, paths_pred):
        """
        Evaluate paths
        """

        metrics_list = []
        successes = 0

        for path_gt, path_pred in zip(paths_gt, paths_pred):
            if path_gt is not None:
                if path_pred is not None:
                    metrics_sample = self.get_path_metrics(path_gt, path_pred, g_gt, g_pred)
                    metrics_list.append(metrics_sample)
                    successes += 1
                else:
                    metrics_list.append({'mmd': 5000.,
                                         'med': 5000.,
                                         })
                    successes += 0

        success_rate = float(successes) / len(paths_gt)

        # summarize metrics_list into a single dict
        metrics = {}
        for key in metrics_list[0].keys():
            metrics[key] = np.mean([d[key] for d in metrics_list])

        metrics['sr'] = success_rate

        return metrics


    def get_path_metrics(self, path_gt, path_pred, g_gt, g_pred):

        path_gt = [nx.get_node_attributes(g_gt, 'pos')[n] for n in path_gt]
        path_pred = [nx.get_node_attributes(g_pred, 'pos')[n] for n in path_pred]

        # resample both paths to the same number of points
        n = 100
        path_gt = np.array(path_gt)
        path_pred = np.array(path_pred)

        path_gt = interp1d(np.arange(len(path_gt)), path_gt, axis=0)(
            np.linspace(0, len(path_gt) - 1, n))
        path_pred = interp1d(np.arange(len(path_pred)), path_pred, axis=0)(
            np.linspace(0, len(path_pred) - 1, n))

        avg_min_path_distance = np.mean(np.min(cdist(path_gt, path_pred), axis=1))
        endpoint_distance = np.linalg.norm(path_gt[-1] - path_pred[-1])

        length_gt = np.sum(np.linalg.norm(path_gt[1:] - path_gt[:-1], axis=1))

        # compute metrics
        metrics = {
            'mmd': avg_min_path_distance,
            'med': endpoint_distance
        }

        return metrics

    def visualize_graph(self, G, ax, aerial_image=True, node_color='white', edge_color='white'):

        if self.aerial_image is not None and aerial_image:
            ax.imshow(self.aerial_image)

        nx.draw_networkx(G,
                         ax=ax,
                         pos=nx.get_node_attributes(G, 'pos'),
                         edge_color=edge_color,
                         node_color=node_color,
                         with_labels=False,
                         width=2,
                         arrowsize=4,
                         node_size=20)


    def visualize_paths(self, ax, g, path, color):

        for n in path:
            if path.index(n) > len(path) - 2:
                break
            ax.arrow(g.nodes[n]['pos'][0],
                     g.nodes[n]['pos'][1],
                     g.nodes[path[(path.index(n) + 1) % len(path)]]['pos'][0] - g.nodes[n]['pos'][0],
                     g.nodes[path[(path.index(n) + 1) % len(path)]]['pos'][1] - g.nodes[n]['pos'][1],
                     color=color,
                     width=4,
                     length_includes_head=True,
                     head_width=6,
                     head_length=6,
                     )


    def export_results(self, fname):
        with open(fname, 'w') as f:
            json.dump(self.metrics, f, indent=4)


