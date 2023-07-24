import os
import pickle
import networkx as nx

from urbanlanegraph_evaluator.urbanlanegraph_evaluator.evaluator import GraphEvaluator
from urbanlanegraph_dataset.api import UrbanLaneGraphMetadata


# Define split and list of cities to evaluate
split = "test"  # either eval or test
city_names = [
    "austin",
    "detroit",
    "miami",
    "paloalto",
    "pittsburgh",
    "washington"
]

# Load GT graph
# TODO: Add dataset path here
ulg = UrbanLaneGraphMetadata(dataset_path='/ulg-dataset/')

graphs_gt = dict()
for city in city_names:
    graphs_gt[city] = dict()
    graphs_gt[city][split] = dict()
    for tile in ulg.get_tiles(split, city):
        graphs_gt[city][split][tile] = ulg.get_tile_graph(split, city, tile)


# Load pred graph pickle
# TODO: Add predicted graph pkl here
pred_graphs_dir = "/home/USER/predicted_graphs/"
graphs_pred_file = pred_graphs_dir + "predicted_graph.pkl"

with open(graphs_pred_file, 'rb') as f:
    graphs_pred = pickle.load(f)

# Define list of metrics to evaluate
metric_names = ["TOPO Precision",
                "TOPO Recall",
                "GEO Precision",
                "GEO Recall",
                "APLS",
                "SDA20",
                "SDA50",
                "Graph IoU"
                ]

metrics_all = {}


for city in city_names:
    metrics_all[city] = {}
    metrics_all[city][split] = {}

    for sample_id in graphs_gt[city][split]:
        metrics_all[city][split][sample_id] = {}

        if graphs_pred[city][split][sample_id] is None:
            print("No prediction for sample", sample_id)
            metrics_sample = {metric_name: 0.0 for metric_name in metric_names}
        else:
            evaluator = GraphEvaluator()

            import matplotlib.pyplot as plt

            fig, axarr = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
            axarr[0].set_title('Ground Truth')
            axarr[1].set_title('Prediction')

            # gt graph
            nx.draw_networkx(graphs_gt[city][split][sample_id],
                             ax=axarr[0],
                             pos=nx.get_node_attributes(graphs_gt[city][split][sample_id], 'pos'),
                             edge_color='r',
                             node_color='r',
                             with_labels=False,
                             width=2,
                             arrowsize=4,
                             node_size=10)
            # pred graph
            nx.draw_networkx(graphs_pred[city][split][sample_id],
                             ax=axarr[0],
                             pos=nx.get_node_attributes(graphs_pred[city][split][sample_id], 'pos'),
                             edge_color='b',
                             node_color='b',
                             with_labels=False,
                             width=2,
                             arrowsize=4,
                             node_size=10)

            plt.show()

            metrics = evaluator.evaluate_graph(graphs_gt[city][split][sample_id],
                                               graphs_pred[city][split][sample_id],
                                               area_size=[256, 256])

            metrics_sample = {
                "TOPO Precision": metrics['topo_precision'],
                "TOPO Recall": metrics['topo_recall'],
                "GEO Precision": metrics['topo_precision'],
                "GEO Recall": metrics['geo_recall'],
                "APLS": metrics['apls'],
                "SDA20": metrics['sda@20'],
                "SDA50": metrics['sda@50'],
                "Graph IoU": metrics['iou'],
            }

        metrics_all[city][split][sample_id].update(metrics_sample)