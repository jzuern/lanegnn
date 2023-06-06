import argparse
import os
import numpy as np
from collections import defaultdict
import networkx as nx
import pickle
import glob

from lanegnn.utils.params import ParamLib
from dataset_preparation.settings import UrbanLaneGraphMetadata

def produce_result_dict(params, experiment_name, split):
    """
    Loads the graphs of all tiles and add it to the result dict
    :param params: parameter config
    :param experiment_name: name of the experiment
    :param split: split to be evaluated
    """
    ulg = UrbanLaneGraphMetadata(dataset_path=params.driving.ulg_path)

    result_dict = dict()
    for city in ulg.get_cities():
        if split in ulg.get_splits():
            result_dict[city] = dict()
            result_dict[city][split] = dict()
            for tile_no in ulg.get_tiles(split, city):
                    try:
                        result_dict = compile_tile(params,
                                                   ulg,
                                                   params.driving.mode,
                                                   split,
                                                   city,
                                                   tile_no,
                                                   experiment_name,
                                                   result_dict,
                                                   global_offset=True,
                                                   drive_padding=True)
                    except:
                        print("Error in city: ", city, " split: ", split, " tile: ", tile_no)
                        continue
    return result_dict


def compile_tile(params, ulg, mode, split, city, tile_no, experiment, result_dict, global_offset=False, drive_padding=False):
    """
    Loads the graph of the respective tile and add it to the result dict
    """
    print("Compiling city: ", city, " split: ", split, " tile: ", tile_no)
    graph_files = glob.glob(os.path.join(params.driving.output_dir, mode, city, str(tile_no), experiment, 'full_graph', '*.pkl'))
    final_agg_graph_file = sorted(graph_files)[-1]
    print("------  Loading graph from: ", final_agg_graph_file)

    with open(final_agg_graph_file, 'rb') as f:
        graph_data_dict = pickle.load(f)

    G_agg = graph_data_dict['G_agg']

    # APPLY DRIVE-INDUCED TRANSLATION (GLOBAL TRAFO + DRIVE PADDING)
    if global_offset or drive_padding:
        # Default offset values
        tile_offset = np.array([0, 0])
        pad_offset = 0

        if global_offset:
            tile_offset = np.array(ulg.get_tile_offset(split, city, tile_no))
        if drive_padding:
            pad_offset = np.array([512, 512])  # symmetric padding

        for node in G_agg.nodes:
            G_agg.nodes[node]["pos"] += tile_offset - pad_offset

    result_dict[city][split][tile_no] = G_agg
    return result_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DRIVE")
    # General parameters (namespace: main)
    parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)

    opt = parser.parse_args()
    params = ParamLib(opt.config)
    params.main.overwrite(opt)
    params.driving.overwrite(opt)

    # TODO: Define split and experiment name and load the result dict
    split = "test"
    experiment_name = "EXP_NAME"

    result_dict = produce_result_dict(params, experiment_name, split)

    # Save the result dict
    path = os.path.join(params.driving.output_dir, "results/")
    if not os.path.exists(path):
        os.makedirs(path)
    print(path + experiment_name + '.pkl')
    with open(path + experiment_name + '.pkl', 'wb') as f:
        pickle.dump(result_dict, f)

