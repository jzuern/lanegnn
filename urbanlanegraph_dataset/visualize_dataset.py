import os
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import pickle
import argparse
from glob import glob
import cv2
import networkx as nx
import matplotlib.pyplot as plt


def visualize_graph(G, ax, node_color=np.array([255, 0, 142]) / 255., edge_color=np.array([255, 0, 142]) / 255.):

    '''
    Visualize a lane graph on an axis
    Args:
        G: graph
        ax:  axis object
        node_color:  color of nodes
        edge_color:  color of edges

    Returns:
        None
    '''

    nx.draw_networkx(G, ax=ax, pos=nx.get_node_attributes(G, "pos"),
                     edge_color=node_color,
                     node_color=edge_color,
                     with_labels=False,
                     node_size=5,
                     arrowsize=15.0, )


def main():
    parser = argparse.ArgumentParser(description="UrbanLaneGraph Visualizer")

    # General parameters (namespace: main)
    parser.add_argument('--dataset_root', type=str, help='Path to the UrbanLaneGraph dataset')
    parser.add_argument('--plot-single-tiles', action='store_true', help='Plot single tiles')
    parser.add_argument('--city', type=str, help='City to visualize',
                        choices=["Austin", "Detroit", "Miami", "PaloAlto", "Pittsburgh", "Washington"])
    opt = parser.parse_args()

    print("Looking for UrbanLaneGraph dataset in:", opt.dataset_root)

    # Due to the large size of the dataset, we visualize the graphs and write them to a file.

    # load annotation pickle files
    annotation_tile_names_train = glob(os.path.join(opt.dataset_root, opt.city, "tiles", "train", "*.gpickle"))
    annotation_tile_names_eval = glob(os.path.join(opt.dataset_root, opt.city, "tiles", "eval", "*.gpickle"))

    print("Found {} training tiles and {} evaluation tiles".format(len(annotation_tile_names_train), len(annotation_tile_names_eval)))

    # load images
    print("Loading aerial image for city:", opt.city)
    aerial_image_name = os.path.join(opt.dataset_root, opt.city, "{}.png".format(opt.city))
    aerial_image = np.asarray(Image.open(aerial_image_name))

    print("Loading direction image for city:", opt.city)
    direction_image_name = os.path.join(opt.dataset_root, opt.city, "{}_direction.png".format(opt.city))
    direction_image = np.asarray(Image.open(direction_image_name))

    # visualize graphs in aerial image
    for annotation_tile_name in annotation_tile_names_train + annotation_tile_names_eval:
        print("Processing tile:", annotation_tile_name)

        tile_id = int(annotation_tile_name.split("/")[-1].split("_")[1])

        G = pickle.load(open(annotation_tile_name, "rb"))

        if opt.plot_single_tiles:
            # Plot single tile
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.set_aspect('equal')
            visualize_graph(G, ax)
            plt.title("Tile {}".format(annotation_tile_name))
            plt.show()

        # Draw edges as arrows on aerial image
        line_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        for e in G.edges():
            start = G.nodes[e[0]]["pos"]
            end = G.nodes[e[1]]["pos"]
            cv2.arrowedLine(aerial_image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), line_color,
                            thickness=1, line_type=cv2.LINE_AA, tipLength=0.2)

        # Get limits of graph file in each dimension
        node_positions = np.array([G.nodes[n]["pos"] for n in G.nodes()])

        xmin = np.min(node_positions[:, 0])
        xmax = np.max(node_positions[:, 0])
        ymin = np.min(node_positions[:, 1])
        ymax = np.max(node_positions[:, 1])

        # Get center of graph file in each dimension
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2

        # Write tile on image
        font_scale = 5
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_type = cv2.LINE_AA
        text = "{:03d}".format(tile_id)
        text_size = cv2.getTextSize(text, font, font_scale, line_type)[0]
        text_x = x_center - text_size[0] / 2
        text_y = y_center - text_size[1] / 2
        cv2.putText(aerial_image, text, (int(text_x), int(text_y)), font, font_scale, line_color, line_type)


    # now merge two images to visualize both overlays
    viz = cv2.addWeighted(aerial_image, 0.7, direction_image, 0.3, 0)

    aerial_image_viz_name = aerial_image_name.replace(".png", "_viz.png")
    print("Writing visualization to:", aerial_image_viz_name)
    cv2.imwrite(aerial_image_viz_name, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))



if __name__ == '__main__':

    main()