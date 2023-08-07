from abc import ABC
import numpy as np
import torch
import os
from glob import glob
import cv2
import codecs
import json
import time
import torch_geometric.data.dataset


class PreGraphDataset(torch_geometric.data.Dataset, ABC):
    """
    Defines dataset when using preprocessed data during LaneGNN training.
    Segmentation, graph sampling and ground truth generation already done in preprocessing.
    """
    def __init__(self, params, path, visualize: bool = False, index_filter=None):
        super(PreGraphDataset, self).__init__()
        """
        Based on the provided cities and an existing preprocessed set of samples for the Successor-LGP task, 
        this class compiles the dataset used during LaneGNN training.
        :param params: parameter configuration
        :param path: global path that holds preprocessed dataset files.
        :param visualize: Show RGB, ego- and context-regression as well as sampled edges for debugging.
        :param index_filter: Use index-filter to obtain a dummy dataset of cities and tiles satisfying the index_filter.
        """

        self.visualize = visualize
        self.index_filter = index_filter

        self.node_feats_files = []
        self.edge_files = []
        self.edge_attr_files = []
        self.edge_img_feats_files = []
        self.node_gt_files = []
        self.node_endpoint_gt_files = []
        self.edge_gt_files = []
        self.edge_gt_onehot_files = []
        self.gt_graph_files = []
        self.rgb_files = []
        self.rgb_context_files = []
        self.context_regr_smooth_files = []
        self.ego_regr_smooth_files = []


        self.node_feats_files.extend(glob(path + '/*-node-feats.pth'))
        self.edge_files.extend(glob(path + '/*-edges.pth'))
        self.edge_attr_files.extend(glob(path + '/*-edge-attr.pth'))
        self.edge_img_feats_files.extend(glob(path + '/*-edge-img-feats.pth'))
        self.node_gt_files.extend(glob(path + '/*-node-gt.pth'))
        self.node_endpoint_gt_files.extend(glob(path + '/*-node-endpoint-gt.pth'))
        self.edge_gt_files.extend(glob(path + '/*-edge-gt.pth'))
        self.edge_gt_onehot_files.extend(glob(path + '/*-edge-gt-onehot.pth'))
        self.gt_graph_files.extend(glob(path + '/*-gt-graph.pth'))
        self.rgb_files.extend(glob(path + '/*-rgb.pth'))
        self.rgb_context_files.extend(glob(path + '/*-rgb-context.pth'))
        self.context_regr_smooth_files.extend(glob(path + '/*-context-regr-smooth.pth'))
        self.ego_regr_smooth_files.extend(glob(path + '/*-ego-regr-smooth.pth'))

        self.node_feats_files = sorted(self.node_feats_files)
        self.edge_files = sorted(self.edge_files)
        self.edge_attr_files = sorted(self.edge_attr_files)
        self.edge_img_feats_files = sorted(self.edge_img_feats_files)
        self.node_gt_files = sorted(self.node_gt_files)
        self.node_endpoint_gt_files = sorted(self.node_endpoint_gt_files)
        self.edge_gt_files = sorted(self.edge_gt_files)
        self.edge_gt_onehot_files = sorted(self.edge_gt_onehot_files)
        self.gt_graph_files = sorted(self.gt_graph_files)
        self.rgb_files = sorted(self.rgb_files)
        self.rgb_context_files = sorted(self.rgb_context_files)
        self.context_regr_smooth_files = sorted(self.context_regr_smooth_files)
        self.ego_regr_smooth_files = sorted(self.ego_regr_smooth_files)

        # print(len(self.node_feats_files))
        # print(len(self.edge_files))
        # print(len(self.edge_attr_files))
        # print(len(self.edge_img_feats_files))
        # print(len(self.node_gt_files))
        # print(len(self.node_endpoint_gt_files))
        # print(len(self.edge_gt_files))
        # print(len(self.edge_gt_onehot_files))
        # print(len(self.gt_graph_files))
        # print(len(self.rgb_files))
        # print(len(self.rgb_context_files))
        # print(len(self.context_regr_smooth_files))
        # print(len(self.ego_regr_smooth_files))
        print("Found {} samples in path {}".format(len(self.rgb_files), path))


    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, index):

        # Obtain token identifying the sample
        tokens = self.rgb_files[index].split("/")[-1].split("_")
        city = tokens[0]
        tile_id = "_".join(tokens[1:4])
        walk_no = int(tokens[4])
        idx = int(tokens[5])

        if self.index_filter is not None:
            # Return reduced data object if the index is contained in index_filter.
            if self.index_filter[index] == 0:
                return torch_geometric.data.Data(tile_id=torch.tensor(tile_id),
                                                 city=torch.tensor(city),
                                                 walk_no=torch.tensor(walk_no),
                                                 idx=torch.tensor(idx)
                                                 )

        # Also pass the time it takes to load the data.
        start_time = time.time()

        node_feats = torch.load(self.node_feats_files[index])
        edges = torch.load(self.edge_files[index])
        edge_attr = torch.load(self.edge_attr_files[index])
        edge_img_feats = torch.load(self.edge_img_feats_files[index]).to(torch.float32) / 255.0 # cast uint8 to float32
        node_gt = torch.load(self.node_gt_files[index])
        node_endpoint_gt = torch.load(self.node_endpoint_gt_files[index]).float()
        edge_gt = torch.load(self.edge_gt_files[index])
        edge_gt_onehot = torch.load(self.edge_gt_onehot_files[index])
        gt_graph = torch.load(self.gt_graph_files[index])
        rgb = torch.load(self.rgb_files[index])
        rgb_context = torch.load(self.rgb_context_files[index])
        context_regr_smooth = torch.load(self.context_regr_smooth_files[index])
        ego_regr_smooth = torch.load(self.ego_regr_smooth_files[index])


        # For debugging RGB, context- and ego-regression as well as sampled edges.
        if self.visualize:
            import matplotlib.pyplot as plt
            # bin ego_regr_smooth into 10 bins
            rgb = cv2.resize(rgb.numpy()/255, (512, 512))
            ego_regr_smooth = cv2.resize(ego_regr_smooth, (512, 512))
            ego_regr_smooth = np.round(ego_regr_smooth * 6) / 6

            plt.imshow(rgb)
            plt.imshow(ego_regr_smooth, cmap='jet', alpha=0.5)
            plt.show()

            fig, axarr = plt.subplots(1, 5)
            plt.tight_layout()
            [r.imshow(rgb.numpy()/255) for r in axarr]
            axarr[1].imshow(context_regr_smooth[128:128+256, 128:128+256], cmap='jet', alpha=0.5)
            axarr[2].imshow(ego_regr_smooth, cmap='jet', alpha=0.5)
            axarr[3].scatter(node_feats[:, 1].numpy(), node_feats[:, 0].numpy(), color=[0.1, 0.8, 0.8])
            for e in edges:
                if np.linalg.norm(node_feats[e[0]] - node_feats[e[1]]) < 50:
                    axarr[4].plot([node_feats[e[0]][1], node_feats[e[1]][1]], [node_feats[e[0]][0], node_feats[e[1]][0]], color=[0.1, 0.8, 0.8, 0.2])
                    #axarr[3].plot(node_feats[e, 1].numpy(), node_feats[e, 0].numpy(), color=[0.1, 0.8, 0.8, 0.2])
            plt.show()

        data = torch_geometric.data.Data(x=node_feats,
                                         edge_index=edges.t().contiguous(),
                                         edge_attr=edge_attr,
                                         edge_img_feats=edge_img_feats,
                                         node_gt=node_gt.t().contiguous(),
                                         node_endpoint_gt=node_endpoint_gt.t().contiguous(),
                                         edge_gt=edge_gt.t().contiguous(),
                                         edge_gt_onehot=edge_gt_onehot.t().contiguous(),
                                         gt_graph=gt_graph,
                                         num_nodes=node_feats.shape[0],
                                         batch_idx=torch.tensor(len(gt_graph)),
                                         rgb=torch.FloatTensor(rgb / 255.), # [0.0, 1.0]
                                         rgb_context=torch.FloatTensor(rgb_context / 255.), # [0.0, 1.0]
                                         context_regr_smooth=torch.FloatTensor(context_regr_smooth), # [0.0, 1.0]
                                         ego_regr_smooth=torch.FloatTensor(ego_regr_smooth), # [0.0, 1.0]
                                         data_time=torch.tensor(time.time() - start_time), # Time used to load the data
                                         tile_id=tile_id,
                                         walk_no=torch.tensor(walk_no),
                                         idx=torch.tensor(idx),
                                         city=city
                                         )

        return data


def get_next_k_waypoints(json_file, k=1):
    """
    Returns the next k nodes in the ground truth graph provided by a JSON file.
    :param json_file: JSON holding ground truth graph description
    :param k: number of succeeding nodes to extract starting from a node close to the ego-pos (bottom-middle).
    """
    # We must find the ego-position in graph (very close to 128,255)
    #
    # ego_position_index = np.argmin(np.sum((coordinates_anchors - ego_position)**2, axis=1))

    graph = json.loads(codecs.open(json_file, 'r', encoding='utf-8').read())

    waypoints = np.array(graph["bboxes"])
    relation_labels = np.array(graph["relation_labels"])

    ego_index = 0

    # get list of waypoints connected to ego-position

    # Sort waypoints based on distance to ego-position index
    ego_pos = np.array([128, 255])
    distances = np.sum((waypoints - ego_pos)**2, axis=1)
    sorted_indices = np.argsort(distances)
    waypoints = waypoints[sorted_indices]

    # Convert everything into a torch tensor
    waypoints = torch.as_tensor(waypoints[0:k], dtype=torch.float32)

    waypoints_limits = [torch.max(torch.tensor(0.0), torch.min(waypoints[:, 0]-5)),  # xmin
                        torch.max(torch.tensor(0.0), torch.min(waypoints[:, 1]-5)),  # ymin
                        torch.min(torch.tensor(255.0), torch.max(waypoints[:, 0]+5)),  # xmax
                        torch.min(torch.tensor(255.0), torch.max(waypoints[:, 1]+5))]  # ymax

    waypoints = torch.cat((waypoints, torch.ones(waypoints.shape[0], 1)), dim=1)

    n_boxes = 1

    target = {}

    target["boxes"] = torch.tensor([waypoints_limits], dtype=torch.float32)
    target["labels"] = torch.as_tensor([1 for _ in range(n_boxes)], dtype=torch.int64)  # all objects are glue tubes
    target["keypoints"] = waypoints.unsqueeze(0)

    return target


if __name__ == '__main__':
    from lanegnn.utils.params import ParamLib

    params = ParamLib('config.yaml')
    train_path = os.path.join(params.paths.dataroot, params.paths.rel_dataset, "preprocessed", "train", params.paths.config_name)

    ds = PreGraphDataset(params, path=train_path,  visualize=params.preprocessing.visualize, city='mia')

    for i in range(len(ds)):
        data = ds[i]
        print(data)

