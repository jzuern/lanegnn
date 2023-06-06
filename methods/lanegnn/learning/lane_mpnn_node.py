import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal
from torchvision.models import resnext50_32x4d
from torchvision.models import resnet18
import torchvision.transforms
import torchvision.utils
from typing import List, Optional, Set

import torch_geometric.nn
from torch_geometric.typing import Adj, Size
from torch_sparse import SparseTensor
from torch_scatter import gather_csr, scatter, segment_csr

from lane_mpnn import CausalMessagePassing
from lane_mpnn import get_map_encoder, get_large_map_encoder


class LaneGNN(torch.nn.Module):
    def __init__(self, gnn_depth, edge_geo_dim, map_feat_dim, edge_dim, node_dim, msg_dim, in_channels=3, ego_regressor=None, context_regressor=None):
        """
        Places aerial BEV features as node features instead of edge features as described in the ablations of the paper.
        :param gnn_depth: number of message passing steps
        :param edge_geo_dim: dim of geometric edge features
        :param map_feat_dim: dim of BEV features attended onto
        :param edge_dim: dim of edge features when passing messages
        :param node_dim: dim of node features when passing messages
        :param msg_dim: dim of messages during message passing
        :param in_channels: number of image channels (mostly 4)
        """
        super(LaneGNN, self).__init__()
        print("LaneGNN using aerial node features and causal message passing.")

        self.edge_geo_dim = edge_geo_dim
        self.depth = gnn_depth

        self.edge_encoder = nn.Sequential(
            nn.Linear(4, int(edge_dim/2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(edge_dim/2), edge_dim),
        )
        
        self.map_encoder = get_map_encoder(out_features=map_feat_dim, in_channels=in_channels) # default: 64

        self.fuse_node = nn.Sequential(
            nn.Linear(map_feat_dim+node_dim, int(map_feat_dim/2)+node_dim),
            nn.ReLU(inplace=True),
            nn.Linear(int(map_feat_dim/2)+node_dim, node_dim),
        )

        self.pose_encoder = nn.Sequential(
            nn.Linear(2, int(node_dim/2)),
            nn.ReLU(),
            nn.Linear(int(node_dim/2), node_dim)
        )

        self.node_classifier = nn.Sequential(
            nn.Linear(node_dim, int(node_dim/2)),
            nn.ReLU(),
            nn.Linear(int(node_dim/2), int(node_dim/4)),
            nn.ReLU(),
            nn.Linear(int(node_dim/4), 1),
        )

        self.endpoint_classifier = nn.Sequential(
            nn.Linear(node_dim, int(node_dim/2)),
            nn.ReLU(),
            nn.Linear(int(node_dim/2), int(node_dim/4)),
            nn.ReLU(),
            nn.Linear(int(node_dim/4), 1),
        )

        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_dim, int(edge_dim/2)),
            nn.ReLU(),
            nn.Linear(int(edge_dim/2), int(edge_dim/4)),
            nn.ReLU(),
            nn.Linear(int(edge_dim/4), 1),
        )

        self.message_passing = CausalMessagePassing(node_dim=node_dim, edge_dim=edge_dim, msg_dim=msg_dim)
 
    def forward(self, data):

        node_feats, edge_attr, edge_index, node_gt, edge_gt, rgb_context, context_regr, batch = (
            data.x,
            data.edge_attr,
            data.edge_index,
            data.node_gt,
            data.edge_gt,
            data.rgb_context,
            data.context_regr_smooth,
            data.batch_idx
        )

        x = self.pose_encoder(node_feats.float()) # N x D
        initial_x = x

        rgb_context = rgb_context.reshape(-1, 3, 512, 512)
        context_regr = context_regr.reshape(-1, 1, 512, 512)

        map_input = torch.cat([rgb_context, context_regr], dim=1) # B x C X H X W
        node_map_feats = torch.zeros((node_feats.shape[0], 4, 96, 96)).cuda()


        # Check map_input data
        """
        fig, axarr = plt.subplots(1, 2)

        ex_idx = 123
        example = torchvision.transforms.functional.crop(img=map_input[data.batch[ex_idx]],
                                               top=node_feats[ex_idx, 1]+128-48,
                                               left=node_feats[ex_idx, 0]+128-48,
                                               height=96, width=96)
        torchvision.utils.save_image(example.cpu(), 'example_torch.png')

        axarr[0].imshow(rgb_context[data.batch[ex_idx]].permute(1, 2, 0).cpu().numpy())
        axarr[0].scatter(node_feats[ex_idx, 0].item()+128, node_feats[ex_idx, 1].item()+128, c='r', s=10)
        plot_example = example[0:3,:,:].permute(1, 2, 0) * 255.0
        print(torch.min(plot_example), torch.max(plot_example))
        axarr[1].imshow(plot_example.cpu().numpy())
        plt.show()
        plt.savefig('example.png')
        """

        for i in range(node_feats.shape[0]):
            node_map_feats[i] = torchvision.transforms.functional.crop(img=map_input[data.batch[i]],
                                                       top=node_feats[i, 1]+128-48,
                                                       left=node_feats[i, 0]+128-48,
                                                       height=96, width=96)



        edge_attr = self.edge_encoder(edge_attr.float()) # E x D_E1
        node_map_attr = self.map_encoder(node_map_feats) # N x 64

        x = torch.cat([x, node_map_attr], dim=1) # N x (D_E1 + 64)
        x = self.fuse_node(x)

 
        for i in range(self.depth):
            x, edge_attr = self.message_passing.forward(x=x, edge_index=edge_index, edge_attr=edge_attr, initial_x=initial_x)
            
        return self.edge_classifier(edge_attr), self.node_classifier(x), self.endpoint_classifier(x)
