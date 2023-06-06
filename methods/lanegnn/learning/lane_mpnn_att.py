import torch
import torch.nn as nn

from lane_mpnn import CausalMessagePassing
from lane_mpnn import get_map_encoder, get_large_map_encoder


class LaneGNN(torch.nn.Module):
    def __init__(self, gnn_depth, edge_geo_dim, map_feat_dim, edge_dim, node_dim, msg_dim, in_channels=3):
        """
        DEPRECATED (not used in paper for ablations)
        Defines attention-based LaneGNN architecture that attends to image regions based on the edge features
        provided.
        :param gnn_depth: number of message passing steps
        :param edge_geo_dim: dim of geometric edge features
        :param map_feat_dim: dim of BEV features attended onto
        :param edge_dim: dim of edge features when passing messages
        :param node_dim: dim of node features when passing messages
        :param msg_dim: dim of messages during message passing
        :param in_channels: number of image channels (mostly 4)
        """
        super(LaneGNN, self).__init__()


        assert edge_geo_dim == map_feat_dim

        self.depth = gnn_depth
        self.edge_geo_dim = edge_geo_dim

        self.edge_encoder = nn.Sequential(
            nn.Linear(4, int(edge_geo_dim / 4)),
            nn.ReLU(inplace=True),
            nn.Linear(int(edge_geo_dim / 4), int(edge_geo_dim / 2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(edge_geo_dim / 2), edge_geo_dim),
        )

        self.map_encoder = get_map_encoder(out_features=map_feat_dim, in_channels=in_channels)  # default: 64

        self.fuse_edge = nn.Sequential(
            nn.Linear(edge_geo_dim * 2, edge_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(edge_dim * 2, edge_dim),
        )

        self.pose_encoder = nn.Sequential(
            nn.Linear(2, int(node_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(node_dim / 2), node_dim)
        )

        self.node_classifier = nn.Sequential(
            nn.Linear(node_dim, int(node_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(node_dim / 2), int(node_dim / 4)),
            nn.ReLU(),
            nn.Linear(int(node_dim / 4), 1),
        )

        self.endpoint_classifier = nn.Sequential(
            nn.Linear(node_dim, int(node_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(node_dim / 2), int(node_dim / 4)),
            nn.ReLU(),
            nn.Linear(int(node_dim / 4), 1),
        )

        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_dim, int(edge_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(edge_dim / 2), int(edge_dim / 4)),
            nn.ReLU(),
            nn.Linear(int(edge_dim / 4), 1),
        )

        self.message_passing = CausalMessagePassing(node_dim=node_dim, edge_dim=edge_dim, msg_dim=msg_dim)

        self.map_att = nn.MultiheadAttention(embed_dim=edge_geo_dim, num_heads=1, batch_first=True)

    def forward(self, data):
        node_feats, edge_attr, edge_index, node_gt, edge_gt, rgb, context_regr, batch = (
            data.x,
            data.edge_attr,
            data.edge_index,
            data.node_gt,
            data.edge_gt,
            data.rgb,
            data.context_regr_smooth,
            data.batch_idx
        )

        x = self.pose_encoder(node_feats.float())  # N x D
        initial_x = x


        rgb = rgb.reshape(-1, 256, 256, 3)

        context_regr = context_regr.reshape(-1, 512, 512, 1)
        context_regr = context_regr[:, 128:384, 128:384].reshape(-1, 256, 256, 1)
        map_input = torch.cat([rgb, context_regr], dim=3).permute(0,3,1,2) # reshape(1, 4, 256, 256) # B x C X H X W

        edge_attr = self.edge_encoder(edge_attr.float())  # E x D_E1
        map_attr = self.map_encoder(map_input) # single image # B x 256

        edge_batch_ids = data.batch[edge_index[0, :]]
        map_attr_batch = torch.zeros((edge_attr.shape[0], 256)).cuda()

        # Assign map_attr to each edge depending on the batch the edge belongs to
        for batch_dim_idx in range(torch.max(data.batch).item() + 1):
            map_attr_batch[edge_batch_ids == batch_dim_idx] = map_attr[batch_dim_idx, :]

        map_attr_batch = map_attr_batch.unsqueeze(0)
        map_edge_attd, _ = self.map_att(query=edge_attr.unsqueeze(0), key=map_attr_batch, value=map_attr_batch)

        # Combine encoded edge data and oriented BEV feature
        fused_edge_attr = torch.cat([edge_attr, map_edge_attd.squeeze(0)], dim=1)
        edge_attr = self.fuse_edge(fused_edge_attr)  # E x (D_E)

        for i in range(self.depth):
            x, edge_attr = self.message_passing.forward(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                                        initial_x=initial_x)

        return self.edge_classifier(edge_attr), self.node_classifier(x), self.endpoint_classifier(x)
