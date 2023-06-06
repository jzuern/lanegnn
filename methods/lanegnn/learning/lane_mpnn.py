import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import resnet18, resnet50
from typing import List, Optional, Set

import torch_geometric.nn
from torch_geometric.typing import Adj, Size
from torch_sparse import SparseTensor
from torch_scatter import gather_csr, scatter, segment_csr


def get_map_encoder(out_features=64, in_channels=3):
    """
    Returns a non-pretrained ResNet-18 architecture used for encoding the aerial edge features
    :param out_features: dim of output BEV embedding
    :param in_channels: dim of input image
    :return: resnet18 model
    """
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, out_features)

    return model


def get_large_map_encoder(out_features=2048, in_channels=3):
    """
    Returns a non-pretrained ResNet-50 architecture used for encoding the aerial edge features
    :param out_features: dim of output BEV embedding
    :param in_channels: dim of input image
    :return: resnet50 model
    """
    model = resnet50(pretrained=False)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, out_features)

    return model


class LaneGNN(torch.nn.Module):

    def __init__(self, gnn_depth, edge_geo_dim, map_feat_dim, edge_dim, node_dim, msg_dim, in_channels=3):
        """
        Definition of the LaneGNN architecture that performs encoding of node and edge features as well as
        causal message passing.
        :param gnn_depth: number of message passing steps to be performed
        :param edge_geo_dim: dim of geometric edge features
        :param map_feat_dim: dim of aerial edge features
        :param edge_dim: dim of edge features in message passing stage
        :param node_dim: dim of node features in message passing stage
        :param msg_dim: dim of messages propagated in message passing stage
        :param in_channels: dim of edge-wise aerial images
        """

        super(LaneGNN, self).__init__()
        print("LaneGNN using cauasal message passing and aerial edge features")

        self.edge_geo_dim = edge_geo_dim
        self.depth = gnn_depth

        # Encoding of geometric edge features
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, int(edge_geo_dim/2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(edge_geo_dim/2), edge_geo_dim),
        )

        # Encoding of aerial edge features
        self.map_encoder = get_map_encoder(out_features=map_feat_dim, in_channels=in_channels) # default: 64

        self.fuse_edge = nn.Sequential(
            nn.Linear(edge_geo_dim+map_feat_dim, edge_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(edge_dim*2, edge_dim),
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
        node_feats, edge_img_feats, edge_attr, edge_index, batch = (
            data.x,
            data.edge_img_feats,
            data.edge_attr,
            data.edge_index,
            data.batch_idx
        )
        # Encoding of nodes and edges
        x = self.pose_encoder(node_feats.float()) # N x D
        initial_x = x

        edge_attr = self.edge_encoder(edge_attr.float()) # E x D_E1
        map_attr = self.map_encoder(edge_img_feats) # E x D_E2

        # Combine encoded edge data and oriented BEV feature
        fused_edge_attr = torch.cat([edge_attr, map_attr], dim=1) # E x (D_E1+D_E2)
        edge_attr = self.fuse_edge(fused_edge_attr) # E x (D_E)

        # Perform (multiple) message passing steps
        for i in range(self.depth):
            x, edge_attr = self.message_passing.forward(x=x,
                                                        edge_index=edge_index,
                                                        edge_attr=edge_attr,
                                                        initial_x=initial_x)
            
        return self.edge_classifier(edge_attr), self.node_classifier(x), self.endpoint_classifier(x)

        
class CausalMessagePassing(torch_geometric.nn.MessagePassing):
    def __init__(self, node_dim=16, edge_dim=32, msg_dim=32):
        """
        This class defines the causal message passing architecture (Brasó et al.) that is used in the paper.
        :param node_dim: dim of node features during message passing
        :param edge_dim: dim of edge features during message passing
        :param msg_dim:  dim of messages being propagated
        """
        super(CausalMessagePassing, self).__init__(aggr='add', )

        self.edge_update = nn.Sequential(
            nn.Linear(node_dim*2+edge_dim, node_dim*2),
            nn.ReLU(),
            nn.Linear(node_dim*2, edge_dim*2),
            nn.ReLU(),
            nn.Linear(edge_dim*2, edge_dim),
        )

        self.create_past_msgs = nn.Sequential(
            nn.Linear(node_dim*2+edge_dim, msg_dim*2),
            nn.ReLU(),
            nn.Linear(msg_dim*2, msg_dim)
        )

        self.create_future_msgs = nn.Sequential(
            nn.Linear(node_dim*2+edge_dim, msg_dim*2),
            nn.ReLU(),
            nn.Linear(msg_dim*2, msg_dim)
        )

        self.combine_future_past = nn.Sequential(
            nn.Linear(msg_dim*2, node_dim*2),
            nn.ReLU(),
            nn.Linear(node_dim*2, node_dim)
        )

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        pass

    def forward(self, x, edge_index, edge_attr, initial_x):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index,
                              size=(x.size(0), x.size(0)),
                              x=x,
                              edge_attr=edge_attr,
                              initial_x=initial_x)

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages.
        Args:
            edge_index (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
                :obj:`torch_sparse.SparseTensor` that defines the underlying
                graph connectivity/message passing flow.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its
                shape must be defined as :obj:`[2, num_messages]`, where
                messages from nodes in :obj:`edge_index[0]` are sent to
                nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is of type
                :obj:`torch_sparse.SparseTensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                :func:`propagate`.
            size (tuple, optional): The size :obj:`(N, M)` of the assignment
                matrix in case :obj:`edge_index` is a :obj:`LongTensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :obj:`torch_sparse.SparseTensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self.__explain__):
            coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
                                         size, kwargs)

            msg_aggr_kwargs = self.inspector.distribute(
                'message_and_aggregate', coll_dict)
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                         kwargs)

            msg_kwargs = self.inspector.distribute('message', coll_dict)
            past_msgs, future_msgs, update_edges_attr = self.message(**msg_kwargs)

            rows, cols = edge_index

            # aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)

            # Run aggregate for the past and for the future separately
            messages_past = self.aggregate(past_msgs, cols, dim_size=size[1])
            messages_future = self.aggregate(future_msgs, rows, dim_size=size[0])
            # out = self.aggregate(out, **aggr_kwargs)

            messages = torch.cat([messages_past, messages_future], dim=1)

            update_kwargs = self.inspector.distribute('update', coll_dict)

            return self.update(messages, **update_kwargs), update_edges_attr

    def message(self, x_i: Tensor, x_j: Tensor,
                edge_attr, initial_x_i, initial_x_j) -> Tensor:
        r"""Constructs messages from node :math:`j` to node :math:`i`
        in analogy to :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :obj:`edge_index`.
        This function can take any argument as input which was initially
        passed to :meth:`propagate`.
        Furthermore, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """
        edge_update_features = torch.cat([x_i, x_j, edge_attr], dim=1)
        updated_edge_attr = self.edge_update(edge_update_features)

        future_msg_feats = torch.cat([x_i, updated_edge_attr, initial_x_i], dim=1)
        future_msgs = self.create_future_msgs(future_msg_feats)

        past_msg_feats = torch.cat([x_j, updated_edge_attr, initial_x_j], dim=1)
        past_msgs = self.create_past_msgs(past_msg_feats)

        return past_msgs, future_msgs, updated_edge_attr

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.
        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.
        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """

        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)

    def update(self, inputs: Tensor) -> Tensor:
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`.
        """
        #Combine future and past
        updated_nodes = self.combine_future_past(inputs)

        return updated_nodes

class MessagePassing(torch_geometric.nn.MessagePassing):
    def __init__(self, node_dim=16, edge_dim=32, msg_dim=32):
        """
        This class defines the vanilla message passing architecture that is compared against in the paper.
        :param node_dim: dim of node features during message passing
        :param edge_dim: dim of edge features during message passing
        :param msg_dim:  dim of messages being propagated
        """
        super(MessagePassing, self).__init__(aggr='add', )

        self.edge_update_net = nn.Sequential(
            nn.Linear(node_dim*2+edge_dim, node_dim*2),
            nn.ReLU(),
            nn.Linear(node_dim*2, node_dim*2),
            nn.ReLU(),
            nn.Linear(node_dim*2, edge_dim*2),
            nn.ReLU(),
            nn.Linear(edge_dim*2, edge_dim),
        )

        self.create_msgs = nn.Sequential(
            nn.Linear(node_dim*2+edge_dim, node_dim*2),
            nn.ReLU(),
            nn.Linear(node_dim*2, msg_dim*2),
            nn.ReLU(),
            nn.Linear(msg_dim*2, msg_dim)
        )

        self.downproj_node_feats = nn.Sequential(
            nn.Linear(node_dim*2, int(node_dim*3/2)),
            nn.ReLU(),
            nn.Linear(int(node_dim*3/2), node_dim),
        )

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        pass

    def forward(self, x, edge_index, edge_attr, initial_x):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index,
                              size=(x.size(0), x.size(0)),
                              x=x,
                              edge_attr=edge_attr,
                              initial_x=initial_x)

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        size = self.__check_input__(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self.__explain__):
            coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
                                         size, kwargs)

            msg_aggr_kwargs = self.inspector.distribute(
                'message_and_aggregate', coll_dict)
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        if isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                         kwargs)
            msg_kwargs = self.inspector.distribute('message', coll_dict)

            msgs, updated_edges_attr = self.message(**msg_kwargs)
            rows, cols = edge_index

            # Run aggregate for the past and for the future separately
            messages = self.aggregate(msgs, cols, dim_size=size[1])

            update_kwargs = self.inspector.distribute('update', coll_dict)

            return self.update(messages, **update_kwargs), updated_edges_attr

    def message(self, x_i: Tensor, x_j: Tensor,
                edge_attr, initial_x_i, initial_x_j) -> Tensor:
        edge_update_features = torch.cat([x_i, x_j, edge_attr], dim=1)
        updated_edge_attr = self.edge_update_net(edge_update_features)

        msg_feats = torch.cat([x_j, updated_edge_attr, initial_x_j], dim=1)
        msgs = self.create_msgs(msg_feats)

        return msgs, updated_edge_attr

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)

    def update(self, inputs: Tensor) -> Tensor:
        out_node_feats = self.downproj_node_feats(inputs)
        return out_node_feats
