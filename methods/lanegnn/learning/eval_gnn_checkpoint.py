
import os
import wandb
import argparse
import torch
import torch.utils.data
import torch_geometric.data
import sys
# Please only comment out, do not delete
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"


# SELECT MODEL TO BE USED
from lanegnn.learning.lane_mpnn import LaneGNN
from lanegnn.learning.data import PreGraphDataset
from lanegnn.utils.params import ParamLib
from methods.train_lanegnn import Trainer

# For torch_geometric DataParallel training
from torch_geometric.nn import DataParallel
from torch_geometric.loader import DataListLoader

#  Please only commment out, do not delete
# import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt

def main():
    # ----------- Parameter sourcing --------------

    parser = argparse.ArgumentParser(description="Train LaneMP architecture")

    # General parameters (namespace: main)
    parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)
    parser.add_argument('--dataset', type=str, help="dataset path")
    parser.add_argument('--version', type=str, help="define the dataset version that is used")

    # Namespace-specific arguments (namespace: model)
    parser.add_argument('--lr', type=str, help='model path')
    parser.add_argument('--epochs', type=str, help='model path')
    parser.add_argument('--city_train', type=str, help='city to train on or concatentation of cities', default=None)
    parser.add_argument

    opt = parser.parse_args()

    params = ParamLib(opt.config)
    params.main.overwrite(opt)
    params.preprocessing.overwrite(opt)
    params.model.overwrite(opt)

    print("Batch size summed over all GPUs: ", params.model.batch_size)

    if not params.main.disable_wandb:
        wandb.login()
        wandb.init(
            entity='wandb_entity',
            project=params.main.project,
            notes='v1',
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.config.update(params.paths)
        wandb.config.update(params.model)
        wandb.config.update(params.preprocessing)

    # -------  Model, optimizer and data initialization ------

    model = LaneGNN(gnn_depth=params.model.gnn_depth,
                    edge_geo_dim=params.model.edge_geo_dim,
                    map_feat_dim=params.model.map_feat_dim,
                    edge_dim=params.model.edge_dim,
                    node_dim=params.model.node_dim,
                    msg_dim=params.model.msg_dim,
                    in_channels=params.model.in_channels,
                    )
    state_dict = torch.load(os.path.join(params.paths.checkpoints, params.model.checkpoint),
                            map_location=torch.device('cuda')
                            )
    model.load_state_dict(state_dict)
    model = model.to(params.model.device)

    # Make model parallel if available
    if params.model.dataparallel:
        print("Let's use DataParallel with", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model)
    else:
        print("Let's NOT use DataParallel with", torch.cuda.device_count(), "GPUs!")

    weights = [w for w in model.parameters() if w.requires_grad]

    optimizer = torch.optim.Adam(weights,
                                 lr=float(params.model.lr),
                                 weight_decay=float(params.model.weight_decay),
                                 betas=(params.model.beta_lo, params.model.beta_hi))

    # define own collator that skips bad samples
    dataset_path = os.path.join(params.paths.dataroot, params.paths.rel_dataset, "preprocessed", "test",
                             params.paths.config_name)

    dataset = PreGraphDataset(params, path=dataset_path, visualize=params.preprocessing.visualize, city=params.city_test)

    if params.model.dataparallel:
        dataloader_obj = DataListLoader
    else:
        dataloader_obj = torch_geometric.loader.DataLoader

    dataloader = dataloader_obj(dataset,
                                     batch_size=1,
                                     num_workers=1,
                                     shuffle=False)

    trainer = Trainer(params, model, dataloader, dataloader, dataloader, optimizer)

    # Evaluate
    trainer.eval(0, split='test', log_images=False, log_all_images=False)

if __name__ == '__main__':
    main()
