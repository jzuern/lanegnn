from builtins import Exception
import os, psutil
import threading

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import networkx as nx
import wandb
import argparse
from tqdm import tqdm
import numpy as np
import time
import torch
import torch.utils.data
import torch_geometric.data
from torchmetrics.functional.classification.average_precision import average_precision
from torchmetrics.functional.classification.precision_recall import recall

# For torch_geometric DataParallel training
from torch_geometric.nn import DataParallel
from torch_geometric.loader import DataListLoader
from torch_geometric.data import Batch

from urbanlanegraph_evaluator.evaluator import GraphEvaluator

from lanegnn.utils.params import ParamLib
from lanegnn.utils.graph import assign_edge_lengths, unbatch_edge_index, get_target_data, get_gt_graph

# SELECT MODEL TO BE USED
from lanegnn.learning.lane_mpnn import LaneGNN
from lanegnn.learning.data import PreGraphDataset
from lanegnn.inference.traverse_endpoint import preprocess_predictions, predict_lanegraph


import matplotlib.pyplot as plt


class Trainer():
    """
    LaneGNN trainer class
    """
    def __init__(self, params, model, dataloader_train, dataloader_test, dataloader_trainoverfit, optimizer, logging_dir='./logs'):

        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.dataloader_trainoverfit = dataloader_trainoverfit
        self.params = params
        self.optimizer = optimizer
        self.edge_criterion = torch.nn.BCELoss()
        self.node_criterion = torch.nn.BCELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.total_step = 0

        self.logging_dir = logging_dir


        # plt.ion()  # turns on interactive mode
        self.figure, self.axarr = plt.subplots(1, 2)

        it = iter(self.dataloader_train)
        i = 0
        while i < 1:
            i += 1
            self.one_sample_data = next(it)


    def do_logging(self, data, step, plot_text, split=None, sample_token=None):
        """
        Logging function that is triggered to run in a new thread so it does not block the training
        :param data: PyG data object
        :param step: current step index
        :param plot_text: figure title
        :param split: dataset split
        :param sample_token: uses sample token for logging if provided
        """
        print("\nLogging asynchronously...")

        with torch.no_grad():
            edge_scores, node_scores, endpoint_scores = self.model(data)
            edge_scores = torch.nn.Sigmoid()(edge_scores).squeeze()
            node_scores = torch.nn.Sigmoid()(node_scores).squeeze()
            endpoint_scores = torch.nn.Sigmoid()(endpoint_scores).squeeze()

        if self.params.model.dataparallel:
            data_orig = data.copy()
            data = Batch.from_data_list(data)

        data.edge_scores = edge_scores
        data.node_scores = node_scores

        # Do logging
        num_edges_in_batch = unbatch_edge_index(data.edge_index, data.batch)[0].shape[1]
        data.x = data.x[data.batch == 0]
        data.edge_gt = data.edge_gt[:num_edges_in_batch]
        data.edge_gt_onehot = data.edge_gt_onehot[:num_edges_in_batch]
        data.edge_scores = data.edge_scores[:num_edges_in_batch]
        data.edge_index = data.edge_index[:, :num_edges_in_batch]
        data.edge_attr = data.edge_attr[:num_edges_in_batch]
        data.edge_img_feats = data.edge_img_feats[:num_edges_in_batch]
        data.node_gt = data.node_gt[data.batch == 0]
        data.node_scores = data.node_scores[data.batch == 0]
        data.num_nodes = data.node_scores.shape[0]

        len_gt_graph = data.batch_idx.cpu().numpy()[0]
        node_scores = node_scores[data.batch == 0]
        endpoint_scores = endpoint_scores[data.batch == 0]

        # Get networkx graph objects
        if self.params.model.dataparallel:
            _, node_scores_pred, endpoint_scores_pred, _, img_rgb, node_pos, fut_nodes, _, startpoint_idx = preprocess_predictions(
                self.params, self.model, data_orig)
        else:
            _, node_scores_pred, endpoint_scores_pred, _, img_rgb, node_pos, fut_nodes, _, startpoint_idx = preprocess_predictions(
                self.params, self.model, data)

        graph_pred_nx = predict_lanegraph(fut_nodes, startpoint_idx, node_scores_pred, endpoint_scores_pred, node_pos)
        # graph_gt_nx = get_gt_graph(get_target_data(self.params, data, split=self.))

        edge_scores_pred = data.edge_scores.squeeze().cpu().numpy()
        edge_scores_onehot = data.edge_gt_onehot.squeeze().cpu().numpy()
        node_scores_pred = node_scores.squeeze().cpu().numpy()
        node_scores_endpoint_pred = endpoint_scores.squeeze().cpu().numpy()

        gt_graph = data.gt_graph.cpu().numpy().astype(np.int32)

        # networkx plot
        networkx_graph_gt = torch_geometric.utils.to_networkx(data,
                                                              node_attrs=["node_scores"],
                                                              edge_attrs=["edge_scores"])

        cmap = plt.get_cmap('viridis')
        color_edge_pred = np.hstack([cmap(edge_scores_pred)[:, 0:3], edge_scores_pred[:, None]])
        color_node_pred = np.hstack([cmap(node_scores_pred)[:, 0:3], node_scores_pred[:, None]])


        color_node_endpoint_pred = np.hstack([cmap(node_scores_endpoint_pred)[:, 0:3], node_scores_endpoint_pred[:, None]])

        #ego_regression_target = get_ego_regression_target(self.params, data, split)
        img_rgb = data.rgb.cpu().numpy()[0:256, :, :]
        img_rgb_context = data.rgb_context.cpu().numpy()[0:512, :, :]
        node_pos = data.x.cpu().numpy()
        node_pos[:, [1, 0]] = node_pos[:, [0, 1]]

        figure_log, axarr_log = plt.subplots(1, 5, figsize=(25, 5), dpi=200)
        plt.tight_layout()
        axarr_log[0].cla()
        axarr_log[1].cla()
        axarr_log[2].cla()
        axarr_log[3].cla()
        axarr_log[4].cla()
        axarr_log[0].imshow(img_rgb)
        axarr_log[1].imshow(img_rgb_context)
        axarr_log[2].imshow(img_rgb)
        axarr_log[3].imshow(img_rgb)
        axarr_log[4].imshow(img_rgb)
        axarr_log[0].axis('off')
        axarr_log[1].axis('off')
        axarr_log[2].axis('off')
        axarr_log[3].axis('off')
        axarr_log[4].axis('off')
        axarr_log[0].set_xlim([0, img_rgb.shape[1]])
        axarr_log[0].set_ylim([img_rgb.shape[0], 0])
        axarr_log[2].set_xlim([0, img_rgb.shape[1]])
        axarr_log[2].set_ylim([img_rgb.shape[0], 0])

        context_regr_smooth = data.context_regr_smooth.cpu().numpy()[0:512, :]
        ego_regr_smooth = data.ego_regr_smooth.cpu().numpy()[0:256, :]

        axarr_log[1].imshow(context_regr_smooth, cmap='viridis', alpha=0.5)
        axarr_log[3].imshow(ego_regr_smooth, cmap='viridis', alpha=0.5)

        # Draw GT graph
        nx.draw_networkx(networkx_graph_gt, ax=axarr_log[0], pos=node_pos,
                         edge_color=color_edge_pred,
                         node_color=color_node_pred, with_labels=False, node_size=5)


        nx.draw_networkx(graph_pred_nx, ax=axarr_log[2], pos=nx.get_node_attributes(graph_pred_nx, 'pos'),
                         edge_color=plt.get_cmap('viridis')(1-edge_scores_onehot),
                         node_color=plt.get_cmap('viridis')(1),
                         with_labels=False,
                         node_size=0,
                         )

        for e in range(len(gt_graph[0:len_gt_graph])):
            axarr_log[3].arrow(gt_graph[e, 0], gt_graph[e, 1], (gt_graph[e, 2]-gt_graph[e, 0]), (gt_graph[e, 3] - gt_graph[e, 1]),
                               head_width=4, head_length=4, fc='w', ec='w')

        # drawing updated values
        figure_log.canvas.draw()
        figure_log.canvas.flush_events()

        if sample_token is not None:
            imname = os.path.join(self.logging_dir, "{}.png".format(sample_token))
        else:
            imname = os.path.join(self.logging_dir, "{}.png".format(step))
        try:
            plt.savefig(imname)
            print("Saved logging image to {}".format(imname))
        except Exception as e:
            pass

        if not self.params.main.disable_wandb:
            wandb.log({plot_text: figure_log})

        del figure_log
        del axarr_log
        plt.close()

        print("\n...logging completed!")

    def train(self, epoch):

        self.model.train()
        print('Training')

        train_progress = tqdm(self.dataloader_train)
        for step, data in enumerate(train_progress):

            if step > 1000:
                print('Stopping training at {} samples cause they are so many'.format(step))
                break

            t_start = time.time()
            self.optimizer.zero_grad()

            if self.params.model.dataparallel:
                data = [item.to(self.device) for item in data]
            else:
                data = data.to(self.device)

            # loss and optim
            edge_scores, node_scores, endpoint_scores = self.model(data)
            edge_scores = torch.nn.Sigmoid()(edge_scores).squeeze()
            node_scores = torch.nn.Sigmoid()(node_scores).squeeze()
            endpoint_scores = torch.nn.Sigmoid()(endpoint_scores).squeeze()

            # Convert list of Data to DataBatch for post-processing and loss calculation
            if self.params.model.dataparallel:
                data_orig = data.copy()
                data = Batch.from_data_list(data)

            dijkstra_mass = 0.2 * data.edge_gt.shape[0]
            other_mass = 0.8 * data.edge_gt.shape[0]

            dijkstra_weight = dijkstra_mass / torch.sum(data.edge_gt_onehot)
            other_weight = other_mass / (data.edge_gt.shape[0] - torch.sum(data.edge_gt_onehot)) 

            edge_weight = torch.zeros_like(data.edge_gt) 
            edge_weight = edge_weight + other_weight
            edge_weight[data.edge_gt_onehot == 1] = dijkstra_weight

            node_endpoint_weight = torch.ones_like(data.node_endpoint_gt)
            node_endpoint_weight[data.node_endpoint_gt == 1] = 100.
            node_endpoint_weight /= node_endpoint_weight.sum()

            try:
                loss_dict = {
                    'edge_loss': torch.nn.BCELoss(weight=edge_weight)(edge_scores, data.edge_gt_onehot),
                    'node_loss': torch.nn.BCELoss()(node_scores, data.node_gt),
                    'endpoint_loss': torch.nn.BCELoss(weight=node_endpoint_weight)(endpoint_scores, data.node_endpoint_gt)
                }
            except Exception as e:
                print(e)
                continue

            loss = sum(loss_dict.values())

            if not torch.isnan(loss):
                loss.backward()
                self.optimizer.step()
            else:
                # write skipped samples to log file
                with open(os.path.join(str(data.city[0])+"_logfile.txt"), 'a') as f:
                    f.write('{},{},{},{},{}\n'.format(data.tile_no, data.walk_no, data.idx, loss.item(), epoch))


            if not self.params.main.disable_wandb:
                wandb.log({"train/loss_total": loss.item()})
                wandb.log({"train/edge_loss": loss_dict['edge_loss'].item()})
                wandb.log({"train/node_loss": loss_dict['node_loss'].item()})
                wandb.log({"train/endpoint_loss": loss_dict['endpoint_loss'].item()})

            if not torch.isnan(loss):
                # Visualization
                if self.total_step % 300 == 0:
                    if self.params.model.dataparallel:
                        data = data_orig
                    th = threading.Thread(target=self.do_logging, args=(data, self.total_step, 'train/Images', 'train'), )
                    th.start()

            t_end = time.time()

            text = 'Epoch {} / {} step {} / {}, train loss = {:03f} | Batch time: {:.3f} | Data time: {:.3f}'.\
                format(epoch, self.params.model.num_epochs, step+1, len(self.dataloader_train), loss.item(), t_end-t_start, 0.0)
            train_progress.set_description(text)

            self.total_step += 1


    def eval(self, epoch, split='test', log_images=True, log_all_images=False):

        edge_threshold = 0.5
        node_threshold = 0.5

        self.model.eval()
        print('Evaluating on {}'.format(split))

        if split == 'test':
            dataloader = self.dataloader_test
        elif split == 'trainoverfit':
            dataloader = self.dataloader_trainoverfit
        elif split == 'train':
            dataloader = self.dataloader_train
        else:
            raise ValueError('Unknown split: {}'.format(split))

        dataloader_progress = tqdm(dataloader, desc='Evaluating on {}'.format(split))

        node_losses = []
        node_endpoint_losses = []
        edge_losses = []
        ap_edge_list = []
        ap_node_list = []
        recall_edge_list = []
        recall_node_list = []
        metrics_dict_list = []

        for i_val, data in enumerate(dataloader_progress):

            if self.params.model.dataparallel:
                data = [item.to(self.device) for item in data]
            else:
                data = data.to(self.device)

            with torch.no_grad():
                edge_scores, node_scores, endpoint_scores = self.model(data)
                edge_scores = torch.nn.Sigmoid()(edge_scores).squeeze()
                node_scores = torch.nn.Sigmoid()(node_scores).squeeze()
                endpoint_scores = torch.nn.Sigmoid()(endpoint_scores).squeeze()

            # Convert list of Data to DataBatch for post-processing and loss calculation
            if self.params.model.dataparallel:
                data_orig = data.copy()
                data = Batch.from_data_list(data)

            dijkstra_mass = 0.2 * data.edge_gt.shape[0]
            other_mass = 0.8 * data.edge_gt.shape[0]

            dijkstra_weight = dijkstra_mass / torch.sum(data.edge_gt_onehot)
            other_weight = other_mass / (data.edge_gt.shape[0] - torch.sum(data.edge_gt_onehot))

            edge_weight = torch.zeros_like(data.edge_gt)
            edge_weight = edge_weight + other_weight
            edge_weight[data.edge_gt_onehot == 1] = dijkstra_weight

            node_endpoint_weight = torch.ones_like(data.node_endpoint_gt)
            node_endpoint_weight[data.node_endpoint_gt == 1] = 100.
            node_endpoint_weight /= node_endpoint_weight.sum()

            # loss and optim
            try:
                loss_dict = {
                    'edge_loss': torch.nn.BCELoss(weight=edge_weight)(edge_scores, data.edge_gt_onehot),
                    'node_loss': torch.nn.BCELoss()(node_scores, data.node_gt),
                    'endpoint_loss': torch.nn.BCELoss(weight=node_endpoint_weight)(endpoint_scores,
                                                                                   data.node_endpoint_gt)
                }
            except Exception as e:
                print(e)
                continue

            loss = sum(loss_dict.values())

            node_losses.append(loss_dict['node_loss'].item())
            edge_losses.append(loss_dict['edge_loss'].item())
            node_endpoint_losses.append(loss_dict['endpoint_loss'].item())

            ap_edge = average_precision((edge_scores > edge_threshold).int(), (data.edge_gt > edge_threshold).int(), num_classes=1)
            recall_edge = recall((edge_scores > edge_threshold).int(), (data.edge_gt > edge_threshold).int(), num_classes=1, multiclass=False)
            ap_node = average_precision((node_scores > node_threshold).int(), (data.node_gt > node_threshold).int(), num_classes=1)
            recall_node = recall((node_scores > node_threshold).int(), (data.node_gt > node_threshold).int(), num_classes=1, multiclass=False)

            ap_edge_list.append(ap_edge.item())
            recall_edge_list.append(recall_edge.item())
            ap_node_list.append(ap_node.item())
            recall_node_list.append(recall_node.item())

            data.edge_scores = edge_scores
            data.node_scores = node_scores

            # Get networkx graph objects
            if self.params.model.dataparallel:
                _, node_scores_pred, endpoint_scores_pred, _, img_rgb, node_pos, fut_nodes, _, startpoint_idx = preprocess_predictions(self.params, self.model, data_orig)
            else:
                _, node_scores_pred, endpoint_scores_pred, _, img_rgb, node_pos, fut_nodes, _, startpoint_idx = preprocess_predictions(self.params, self.model, data)

            graph_pred_nx = predict_lanegraph(fut_nodes, startpoint_idx, node_scores_pred, endpoint_scores_pred, node_pos, endpoint_thresh=0.5, rad_thresh=20)
            graph_gt_nx = get_gt_graph(get_target_data(self.params, data, split=split))

            graph_pred_nx = assign_edge_lengths(graph_pred_nx)
            graph_gt_nx = assign_edge_lengths(graph_gt_nx)

            # metrics_dict = calc_all_metrics(graph_gt_nx, graph_pred_nx, split=split)
            metrics_dict = GraphEvaluator().evaluate_graph(graph_gt_nx, graph_pred_nx, area_size=[256, 256])
            metrics_dict_list.append(metrics_dict)

            tile_no = int(data.tile_no[0].cpu().detach().numpy())
            walk_no = int(data.walk_no[0].cpu().detach().numpy())
            idx = int(data.idx[0].cpu().detach().numpy())
            city = data.city[0]
            sample_token = "{}_{:03d}_{:03d}_{:03d}".format(city, tile_no, walk_no, idx)

            # Visualization if not torch.isnan(loss):
            if not torch.isnan(loss):
                if (i_val % 25 == 0 and log_images) or log_all_images:
                    if self.params.model.dataparallel:
                        data = data_orig
                    self.do_logging(data, self.total_step, plot_text='test/Images', split='test', sample_token=sample_token)
            else:
                # write skipped samples to log file
                with open(os.path.join(str(data.city[0])+"_test_logfile.txt"), 'a') as f:
                    f.write('{},{},{},{},{}\n'.format(data.tile_no, data.walk_no, data.idx, loss.item(), epoch))

        ap_edge_mean = np.nanmean(ap_edge_list)
        recall_edge_mean = np.nanmean(recall_edge_list)
        ap_node_mean = np.nanmean(ap_node_list)
        recall_node_mean = np.nanmean(recall_node_list)

        # Calculate mean values for all metrics in metrics_dict
        metrics_dict_mean = {}
        for key in metrics_dict_list[0].keys():
            metrics_dict_mean[key] = np.nanmean([metrics_dict[key] for metrics_dict in metrics_dict_list])
            print('{}: {:.3f}'.format(key, metrics_dict_mean[key]))

        print('Edge AP: {:.3f}, Recall: {:.3f}'.format(ap_edge_mean, recall_edge_mean))
        print('Node AP: {:.3f}, Recall: {:.3f}'.format(ap_node_mean, recall_node_mean))
        print('Edge loss: {:.3f}, Node loss: {:.3f}, Node endpoint loss: {:.3f}'.format(np.mean(edge_losses), np.mean(node_losses), np.mean(node_endpoint_losses)))
        print('Total loss: {:.3f}'.format(np.mean(edge_losses) + np.mean(node_losses)))

        log_dict = {"{}/Edge AP".format(split): ap_edge_mean,
                       "{}/Edge Recall".format(split): recall_edge_mean,
                       "{}/Node AP".format(split): ap_node_mean,
                       "{}/Node Recall".format(split): recall_node_mean,
                       "{}/Edge Loss".format(split): np.mean(edge_losses),
                       "{}/Node Loss".format(split): np.mean(node_losses),}

        print(log_dict)
        print(metrics_dict_mean)

        if not self.params.main.disable_wandb:
            wandb.log(log_dict)
            wandb.log(metrics_dict_mean)


def main():

    # ----------- Parameter sourcing --------------

    parser = argparse.ArgumentParser(description="Train LaneMP architecture")

    # General parameters (namespace: main)
    parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)
    parser.add_argument('--dataset', type=str, help="dataset path")
    parser.add_argument('--version', type=str, help="define the dataset version that is used")

    # Namespace-specific arguments (namespace: training)
    parser.add_argument('--lr', type=str, help='model path')
    parser.add_argument('--epochs', type=str, help='model path')
    parser.add_argument('--city_train', type=str, help='city to train on or concatentation of cities', default=None)
    parser.add_argument('--city_test', type=str, help='city to test on or concatentation of cities', default=None)
    parser.add_argument('--logging_dir', type=str, help='directory to log to', default='logs')

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
    train_path = os.path.join(params.paths.dataroot, params.paths.rel_dataset, "preprocessed", "train", params.paths.config_name)
    test_path = os.path.join(params.paths.dataroot, params.paths.rel_dataset, "preprocessed", "test", params.paths.config_name)

    dataset_train = PreGraphDataset(params, path=train_path,  visualize=params.preprocessing.visualize, city=opt.city_train)
    dataset_test = PreGraphDataset(params, path=test_path, visualize=params.preprocessing.visualize, city=opt.city_test)

    if params.model.dataparallel:
        dataloader_obj = DataListLoader
    else:
        dataloader_obj = torch_geometric.loader.DataLoader

    dataloader_train = dataloader_obj(dataset_train,
                                      batch_size=params.model.batch_size,
                                      num_workers=params.model.loader_workers,
                                      shuffle=True)
    dataloader_test = dataloader_obj(dataset_test,
                                     batch_size=1,
                                     num_workers=1,
                                     shuffle=False)
    dataloader_trainoverfit = dataloader_obj(dataset_train,
                                             batch_size=1,
                                             num_workers=1,
                                             shuffle=False)


    trainer = Trainer(params, model, dataloader_train, dataloader_test, dataloader_trainoverfit, optimizer, opt.logging_dir)

    for epoch in range(params.model.num_epochs):
        trainer.train(epoch)

        if not params.main.disable_wandb:
            wandb_run_name = wandb.run.name

            fname = 'lanegnn/{}_{:03d}.pth'.format(wandb_run_name, epoch)

            # save checkpoint locally and in wandb
            torch.save(model.state_dict(), params.paths.checkpoints + fname)
            wandb.save(params.paths.home + fname)

        # Evaluate
        trainer.eval(epoch, split='test', log_images=True)

        process = psutil.Process(os.getpid())
        print("RAM: ", str(process.memory_info().rss / 1024 / 1024))  # in MB


if __name__ == '__main__':
    main()
