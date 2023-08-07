import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import wandb
import torch
import cv2
from methods.regressors import build_net
import numpy as np
from matplotlib import cm
from torch import nn
import time
from utils import AverageMeter, calc_threshold_acc, calc_regression_iou
import os
from utils import make_grid
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm


class CenterlineDataset(Dataset):
    """
    Dataset definition for centerline / ego-centerline incl. context dataset.
    """
    def __init__(self, root, split, sdf_version):
        """
        Compiles file list of all samples based on the provided SDF type.
        :param root: base directory path
        :param split: dataset split
        :param sdf_version: string defining what kind of segmentation model to train (see below)
        """
        if sdf_version == 'centerlines-sdf-ego':
            rgb_search_string = '*-rgb.png'
            sdf_search_string = '*-centerlines-sdf-ego.png'
        elif sdf_version == 'centerlines-sdf-context':
            rgb_search_string = '*-rgb-context.png'
            sdf_search_string = '*-centerlines-sdf-context.png'
        elif sdf_version == 'centerlines-sdf-ego-context':
            rgb_search_string = '*-rgb-context.png'
            sdf_search_string = '*-centerlines-sdf-ego-context.png'
        else:
            raise Exception('Unknown sdf_version')

        self.rgb_files = []
        self.centerlines_files = []

        print("Split: {}".format(split))

        print("     Searching for files in", os.path.join(root, "*", split, rgb_search_string))
        rgb_files = sorted(glob(os.path.join(root, "*", split, rgb_search_string)))
        centerlines_files = sorted(glob(os.path.join(root, "*", split, sdf_search_string)))
        self.rgb_files.extend(rgb_files)
        self.centerlines_files.extend(centerlines_files)

        assert len(self.rgb_files) == len(self.centerlines_files), "Number of rgb files and centerlines files must match"
        assert len(self.rgb_files) > 0, "No files found. Do you have both train and eval folders?"


        print('Found {} images'.format(len(self.rgb_files)))
        print('Found {} centerlines_files'.format(len(self.centerlines_files)))

        # Shuffle the files together
        c = list(zip(self.rgb_files, self.centerlines_files))
        np.random.shuffle(c)
        self.rgb_files, self.centerlines_files = zip(*c)


        assert len(self.rgb_files) > 0
        assert len(self.rgb_files) == len(self.centerlines_files)

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, index):

        rgb_file = self.rgb_files[index]
        centerlines_file = self.centerlines_files[index]

        rgb = cv2.imread(rgb_file)
        cl = cv2.imread(centerlines_file, cv2.IMREAD_GRAYSCALE)

        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        cl = torch.from_numpy(cl).float() / 255.0

        data_dict = {
            'rgb': rgb,
            'cl': cl
        }

        return data_dict



class Trainer():
    """
    Model trainer class for all lane segmentation models.
    """
    def __init__(self, args, model, context_sdf_model, train_loader, test_loader, optimizer):

        self.model = model
        self.context_sdf_model = context_sdf_model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.optimizer = optimizer
        self.criterion_lanenet = torch.nn.BCELoss()
        self.avg_loss = AverageMeter()
        self.best_mean_acc = -1.0

    def train(self, epoch):

        self.model.train()
        print('Training')

        for i, data in enumerate(self.train_loader):

            if i > 10000:
                print('Stopping epoch after 10000 iterations')
                break

            self.optimizer.zero_grad()

            sdf_target = data['cl'].cuda().unsqueeze(1)
            input_data = data['rgb'].cuda()

            if self.context_sdf_model is not None:
                with torch.no_grad():
                    context_sdf = self.context_sdf_model(input_data)
                    context_sdf = torch.nn.Sigmoid()(context_sdf)
                input_data = torch.cat((input_data, context_sdf), dim=1)

            rgb = input_data[:, 0:3, :, :]

            pred = self.model(input_data)
            pred = torch.sigmoid(pred)
            loss = self.criterion_lanenet(pred, sdf_target)

            loss.backward()

            self.optimizer.step()
            self.avg_loss.update(loss.item())

            if i % 10 == 0:
                print('Epoch {} / {} step {} / {}, train loss = {}'.format(epoch, self.args.n_epochs, i + 1,
                                                                           len(self.train_loader),
                                                                           self.avg_loss.mean))

                if not args.disable_wandb:
                    wandb.log({'Train loss': self.avg_loss.mean})

                sdf_target = sdf_target.cpu().detach().numpy()[0, 0]
                sdf_target_viz = (cm.plasma(sdf_target)[:, :, 0:3] * 255).astype(np.uint8)
                pred = pred.cpu().detach().numpy()[0, 0]
                pred_viz = (cm.plasma(pred)[:, :, 0:3] * 255).astype(np.uint8)
                rgb_viz = rgb.cpu().numpy()[0]
                rgb_viz = np.transpose(rgb_viz, (1, 2, 0))
                rgb_viz = (rgb_viz * 255.).astype(np.uint8)
                pred_viz_overlay = cv2.addWeighted(rgb_viz, 0.5, pred_viz, 0.5, 0)
                target_viz_overlay = cv2.addWeighted(rgb_viz, 0.5, sdf_target_viz, 0.5, 0)

                # concatenate the images
                viz = np.concatenate((rgb_viz, pred_viz_overlay, target_viz_overlay), axis=1)

                if self.context_sdf_model is not None:
                    context_sdf = context_sdf.cpu().detach().numpy()[0, 0]
                    sdf_context_pred_viz = (cm.plasma(context_sdf)[:, :, 0:3] * 255).astype(np.uint8)
                    sdf_context_pred_viz_overlay = cv2.addWeighted(rgb_viz, 0.5, sdf_context_pred_viz, 0.5, 0)
                    viz = np.concatenate((viz, sdf_context_pred_viz_overlay), axis=1)

                if args.visualize:
                    cv2.imshow('viz', viz)
                    cv2.waitKey(1)

    def eval(self, split, epoch):

        self.model.eval()
        print('Evaluating split: {}'.format(split))

        accs = []
        ious = []
        losses = []

        target_overlay_list = []
        pred_overlay_list = []

        for i, data in tqdm(enumerate(self.test_loader)):

            if i > 1000:
                print('Stopping eval early')
                break


            target = data['cl'].cuda().unsqueeze(1)
            input_data = data['rgb'].cuda()

            if self.context_sdf_model is not None:
                with torch.no_grad():
                    context_sdf = self.context_sdf_model(input_data)
                    context_sdf = torch.nn.Sigmoid()(context_sdf)
                input_data = torch.cat((input_data, context_sdf), dim=1)

            rgb = input_data[:, 0:3, :, :]

            with torch.no_grad():
                pred = self.model(input_data)
                pred = torch.sigmoid(pred)
                loss_eval = self.criterion_lanenet(pred, target)


            losses.append(loss_eval.item())
            accs.append(calc_threshold_acc(pred, target, threshold=1.25))
            ious.append(calc_regression_iou(pred, target, threshold=0.5))

            # log the first images in eval dataset
            target = target.cpu().detach().numpy()[0, 0]
            pred = pred.cpu().detach().numpy()[0, 0]

            # color coding
            pred_viz = (cm.plasma(pred)[:, :, 0:3] * 255).astype(np.uint8)
            target_viz = (cm.plasma(target)[:, :, 0:3] * 255).astype(np.uint8)

            rgb_viz = rgb.cpu().numpy()[0]
            rgb_viz = np.transpose(rgb_viz, (1, 2, 0))
            rgb_viz = (rgb_viz * 255.).astype(np.uint8)

            target_overlay = cv2.addWeighted(np.ascontiguousarray(rgb_viz), 0.5, np.ascontiguousarray(target_viz), 0.5, 0)
            pred_overlay = cv2.addWeighted(np.ascontiguousarray(rgb_viz), 0.5, np.ascontiguousarray(pred_viz), 0.5, 0)

            target_overlay_list.append(target_overlay)
            pred_overlay_list.append(pred_overlay)

        mean_loss = np.mean(np.array(losses))
        mean_acc = np.mean(np.array(accs))
        mean_iou = np.mean(np.array(ious))

        # Make grid of images
        target_overlay_grid = make_grid(target_overlay_list, nrow=8, ncol=8)
        pred_overlay_grid = make_grid(pred_overlay_list, nrow=8, ncol=8)

        if not args.disable_wandb:
            wandb.log({"Samples {}".format(split): [wandb.Image(target_overlay_grid, caption="GT"),
                                                    wandb.Image(pred_overlay_grid, caption="Pred")],
                       "Test loss {}".format(split): mean_loss,
                       "Test accuracy {}".format(split): mean_acc,
                       "Test IoU {}".format(split): mean_iou})

            if split == "test":
                wandb_run_name = wandb.run.name

                if not os.path.exists('./checkpoints/regressor-centerline'):
                    os.mkdir('./checkpoints/regressor-centerline')

                model_dir = './checkpoints/regressor-centerline/{}_{}_e_{:03d}_acc_{:.3f}.pth'.\
                    format(args.sdf_version, wandb_run_name, epoch, mean_acc)

                print('Saving model to: {}'.format(model_dir))
                state = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer}
                torch.save(state, model_dir)
            else:
                print('NOT saving model cause no wandb logging.')


class Scratch(object):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Lane Regression training')
    parser.add_argument('--bsize', type=int, help='batch size for training', default=8)
    parser.add_argument('-d', '--disable-wandb', help='Use W&B for datalogging', action='store_true')
    parser.add_argument('--lr', type=float, help='learning rate)', default=1e-3)
    parser.add_argument('--n_epochs', type=int, help='number of epochs)', default=1000)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--sdf_version", choices=["centerlines-sdf-ego", "centerlines-sdf-context", "centerlines-sdf-ego-context"], default="centerlines-sdf-context")
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--augment", action='store_true')
    parser.add_argument('--checkpoint_path_context_regression', type=str, help='path to checkpoint of context regression model', default=None)
    parser.add_argument('--checkpoint_path', type=str, help='path to checkpoint of model')

    args = parser.parse_args()

    if not args.disable_wandb:
        wandb.init(entity='wandb_entity', project="centerlines", config=args)

    dataset_train = CenterlineDataset(args.dataset_root,
                                      split="train",
                                      sdf_version=args.sdf_version)
    dataset_test = CenterlineDataset(args.dataset_root,
                                     split='eval',
                                     sdf_version=args.sdf_version)

    dataloader_train = DataLoader(dataset_train,
                                  batch_size=args.bsize,
                                  shuffle=True,
                                  num_workers=4,
                                  drop_last=True)
    dataloader_test = DataLoader(dataset_test,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=1,
                                 drop_last=False)

    context_sdf_model = None

    if args.checkpoint_path_context_regression is not None:
        context_sdf_model = build_net.build_network(snapshot=None, num_channels=3, backend='resnet152', use_cuda=True,
                                                    n_classes=1)
        context_sdf_model = nn.DataParallel(context_sdf_model)

        checkpoint = torch.load(args.checkpoint_path_context_regression)
        context_sdf_model.load_state_dict(checkpoint['state_dict'])

        print("Loaded context regression model from {}".format(args.checkpoint_path_context_regression))

        in_channels = 4
    else:
        in_channels = 3

    print("Using {} channels for model".format(in_channels))

    model = build_net.build_network(snapshot=None, num_channels=in_channels, backend='resnet152', use_cuda=True, n_classes=1)
    model = nn.DataParallel(model)

    if args.checkpoint_path is not None:
        data = torch.load(args.checkpoint_path)
        model.load_state_dict(data['state_dict'])
        print('Loaded model checkpoint from {}'.format(args.checkpoint_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    trainer = Trainer(args, model, context_sdf_model, dataloader_train, dataloader_test, optimizer)

    for epoch in range(args.n_epochs):
        trainer.train(epoch)
        trainer.eval("eval", epoch)