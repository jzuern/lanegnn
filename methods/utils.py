import numpy as np
import re


# epsilon for numerical stability
EPS = 1e-6


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


def round_even(x):
  return round(x/2.)*2


def closestNumber(n, m):
    q = round(int(n) / int(m))
    return q*m



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, max_len=100):
        self.reset()
        self.max_len = max_len

    def reset(self):
        self.vals = []
        self.num_vals = 0

    def update(self, val):
        if len(self.vals) < self.max_len:
            self.vals.append(val)
            self.num_vals += 1
        else:
            self.vals.pop(0)
            self.vals.append(val)
        self.mean = np.mean(self.vals)




def calc_iou(intersection_pred, intersection_gt):

    """

    :param intersection_pred: predicted areas of intersection
    :param intersection_gt: Ground Truth areas of intersection
    :return: IoU
    """
    intersection_pred = intersection_pred.squeeze()
    intersection_gt = intersection_gt.squeeze()

    intersection_pred = (intersection_pred > 0.5).int()
    intersection_gt = intersection_gt.int()

    inters = (intersection_pred & intersection_gt).float().sum()
    union = (intersection_pred | intersection_gt).float().sum()

    iou = (inters + EPS) / (union + EPS)
    iou = iou.cpu().numpy()

    return iou

def calc_regression_iou(reg_pred, reg_gt, threshold=0.5):

    """

    :param intersection_pred: predicted areas of intersection
    :param intersection_gt: Ground Truth areas of intersection
    :return: IoU
    """
    reg_pred = reg_pred.squeeze()
    reg_gt = reg_gt.squeeze()

    reg_pred = (reg_pred > threshold).int()
    reg_gt = (reg_gt > threshold).int()

    inters = (reg_pred & reg_gt).float().sum()
    union = (reg_pred | reg_gt).float().sum()

    iou = (inters + EPS) / (union + EPS)
    iou = iou.cpu().numpy()

    return iou

def calc_threshold_acc(scalar_field_pred, scalar_field_gt, threshold=1.25):

    """

    :param scalar_field_pred: H x W scalar field
    :param scalar_field_gt: H x W scalar field
    :param threshold:
    :return: thresholded accuracy
    """

    scalar_field_pred = scalar_field_pred.detach().cpu().numpy()
    scalar_field_gt = scalar_field_gt.detach().cpu().numpy()

    mask = scalar_field_gt > EPS

    scalar_field_gt = scalar_field_gt[mask]
    scalar_field_pred = scalar_field_pred[mask]


    d1 = scalar_field_pred / scalar_field_gt
    d2 = 1. / (d1 + 1e-8)

    max_val = np.maximum(d1, d2)

    if d1.size > 0:
        frac = float(np.count_nonzero(max_val < threshold)) / d1.size
    else:
        frac = 0.0

    return frac


def make_grid(img_list, nrow, ncol):
    """
    Args:
        img_list: list of images
        nrow: number of rows
        ncol: number of columns
    """
    df = 4
    canvas = np.zeros((nrow * img_list[0].shape[0]//df, ncol * img_list[0].shape[1]//df, 3), dtype=np.uint8)
    for i in range(nrow):
        for j in range(ncol):
            if i * ncol + j < len(img_list):
                img = img_list[i * ncol + j][::df, ::df, :]
                canvas[i * img.shape[0]:(i + 1) * img.shape[0], j * img.shape[1]:(j + 1) * img.shape[1], :] = img[:, :, :]

    return canvas