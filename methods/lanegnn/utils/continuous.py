import numpy as np
import skfmm
import cv2
import numpy as np
import yaml
import os
import sys
import cv2
import matplotlib.pyplot as plt
import json
import codecs
from shapely.geometry import LineString, Point
from scipy.interpolate import griddata
import time
import argparse
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import networkx as nx
from torch_geometric.utils import degree
import torch
import torchvision
from collections import defaultdict


def get_cropped_edge_img(edge_angle, mid_x, mid_y, rgb_context):
    """
    Produce masked aerial image of a given edge by rotating and translating the original image.
    :param (float) edge_angle: angle of the edge (in radian)
    :param mid_x: x coordinate of the edge
    :param mid_y: y coordinate of the edge
    :param rgb_context:  (aerial) image
    """
    # Size of quadratic destination image
    crop_size = 100
    imsize = 512

    crop_rad = edge_angle
    crop_x = mid_x + rgb_context.shape[1] // 4
    crop_y = mid_y + rgb_context.shape[1] // 4
    center = np.array([crop_x, crop_y])

    # Source point coordinates already in coordinate system around center point of future crop
    src_pts = np.array([[-crop_size // 2, crop_size // 2 - 1],
                        [-crop_size // 2, -crop_size // 2],
                        [crop_size // 2 - 1, -crop_size // 2],
                        [crop_size // 2 - 1, crop_size // 2 - 1]])

    # Rotate source points
    R = np.array([[np.cos(crop_rad), -np.sin(crop_rad)],
                  [np.sin(crop_rad), np.cos(crop_rad)]])
    src_pts = np.matmul(R, src_pts.T).T + center
    src_pts = src_pts.astype(np.float32)

    # Destination points are simply the corner points in the new image
    dst_pts = np.array([[0, crop_size - 1],
                        [0, 0],
                        [crop_size - 1, 0],
                        [crop_size - 1, crop_size - 1]],
                       dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the rectangle
    crop = cv2.warpPerspective(rgb_context, M, (crop_size, crop_size))

    """
    # visualize cropped region
    for i in range(len(src_pts)):
        #print(i, i-1, (src_pts[i, 0], src_pts[i, 1]), (src_pts[i-1, 0], src_pts[i-1, 1]))
        cv2.line(rgb_context, (src_pts[i, 0], src_pts[i, 1]), (src_pts[i-1, 0], src_pts[i-1, 1]), (255, 255, 0), 2)
        # Query image
        cv2.line(rgb_context, (128, 128), (128+256, 128), (255, 0, 0.0), 2)
        cv2.line(rgb_context, (128+256, 128), (128+256, 128+256), (255, 0, 0), 2)
        cv2.line(rgb_context, (128, 128+256), (128+256, 128+256), (255, 0, 0), 2)
        cv2.line(rgb_context, (128, 128), (128, 128+256), (255, 0, 0), 2)

    fig, axarr = plt.subplots(1, 2)
    fig.set_figheight(16)
    fig.set_figwidth(16)
    axarr[0].imshow(rgb_context)
    axarr[1].imshow(crop)
    plt.show()
    """

    return crop


def is_in_mask_loop(mask, x1, y1, x2, y2, N_interp):
    """
    Function that checks whether a candidate edge defined by two points lies inside a mask.
    A number of N_interp points along the line is evaluated in order to make that decision.
    :param mask: mask that defines the region of interest
    :param x1: x coordinate of the first point
    :param y1: y coordinate of the first point
    :param x2: x coordinate of the second point
    :param y2: y coordinate of the second point
    :param N_interp: number of points along the line that are evaluated
    """
    # Interpolate N times between the two points
    for j in range(N_interp):
        # Interpolate between the two points
        x = int((x1 + (float(j) / N_interp) * (x2 - x1)))
        y = int((y1 + (float(j) / N_interp) * (y2 - y1)))

        # Check if the point is inside the mask
        if mask[y, x]:
            return False

    return True


def make_sdf(mask):
    """
    This function produces a signed-distance function given an input mask.
    Tune the downsample factor if necessary.
    :param mask: mask that is used to produce the sdf
    """
    dx = 2 # downsample factor
    f = 20 / dx  # distance function scale

    try:
        mask = cv2.resize(mask, None, fx=1/dx, fy=1/dx, interpolation=cv2.INTER_NEAREST)
        sdf = skfmm.distance(1 - mask)
        sdf[sdf > f] = f
        sdf = sdf / f
        sdf = 1 - sdf
        sdf = cv2.resize(sdf, None, fx=dx, fy=dx, interpolation=cv2.INTER_NEAREST)
    except:
        sdf = np.zeros_like(mask)
        # plt.imshow(mask, cmap='gray')
        # plt.imshow(sdf, cmap='viridis', alpha=0.5)
        # plt.show()

    return sdf


def get_nn_direction(angles, angles_mask):
    """
    Produce dense direction map using nearest neighbor interpolation
    :param angles: array of angles
    :param angles_mask: region that defines the region of interest
    """
    # Get dense depth using griddata
    dir_pixels = np.where(angles_mask != 0)
    xs = dir_pixels[1]
    ys = dir_pixels[0]
    points = np.vstack((xs, ys)).T

    # Extrapolate
    grid_x, grid_y = np.mgrid[0:256, 0:256]

    dense_dir = griddata(points, angles[dir_pixels], (grid_y, grid_x), method='nearest')

    return dense_dir


def get_gt_sdf_with_direction(gt_lines_shapely):
    """
    Produce signed distance field based on GT lane graph in form of a shapely multiline object.
    Also, generate a dense angle map based on NN angle interpolation of the ground truth graph.
    :param gt_lines_shapely: Shapely multiline object representing the undirected graph.
    :return gt_sdf: distance-to-GT-graph SDF
    :return angles_gt_dense: dense angle map
    """
    # Get dense Pred angles using griddata
    angles_gt = np.zeros([256, 256])
    angles_gt_mask = np.zeros([256, 256])

    for gt_line in gt_lines_shapely:
        gt_x1 = int(gt_line.coords[0][0])
        gt_y1 = int(gt_line.coords[0][1])
        gt_x2 = int(gt_line.coords[1][0])
        gt_y2 = int(gt_line.coords[1][1])

        dir_rad = np.arctan2(float(gt_y2 - gt_y1), float(gt_x2 - gt_x1))
        cv2.line(angles_gt, (gt_x1, gt_y1), (gt_x2, gt_y2), color=dir_rad, thickness=2)
        cv2.line(angles_gt_mask, (gt_x1, gt_y1), (gt_x2, gt_y2), color=1, thickness=2)

    # Get dense GT angles using griddata
    angles_gt_dense = get_nn_direction(angles_gt, angles_gt_mask)

    # Generate GT sdf mask
    gt_mask = np.zeros([256, 256])
    for gt_line in gt_lines_shapely:
        cv2.line(gt_mask, (int(gt_line.coords[0][0]), int(gt_line.coords[0][1])), (int(gt_line.coords[1][0]), int(gt_line.coords[1][1])), color=1, thickness=4)
    gt_sdf = make_sdf(gt_mask)

    return gt_sdf, angles_gt_dense


def get_pointwise_edge_gt(s_x, s_y, e_x, e_y, N_interp, gt_multiline_shapely, angles_gt_dense):
    """
    Given an edge, this function interpolates along the edge and checks the distance to the GT graph (shapely)
    Secondly, a dense GT angle map is used to measure the angle difference between edge and GT graph.
    :param s_x: edge startpoint x-coord
    :param s_y: edge startpoint y-coord
    :param e_x: edge endpoint x-coord
    :param e_y: edge endpoint y-coord
    :param N_interp: int number of interpolation points used for distance calculation
    :param gt_multiline_shapely: Shapely multiline object descrribing the undirected ground truth graph
    :param angles_gt_dense: dense angle map based on NN interpolation.
    :returns cum_edge_distance: mean Euclidean distance to GT graph
    :returns angle_panelty: mean angle distance to GT graph
    """
    interp_dist = list()
    interp_angle = list()

    angle_pred = np.arctan2(e_y - s_y, e_x - s_x)

    for j in range(N_interp+1):
        # Interpolate between the two points
        x = int((s_x + (float(j) / N_interp) * (e_x - s_x)))
        y = int((s_y + (float(j) / N_interp) * (e_y - s_y)))

        interp_point = Point([(x, y)])
        interp_dist.append(interp_point.distance(gt_multiline_shapely))

        angle_gt = angles_gt_dense[x, y]
        angle_relative = np.abs(angle_pred - angle_gt)

        # force angle to be between 0 and pi
        if angle_relative > np.pi:
            angle_relative = 2 * np.pi - angle_relative

        interp_angle.append(angle_relative)

    cum_edge_distance = np.array(interp_dist).mean()
    cum_angle_distance_normalized = np.array(interp_angle).mean() / np.pi

    angle_penalty = (1 - cum_angle_distance_normalized)

    return cum_edge_distance, angle_penalty


def get_ego_regression_target(params, data, split):
    """
    currently not being used
    Used for debugging in dataloader to retrieve centerline regression target.
    """
    tile_no = int(data.tile_no[0].cpu().detach().numpy())
    walk_no = int(data.walk_no[0].cpu().detach().numpy())
    idx = int(data.idx[0].cpu().detach().numpy())

    image_fname = "{}{}{}/{:03d}_{:03d}_{:03d}-centerlines-sdf-ego.png".format(params.paths.dataroot, params.paths.rel_dataset, split, tile_no, walk_no, idx)
    im = cv2.imread(image_fname, cv2.IMREAD_GRAYSCALE)

    return im


def transform_keypoint_to_world(keypoint, ego_x_y_yaw):
    """
    Transform keypoints from image coordinates to world coordinates.
    :param keypoints: list of keypoints in image coordinates
    :param ego_x_y_yaw: ego pose in world coordinates
    :return: list of keypoints in world coordinates
    """
    [x, y, yaw] = ego_x_y_yaw.squeeze().tolist()
    yaw = yaw + np.pi / 2

    R = np.array([[np.cos(yaw), -np.sin(yaw)],
                  [np.sin(yaw), np.cos(yaw)]])

    T = np.eye(3)
    T[0:2, 2] = [x, y]
    T[0:2, 0:2] = R

    keypoint = np.array(keypoint).astype(np.float32)
    keypoint -= np.array([128, 256])
    keypoint = np.append(keypoint, 1)
    keypoint = np.dot(T, keypoint)  # Transform keypoint
    keypoint = keypoint[0:2]
    # keypoint -= np.array([roi_area_xxyy[0], roi_area_xxyy[2]]) # Offset x and y to account for to ROI

    return keypoint


def is_in_roi(ego_x_y_yaw, roi_area_xxyy, margin=0.0):
    """
    Checks if the ego position is within the region of interest area.
    :param ego_x_y_yaw: ego position in x, y, yaw
    :param roi_area_xxyy: region of interest area in x1, x2, y1, y2
    :param margin: margin around the roi area
    :return: True if ego position is within the roi area, False otherwise
    """
    return ego_x_y_yaw[0]-margin > roi_area_xxyy[0] and \
           ego_x_y_yaw[0]+margin < roi_area_xxyy[1] and \
           ego_x_y_yaw[1]-margin > roi_area_xxyy[2] and \
           ego_x_y_yaw[1]+margin < roi_area_xxyy[3]
