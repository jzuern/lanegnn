import os
import glob
import json
import numpy as np
import cv2
import networkx as nx
import random

import torch
import torch_geometric
import torchvision.transforms as T

from lanegnn.utils.sampling import halton, get_random_edges
from lanegnn.utils.continuous import is_in_mask_loop, get_cropped_edge_img
from lanegnn.learning.lane_mpnn import LaneGNN
from methods.regressors import build_net



def get_context_centerline_regressor(ckpt=None, use_cuda=False):
    model = build_net.build_network(snapshot=ckpt, backend='resnet152', use_cuda=use_cuda, n_classes=1)
    return model

def get_ego_centerline_regressor(ckpt=None, use_cuda=False, num_channels=3):
    model = build_net.build_network(snapshot=ckpt, backend='resnet152', use_cuda=use_cuda, n_classes=1, num_channels=num_channels)
    return model


def live_halton_sampling(non_drivable_mask, num_node_samples: int):
    """
    Performs point sampling based on Halton sequences. Filter points based on mask.
    :param non_drivable_mask: Filter for non-drivable areas
    :param num_node_samples: number of points to sample.
    """
    point_coords = halton(2, num_node_samples - 1) * 255
    halton_points = point_coords.astype(np.int32)

    # filter all points where non_drivable_mask is True
    point_coords = halton_points[np.logical_not(non_drivable_mask[halton_points[:, 0], halton_points[:, 1]])]

    point_coords = np.concatenate((point_coords, np.array([[255, 128]])), axis=0)

    return point_coords


def generate_edges_live(params, rgb_context, context_regr_smooth, non_drivable_mask, point_coords):
    """
    Performs node and edge feature generation based on sampled points.
    Checks whether an edge falls within the drivable area.
    Generates cropped edge BEV feature as well as geometric edge feature.
    :param params: Parameter configuration
    :param rgb_context: Context image
    :param context_regr_smooth: Regression of context image
    :param non_drivable_mask: Mask used for checking whether edge lies within drivable area.
    :param point_coords: Coordinates of sampled node positions.
    :return:
    """
    edge_proposal_pairs = get_random_edges(point_coords)
    edge_proposal_pairs = np.unique(edge_proposal_pairs, axis=0)
    edge_proposal_pairs = edge_proposal_pairs.tolist()

    edges = list()
    edges_locs = list()
    node_feats_list = list()

    for i, anchor in enumerate(point_coords):
        node_tensor = torch.tensor([anchor[0], anchor[1]]).reshape(1, -1)
        node_feats_list.append(node_tensor)

    for [i, j] in edge_proposal_pairs:
        anchor = point_coords[i]
        point = point_coords[j]

        if is_in_mask_loop(non_drivable_mask, anchor[1], anchor[0], point[1], point[0], params.preprocessing.N_interp):
            edges_locs.append((anchor, point))
            edges.append((i, j))

    # Scales edge img feature to VGG16 input size
    transform2vgg = T.Compose([
        T.ToPILImage(),
        T.Resize(32),
        T.ToTensor()])

    # Declare to be filled array
    edge_img_feats = np.zeros((len(edges), 1, 4, 32, 32), dtype=np.float32)

    # This function is called from each pool worker
    global multiprocess_edge

    def multiprocess_edge(i, def_param=edge_img_feats):
        edge = edges[i]

        i, j = edge
        s_x, s_y = point_coords[i][1], point_coords[i][0]
        e_x, e_y = point_coords[j][1], point_coords[j][0]

        delta_x, delta_y = e_x - s_x, e_y - s_y
        mid_x, mid_y = s_x + delta_x / 2, s_y + delta_y / 2

        edge_len = np.sqrt(delta_x ** 2 + delta_y ** 2)
        edge_angle = np.arctan(delta_y / (delta_x + 1e-6))

        edge_tensor = torch.tensor([edge_angle, edge_len, mid_x, mid_y]).reshape(1, -1)
        edge_attr_list.append(edge_tensor)
        edge_idx_list.append((i, j))

        # Crop edge images:
        crop_img_rgb = get_cropped_edge_img(edge_angle, mid_x, mid_y, rgb_context)
        crop_img_rgb_resized = transform2vgg(crop_img_rgb).unsqueeze(0).numpy()
        crop_img_sdf = get_cropped_edge_img(edge_angle, mid_x, mid_y, context_regr_smooth)
        crop_img_sdf_resized = transform2vgg(crop_img_sdf).unsqueeze(0).numpy()

        # RGB and SDF in range [0.0, 1.0] float32
        feats = np.concatenate([crop_img_rgb_resized, crop_img_sdf_resized], axis=1)

        edge_img_feats[i] = feats

    edge_attr_list = list()
    edge_idx_list = list()

    # Multiprocessing pool
    # Not faster if number of edges @ ~400
    # pool = multiprocessing.Pool(processes=4)
    # pool.map(multiprocess_edge, range(len(edges)))

    for i in range(len(edges)):
        multiprocess_edge(i)

    edge_img_feats = torch.from_numpy(edge_img_feats).float().squeeze()
    edges = torch.tensor(edges)
    edge_attr = torch.cat(edge_attr_list, dim=0) if edge_attr_list else torch.tensor([])
    node_feats = torch.cat(node_feats_list, dim=0) if edge_attr_list else torch.tensor([])

    return edges, edge_img_feats, edge_attr, node_feats


def generate_data_live(ego_regressor, context_regressor, params, satellite_image, ego_x_y_yaw, roi_area_xxyy, step_counter):
    """
    1. Performance inference to retrieve ego/context regression.
    2. Crops and rotates satellite image, ego/context regression.
    3. Generates non-drivable mask and produces Halton sampling on drivable area
    4. Generates edge and nodes incl. their features
    5. Produces Data() object used in LaneGNN
    :param ego_regressor: model trained to be perform ego-drivable regression
    :param context_regressor: model trained for performing context regression
    :param params: parameter configuration
    :param satellite_image: satellite image to be cropped and rotated given the pose (next)
    :param ego_x_y_yaw: current pose used for transform and cropping of sat image
    :param roi_area_xxyy: bounds on sat image
    :param step_counter: used for output file names when plotting
    :returns data, ego_regr_smooth, context_regr_smooth: PyG data object, ego and context regression
    """
    # Crop satellite image at ego position
    csize = 512

    [x, y, yaw] = ego_x_y_yaw.squeeze().tolist()
    x = x - roi_area_xxyy[0]
    y = y - roi_area_xxyy[2]
    yaw = yaw + np.pi / 2

    R = np.array([[np.cos(yaw), -np.sin(yaw)],
                  [np.sin(yaw), np.cos(yaw)]])

    center = np.array([csize, csize])

    # Define crop area
    satellite_image = satellite_image[int(y - csize) : int(y + csize),
                                      int(x - csize) : int(x + csize)].copy()

    # For bottom centered
    src_pts = np.array([[-128, 0],
                        [-128, -255],
                        [127, -255],
                        [127, 0]])
    src_pts_context = np.array([[-256, 128],
                                [-256, -383],
                                [255, -383],
                                [255, 128]])

    # Rotate source points
    src_pts = (np.matmul(R, src_pts.T).T + center).astype(np.float32)
    src_pts_context = (np.matmul(R, src_pts_context.T).T + center).astype(np.float32)


    # Destination points are simply the corner points
    dst_pts = np.array([[0, csize // 2 - 1],
                        [0, 0],
                        [csize // 2 - 1, 0],
                        [csize // 2 - 1, csize // 2 - 1]],
                       dtype="float32")
    dst_pts_context = np.array([[0, csize - 1],
                                [0, 0],
                                [csize - 1, 0],
                                [csize - 1, csize - 1]],
                               dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_context = cv2.getPerspectiveTransform(src_pts_context, dst_pts_context)

    rgb = cv2.warpPerspective(satellite_image, M, (csize//2, csize//2), cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    rgb_context = cv2.warpPerspective(satellite_image, M_context, (csize, csize), cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    rgb2sdf_context = torch.from_numpy(cv2.cvtColor(rgb_context, cv2.COLOR_RGB2BGR)).permute(2, 0, 1).float() / 255.0
    rgb2sdf_context = rgb2sdf_context.to(params.driving.device)

    # Get context regressor output
    with torch.no_grad():
        context_regr = torch.nn.Sigmoid()(context_regressor(rgb2sdf_context.unsqueeze(0)))
    context_regr_ = context_regr.detach().cpu().numpy()[0, 0]
    context_regr_smooth = context_regr_ # Omit smoothing: cv2.GaussianBlur(context_regr_, (1, 1), cv2.BORDER_CONSTANT)

    if params.driving.render_onscreen:
        cv2.imshow("context_regr_smooth", context_regr_smooth)
        cv2.imshow("context_regr", context_regr_)

    # Get ego regressor output
    if params.preprocessing.ego_regressor_num_channels == 4:
        rgb_for_cat = torch.from_numpy(cv2.cvtColor(rgb_context, cv2.COLOR_RGB2BGR)).permute(2, 0, 1).float() / 255.0
        rgb_for_cat = rgb_for_cat.to(params.driving.device)
        rgb2sdf = torch.cat((rgb_for_cat, context_regr[0]), dim=0)
    else:
        rgb2sdf = torch.from_numpy(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)).permute(2, 0, 1).float() / 255.0

    with torch.no_grad():
        ego_regr = torch.nn.Sigmoid()(ego_regressor(rgb2sdf.unsqueeze(0)))
    ego_regr_ = ego_regr.detach().cpu().numpy()[0, 0]
    ego_regr_smooth = ego_regr_ # Omit smoothing cv2.GaussianBlur(ego_regr_, (1, 1), cv2.BORDER_CONSTANT)

    if params.preprocessing.ego_regressor_num_channels == 4:
        # We need to cut this down to proper size again (remove context)
        ego_regr_smooth = ego_regr_smooth[128:128 + 256, 128:128 + 256]

    # Get non_drivable_mask.
    non_drivable_mask = ego_regr_smooth < params.driving.non_drivable_thresh

    output_dir_viz = os.path.join(params.driving.experiment_dir,
                                  "continuous_viz/")

    # Onscreen rendering / writing continuous data
    if params.driving.render_onscreen:
        cv2.imshow("ego_regr_smooth", ego_regr_smooth)
        cv2.imshow("ego_regr", ego_regr)
        cv2.waitKey(1)
    if params.driving.render_onscreen or params.driving.write_continuous_viz:
        # Cv2 shenanigans
        non_drivable_mask_viz = non_drivable_mask.astype(np.uint8) * 255
        non_drivable_mask_viz = cv2.applyColorMap(non_drivable_mask_viz, cv2.COLORMAP_VIRIDIS)
        context_regr_smooth_viz = cv2.applyColorMap((context_regr_smooth * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        ego_regr_smooth_viz = cv2.applyColorMap((ego_regr_smooth * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

        rgb_context = cv2.cvtColor(rgb_context, cv2.COLOR_BGR2RGB)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        context_regr_smooth_viz = cv2.addWeighted(context_regr_smooth_viz, 0.5, rgb_context, 0.5, 0)
        ego_regr_smooth_viz = cv2.addWeighted(ego_regr_smooth_viz, 0.5, rgb, 0.5, 0)
        non_drivable_mask_viz = cv2.addWeighted(non_drivable_mask_viz, 0.5, rgb, 0.5, 0)
    if params.driving.render_onscreen:
        cv2.imshow('non_drivable_mask', non_drivable_mask_viz)
        cv2.imshow('context_regr_smooth', context_regr_smooth_viz)
        cv2.imshow('ego_regr_smooth', ego_regr_smooth_viz)
        cv2.imshow('rgb_context', rgb_context)
        cv2.imshow('rgb', rgb)
    if params.driving.write_continuous_viz:
        cv2.imwrite(os.path.join(output_dir_viz, "{:05d}-non_drivable_mask.png").format(step_counter), non_drivable_mask_viz)
        cv2.imwrite(os.path.join(output_dir_viz, "{:05d}-context_regr_smooth.png").format(step_counter), context_regr_smooth_viz)
        cv2.imwrite(os.path.join(output_dir_viz, "{:05d}-ego_regr_smooth.png").format(step_counter), ego_regr_smooth_viz)
        cv2.imwrite(os.path.join(output_dir_viz, "{:05d}-rgb_context.png").format(step_counter), rgb_context)
        cv2.imwrite(os.path.join(output_dir_viz, "{:05d}-rgb.png").format(step_counter), rgb)

    # Halton sampling of points
    point_coords = live_halton_sampling(non_drivable_mask, params.preprocessing.num_node_samples)

    # Generate edge proposals
    edges, edge_img_feats, edge_attr, node_feats = generate_edges_live(params, rgb_context, context_regr_smooth, non_drivable_mask, point_coords)

    data = torch_geometric.data.Data(x=node_feats,
                                     edge_index=edges.t().contiguous(),
                                     edge_attr=edge_attr,
                                     edge_img_feats=edge_img_feats,
                                     num_nodes=node_feats.shape[0],
                                     batch_idx=torch.tensor(0),
                                     rgb=torch.FloatTensor(rgb / 255.),
                                     ).to(params.driving.device)

    return data, ego_regr_smooth, context_regr_smooth


def get_init_poses(params):
    """
    Loads init poses (provided by compile_init_poses.py) for driving.
    :param params: parameter configuration
    :return: drive initialization poses
    """
    if params.driving.mode == "multiple" or params.driving.mode == "distributed":
        poses_path = glob.glob(os.path.join(params.driving.data_dir,
                                            params.driving.city,
                                            "*_{}_*_init_poses.json".format(params.driving.tile_no)))[0]
        with open(poses_path, 'r') as f:
            poses = json.load(f)
    else:
        poses = None
    return poses

def load_sat_img(params,):
    """
    Loads satellite image.
    :param params: parameter configuration
    :return: satellite image
    """
    # Obtain satellite image
    sat_image_path = glob.glob(os.path.join(params.driving.data_dir,
                                            params.driving.city,
                                            "*_{}_*-raw.png".format(params.driving.tile_no)))[0]
    sat_image = cv2.cvtColor(cv2.imread(sat_image_path), cv2.COLOR_BGR2RGB)
    return sat_image

def init_model(params):
    """
    Loads pretrained LaneGNN, ego regressor, context regressor models for inference.
    :param params: parameter configuration
    :return: LaneGNN, ego-regressor, context-regressor models and satellite image.
    """
    sat_image = load_sat_img(params)

    model = LaneGNN(gnn_depth=params.model.gnn_depth,
                    edge_geo_dim=params.model.edge_geo_dim,
                    map_feat_dim=params.model.map_feat_dim,
                    edge_dim=params.model.edge_dim,
                    node_dim=params.model.node_dim,
                    msg_dim=params.model.msg_dim,
                    in_channels=params.model.in_channels,
                    )
    model = model.to(params.driving.device)
    model.load_state_dict(torch.load(os.path.join(params.paths.checkpoints,
                                                  params.model.checkpoint)))
    model.eval()
    print("LaneGNN loaded @ cuda: {}".format(next(model.parameters()).is_cuda))

    ego_regressor = get_ego_centerline_regressor(ckpt=os.path.join(params.paths.checkpoints,
                                                                   params.preprocessing.ego_regressor_ckpt),
                                                 num_channels=params.preprocessing.ego_regressor_num_channels,
                                                 use_cuda=False).eval()
    ego_regressor.to(params.driving.device)
    print("ego_centerline_regressor loaded @cuda: {}".format(next(model.parameters()).is_cuda))

    context_regressor = get_context_centerline_regressor(ckpt=os.path.join(params.paths.checkpoints,
                                                                           params.preprocessing.context_regressor_ckpt),
                                                         use_cuda=False).eval()
    context_regressor.to(params.driving.device)
    print("context_centerline_regressor loaded @cuda: {}".format(next(model.parameters()).is_cuda))

    return model, context_regressor, ego_regressor, sat_image


def render_current_graph(params, agg_graph, agg_graph_before, pred_graph, visited_edges, satellite_image, ego_x_y_yaw,
                         step_vector, roi_area_xxyy, split_edge_list, step_counter):
    """
    Function used for plotting driving results in an onscreen fashion or saves it.
    :param params: parameter configuratoin
    :param agg_graph: aggregated graph
    :param agg_graph_before: previously aggregated graph
    :param pred_graph: predicted graph
    :param visited_edges: seen edges (traversed before)
    :param satellite_image: aerial image
    :param ego_x_y_yaw: current pose
    :param step_vector: current vector generated via last traversal step
    :param roi_area_xxyy: aerial image bounds
    :param split_edge_list: list containing all potential frontier edges
    :param step_counter: counter used for saving plots
    :return: plotting canvas for onscreen visualization
    """
    graph_canvas = satellite_image.copy()
    before_agg_graph_canvas = satellite_image.copy()

    print("split_edge_list", split_edge_list)

    #Draw the graph
    for e in split_edge_list:
        inv_weight, e = e
        #if e[0] in agg_graph.nodes and e[1] in agg_graph.nodes:
        p1 = np.array([e[0][0], e[0][1]]) - roi_area_xxyy[0::2]
        p2 = np.array([e[1][0], e[1][1]]) - roi_area_xxyy[0::2]
        p1 = (int(p1[0]), int(p1[1]))
        p2 = (int(p2[0]), int(p2[1]))
        cv2.arrowedLine(before_agg_graph_canvas, p1, p2, (0, 255, 0), 3)
        cv2.arrowedLine(graph_canvas, p1, p2, (0, 255, 0), 3)


    # BEFORE AGG
    for e in agg_graph_before.edges():
        p1 = agg_graph_before.nodes[e[0]]['pos'] - roi_area_xxyy[0::2]
        p2 = agg_graph_before.nodes[e[1]]['pos'] - roi_area_xxyy[0::2]
        if e in visited_edges:
            cv2.arrowedLine(before_agg_graph_canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                            (255, 255, 255),
                            thickness=1,
                            line_type=cv2.LINE_AA, tipLength=0.1)
        else:
            cv2.arrowedLine(before_agg_graph_canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0),
                            thickness=1,
                            line_type=cv2.LINE_AA, tipLength=0.1)
    for e in pred_graph.edges():
        p1 = pred_graph.nodes[e[0]]['pos'] - roi_area_xxyy[0::2]
        p2 = pred_graph.nodes[e[1]]['pos'] - roi_area_xxyy[0::2]
        cv2.arrowedLine(before_agg_graph_canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), thickness=1,
                        line_type=cv2.LINE_AA, tipLength=0.1)

    # AFTER AGG
    for e in agg_graph.edges():
        p1 = agg_graph.nodes[e[0]]['pos'] - roi_area_xxyy[0::2]
        p2 = agg_graph.nodes[e[1]]['pos'] - roi_area_xxyy[0::2]
        if e in visited_edges:
            cv2.arrowedLine(graph_canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 255), thickness=1,
                            line_type=cv2.LINE_AA, tipLength=0.1)
        else:
            cv2.arrowedLine(graph_canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0), thickness=1,
                            line_type=cv2.LINE_AA, tipLength=0.1)
    for e in pred_graph.edges():
        p1 = pred_graph.nodes[e[0]]['pos'] - roi_area_xxyy[0::2]
        p2 = pred_graph.nodes[e[1]]['pos'] - roi_area_xxyy[0::2]
        cv2.arrowedLine(graph_canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), thickness=1,
                        line_type=cv2.LINE_AA, tipLength=0.1)

    # Render ego position on satellite image and yaw
    ego_in_roi = ego_x_y_yaw[0:2] - np.array([roi_area_xxyy[0], roi_area_xxyy[2]])
    cv2.circle(graph_canvas, (int(ego_in_roi[0]), int(ego_in_roi[1])), 3, (255, 255, 0), -1)
    cv2.circle(before_agg_graph_canvas, (int(ego_in_roi[0]), int(ego_in_roi[1])), 3, (255, 255, 0), -1)

    # check if node has score attr
    agg_out_degree = dict(agg_graph.out_degree(list(agg_graph.nodes())))
    for n in agg_graph.nodes():
        if agg_out_degree[n] >= 2:
            p = agg_graph.nodes[n]['pos'] - roi_area_xxyy[0::2]
            tree = nx.dfs_tree(agg_graph, source=n, depth_limit=10)
            tree_weight = np.sum([agg_graph.nodes[k]['weight'] for k in tree])

            # Draw split point
            cv2.circle(graph_canvas, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)

            # Write DFS tree weight next to each split node
            cv2.putText(graph_canvas, "tw=" + str(int(tree_weight)), (int(p[0]) - 10, int(p[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)


    start = ego_in_roi - step_vector

    #cv2.arrowedLine(graph_canvas, start_tuple, end_tuple, (255, 0, 142), thickness=1, line_type=cv2.LINE_AA, tipLength=0.2)
    cv2.putText(graph_canvas, "" + str(step_counter), (int(start[0])-10, int(start[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 142), 1, cv2.LINE_AA)

    combined_plot = cv2.addWeighted(satellite_image, 0.3, graph_canvas, 0.7, 0.0)
    # combined_plot_before = cv2.addWeighted(satellite_image, 0.3, before_agg_graph_canvas, 0.7, 0.0)

    # Serialize
    if step_counter % 1 == 0 and params.driving.write_graph_viz:
        hash = str(random.getrandbits(8))
        write_path = os.path.join(params.driving.experiment_dir,
                                  "prints/"+hash+"_{:03d}.png".format(step_counter))

        cv2.imwrite(write_path, cv2.cvtColor(combined_plot, cv2.COLOR_BGR2RGB))
        # write_path_agg_before = os.path.join(params.driving.experiment_dir,
        #                                      "prints/"+hash+"_{:03d}_before.png".format(step_counter))
        # cv2.imwrite(write_path_agg_before, cv2.cvtColor(combined_plot_before, cv2.COLOR_BGR2RGB))

    # Visualize current crop
    if params.driving.render_onscreen:
        x_min = int(ego_in_roi[0] - 300)
        x_max = int(ego_in_roi[0] + 300)
        y_min = int(ego_in_roi[1] - 300)
        y_max = int(ego_in_roi[1] + 300)
        combined_plot_crop = combined_plot[y_min:y_max, x_min:x_max].copy()
        combined_plot_crop = cv2.resize(combined_plot_crop, (0, 0), fx=1.5, fy=1.5)

        cv2.imshow("Prediction", cv2.cvtColor(combined_plot_crop, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

    return graph_canvas


def create_drive_experiment_folders(params):
    """
    Function that generates the folder structure for all kinds of driving experiment instances.
    :param params: parameter configuration
    :return: list of all generated directories
    """
    import datetime
    now = datetime.datetime.now()
    # dd-mm-yy_H-M-S
    dt_string = now.strftime("%d-%m-%y")
    dt_string += "_px{}_steps{}_age{}_branch{}".format(params.driving.threshold_px,
                                                       params.driving.max_steps,
                                                       params.driving.max_branch_age,
                                                       params.driving.max_branch_id)

    paths = list()
    params.driving.experiment_dir = os.path.join(params.driving.output_dir,
                                                        params.driving.mode,
                                                        params.driving.city,
                                                        str(params.driving.tile_no),
                                                        dt_string,
                                                 )
    paths.append(params.driving.experiment_dir)
    if not os.path.exists(params.driving.experiment_dir):
        os.makedirs(params.driving.experiment_dir)

    paths.append(os.path.join(params.driving.experiment_dir, "continuous_viz"))
    if not os.path.exists(os.path.join(params.driving.experiment_dir, "continuous_viz")):
        os.makedirs(os.path.join(params.driving.experiment_dir, "continuous_viz"))

    paths.append(os.path.join(params.driving.experiment_dir, "prints"))
    if not os.path.exists(os.path.join(params.driving.experiment_dir, "prints")):
        os.makedirs(os.path.join(params.driving.experiment_dir, "prints"))

    paths.append(os.path.join(params.driving.experiment_dir, "single_graphs"))
    if not os.path.exists(os.path.join(params.driving.experiment_dir, "single_graphs")):
        os.makedirs(os.path.join(params.driving.experiment_dir, "single_graphs"))

    paths.append(os.path.join(params.driving.experiment_dir, "full_graph"))
    if not os.path.exists(os.path.join(params.driving.experiment_dir, "full_graph")):
        os.makedirs(os.path.join(params.driving.experiment_dir, "full_graph"))

    if params.driving.mode == "distributed":
        paths.append(os.path.join(params.driving.experiment_dir, "full_graph_chunks"))
        if not os.path.exists(os.path.join(params.driving.experiment_dir, "full_graph_chunks")):
            os.makedirs(os.path.join(params.driving.experiment_dir, "full_graph_chunks"))

    print("Directories set up: ")
    for path in paths:
        print("----  ", path)
    print("\n")

    print("Driving Parameters: ")
    for key, value in vars(params.driving).items():
        print("----  ", key, ":", value)
    print("\n")

    return paths
