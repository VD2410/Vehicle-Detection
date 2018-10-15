import torch
import math

''' Prior Bounding Box  ------------------------------------------------------------------------------------------------
'''
image_height=300
image_width = 300

def generate_prior_bboxes(prior_layer_cfg):
    """
    Generate prior bounding boxes on different feature map level. This function used in 'cityscape_dataset.py'
    Use VGG_SSD 300x300 as example:
    Feature map dimension for each output layers:
       Layer    | Map Dim (h, w) | Single bbox size that covers in the original image
    1. Conv4    | (38x38)        | (30x30) (unit. pixels)
    2. Conv7    | (19x19)        | (60x60)
    3. Conv8_2  | (10x10)        | (111x111)
    4. Conv9_2  | (5x5)          | (162x162)
    5. Conv10_2 | (3x3)          | (213x213)
    6. Conv11_2 | (1x1)          | (264x264)
    NOTE: The setting may be different using MobileNet v3, you have to set your own implementation.
    Tip: see the reference: 'Choosing scales and aspect ratios for default boxes' in original paper page 5.
    :param prior_layer_cfg: configuration for each feature layer, see the 'example_prior_layer_cfg' in the following.
    :return prior bounding boxes with form of (cx, cy, w, h), where the value range are from 0 to 1, dim (1, num_priors, 4)
    """
    example_prior_layer_cfg = [
        # Example:
        {'layer_name': 'Conv4', 'feature_dim_hw': (64, 64), 'bbox_size': (60, 60),
         'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
        {'layer_name': 'Conv4', 'feature_dim_hw': (64, 64), 'bbox_size': (60, 60),
         'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)}
        # ...
        # TODO: define your feature map settings
    ]

    prior_bound_boxes = []
    # init k+1 bbox size to avoid error

    for feat_level_idx in range(0, len(prior_layer_cfg)):
        layer_cfg = prior_layer_cfg[feat_level_idx]
        layer_feature_dim = layer_cfg['feature_dim_hw']
        layer_aspect_ratio = layer_cfg['aspect_ratio']
        bbox_size = layer_cfg['bbox_size']

        # Todo: compute S_{k} (reference: SSD Paper equation 4.)
        sk = bbox_size[0] / image_height
        if feat_level_idx == len(prior_layer_cfg) - 1:
            skplus1 = 1.04
        else:
            layer_cfgplus1 = prior_layer_cfg[feat_level_idx + 1]
            bound_box_1 = layer_cfgplus1['bbox_size']
            skplus1 = bound_box_1[0] / image_height

        fk = layer_feature_dim[0]
        for y in range(0, layer_feature_dim[0]):
            for x in range(0, layer_feature_dim[0]):

                # Todo: compute bounding box center
                cx = (x + 0.5) / fk
                cy = (y + 0.5) / fk

                # Todo: generate prior bounding box with respect to the aspect ratio
                for aspect_ratio in layer_aspect_ratio:
                    if aspect_ratio == 1:
                        h = sk
                        w = sk
                        prior_bound_boxes.append([cx, cy, w, h])
                        h = math.sqrt(sk * skplus1)
                        w = math.sqrt(sk * skplus1)
                        prior_bound_boxes.append([cx, cy, w, h])
                    else:
                        h = sk / math.sqrt(aspect_ratio)
                        w = sk * math.sqrt(aspect_ratio)
                        prior_bound_boxes.append([cx, cy, w, h])

    # Convert to Tensor
    prior_bound_boxes = torch.tensor(prior_bound_boxes)
    prior_bound_boxes = torch.clamp(prior_bound_boxes, 0.0, 1.0)
    num_priors = prior_bound_boxes.shape[0]
    print(num_priors)

    # [DEBUG] check the output shape
    assert prior_bound_boxes.dim() == 2
    assert prior_bound_boxes.shape[1] == 4
    return prior_bound_boxes


def iou_intersect(a, b):
    A = a.size(0)
    B = b.size(0)

    max_xy = torch.min(a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(a[:, :2].unsqueeze(1).expand(A, B, 2),
                       b[:, :2].unsqueeze(0).expand(A, B, 2))

    inter = torch.clamp((max_xy - min_xy), min=0)

    iou_intersect_area = inter[:, :, 0] * inter[:, :, 1]

    return iou_intersect_area


def iou(a: torch.Tensor, b: torch.Tensor):
    """
    # Compute the Intersection over Union
    Note: function iou(a, b) used in match_priors
    :param a: bounding boxes, dim: (n_items, 4)
    :param b: bounding boxes, dim: (n_items, 4) or (1, 4) if b is a reference
    :return: iou value: dim: (n_item)
    """
    # [DEBUG] Check if input is the desire shape
    assert a.dim() == 2
    assert a.shape[1] == 4
    assert b.dim() == 2
    assert b.shape[1] == 4

    # TODO: implement IoU of two bounding box
    # area (A union B) = area(A) + area(B) = area(A iou_intersect B)
    inter = iou_intersect(a, b)
    a_area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]).unsqueeze(1).expand_as(inter)
    b_area = (b[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]).unsqueeze(0).expand_as(inter)
    union = a_area + b_area - inter
    iou = inter / union


    # [DEBUG] Check if output is the desire shape
    assert iou.dim() == 2
    assert iou.shape[0] == a.shape[0]
    return iou


def match_priors(prior_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, gt_labels: torch.Tensor, iou_threshold: float):
    """
    Match the ground-truth boxes with the priors.
    Note: Use this function in your ''cityscape_dataset.py', see the SSD paper page 5 for reference. (note that default box = prior boxes)
    :param gt_bboxes: ground-truth bounding boxes, dim:(n_samples, 4)
    :param gt_labels: ground-truth classification labels, negative (background) = 0, dim: (n_samples)
    :param prior_bboxes: prior bounding boxes on different levels, dim:(num_priors, 4)
    :param iou_threshold: matching criterion
    :return matched_boxes: real matched bounding box, dim: (num_priors, 4)
    :return matched_labels: real matched classification label, dim: (num_priors)
    """
    # [DEBUG] Check if input is the desire shape
    assert gt_bboxes.dim() == 2
    assert gt_bboxes.shape[1] == 4
    assert gt_labels.dim() == 1
    assert gt_labels.shape[0] == gt_bboxes.shape[0]
    assert prior_bboxes.dim() == 2
    assert prior_bboxes.shape[1] == 4

    gt_iou = iou(gt_bboxes, center2corner(prior_bboxes))

    # iou_val, max_idx = gt_iou.max(0, keepdim=True)
    # max_idx.squeeze_(0)
    # iou_val.squeeze_(0)
    # print(iou_val.shape)
    # print(max_idx.shape)
    #
    # matched_boxes = gt_bboxes[max_idx]

    best_prior, best_prior_idx = gt_iou.max(1, keepdim=True)


    best_prior, best_prior_idx = gt_iou.max(0, keepdim=True)


    best_prior_idx.squeeze_(0)
    best_prior.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior.squeeze_(1)
    best_prior_idx.index_fill_(0, best_prior_idx, 2)

    for j in range(best_prior_idx.size(0)):
        best_prior_idx[best_prior_idx[j]] = j
    matched_boxes = gt_bboxes[best_prior_idx]
    variances = [0.1, 0.2]
    cent_x_y = (matched_boxes[:, :2] + matched_boxes[:, 2:]) / 2 - prior_bboxes[:, :2]
    cent_x_y /= variances[0] * prior_bboxes[:, 2:]
    w_h = (matched_boxes[:, 2:] - matched_boxes[:, :2]) / prior_bboxes[:, 2:]
    w_h = torch.log(w_h) / variances[1]

    matched_boxes = torch.cat([cent_x_y, w_h], 1)

    matched_labels = gt_labels[best_prior_idx] + 1
    matched_labels[best_prior < iou_threshold] = 0

    # [DEBUG] Check if output is the desire shape
    assert matched_boxes.dim() == 2
    assert matched_boxes.shape[1] == 4
    assert matched_labels.dim() == 1
    assert matched_labels.shape[0] == matched_boxes.shape[0]

    return matched_boxes, matched_labels


''' NMS ----------------------------------------------------------------------------------------------------------------
'''


def nms_bbox(bbox_loc, bbox_confid_scores, overlap_threshold=0.5, prob_threshold=0.6):
    """
    Non-maximum suppression for computing best overlapping bounding box for a object
    Use this function when testing the samples.
    :param bbox_loc: bounding box loc and size, dim: (num_priors, 4)
    :param bbox_confid_scores: bounding box confidence probabilities, dim: (num_priors, num_classes)
    :param overlap_threshold: the overlap threshold for filtering out outliers
    :param prob_threshold: filter out values
    :return: selected bounding box with classes
    """

    # [DEBUG] Check if input is the desire shape
    # [DEBUG] Check if input is the desire shape
    assert bbox_loc.dim() == 2
    assert bbox_loc.shape[1] == 4
    assert bbox_confid_scores.dim() == 2
    assert bbox_confid_scores.shape[0] == bbox_loc.shape[0]

    sel_bbox = []

    # Todo: implement nms for filtering out the unnecessary bounding boxes

    #convert bboxes from center format to corner format
    bbox_loc = center2corner(bbox_loc)
    num_classes = bbox_confid_scores.shape[1]
    for class_idx in range(0, num_classes):

        # Tip: use prob_threshold to set the prior that has higher scores and filter out the low score items for fast
        # computation
        #filtering scores using probability threshold
        class_score = bbox_confid_scores[:, class_idx]
        filter = class_score > prob_threshold
        mask = filter.unsqueeze_(1).expand_as(class_score)
        prob_fil_scores = class_score[mask]

        pick = []
        left = bbox_loc[:, 0]
        top = bbox_loc[:, 1]
        right = bbox_loc[:, 2]
        bottom = bbox_loc[:, 3]

        #calculating area
        areas = (right - left) * (bottom - top)
        order = prob_fil_scores.argsort()[::-1]

    return sel_bbox


''' Bounding Box Conversion --------------------------------------------------------------------------------------------
'''


def loc2bbox(loc, priors, center_var=0.1, size_var=0.2):
    """
    Compute SSD predicted locations to boxes(cx, cy, h, w).
    :param loc: predicted location, dim: (N, num_priors, 4)
    :param priors: default prior boxes, dim: (1, num_prior, 4)
    :param center_var: scale variance of the bounding box center point
    :param size_var: scale variance of the bounding box size
    :return: boxes: (cx, cy, h, w)
    """
    assert priors.shape[0] == 1
    assert priors.dim() == 3

    # prior bounding boxes
    p_center = priors[..., :2]
    p_size = priors[..., 2:]

    # locations
    l_center = loc[..., :2]
    l_size = loc[..., 2:]

    # real bounding box
    return torch.cat([
        center_var * l_center * p_size + p_center,  # b_{center}
        p_size * torch.exp(size_var * l_size)  # b_{size}
    ], dim=-1)


def bbox2loc(bbox, priors, center_var=0.1, size_var=0.2):
    """
    Compute boxes (cx, cy, h, w) to SSD locations form.
    :param bbox: bounding box (cx, cy, h, w) , dim: (N, num_priors, 4)
    :param priors: default prior boxes, dim: (1, num_prior, 4)
    :param center_var: scale variance of the bounding box center point
    :param size_var: scale variance of the bounding box size
    :return: loc: (cx, cy, h, w)
    """
    assert priors.shape[0] == 1
    assert priors.dim() == 3

    # prior bounding boxes
    p_center = priors[..., :2]
    p_size = priors[..., 2:]

    # locations
    b_center = bbox[..., :2]
    b_size = bbox[..., 2:]

    return torch.cat([
        1 / center_var * ((b_center - p_center) / p_size),
        torch.log(b_size / p_size) / size_var
    ], dim=-1)


def center2corner(center):
    """
    Convert bounding box in center form (cx, cy, w, h) to corner form (x,y) (x+w, y+h)
    :param center: bounding box in center form (cx, cy, w, h)
    :return: bounding box in corner form (x,y) (x+w, y+h)
    """
    return torch.cat([center[..., :2] - center[..., 2:] / 2,
                      center[..., :2] + center[..., 2:] / 2], dim=-1)


def corner2center(corner):
    """
    Convert bounding box from corner form (x,y) (x+w, y+h) to  center form (cx, cy, w, h)
    :param center: bounding box in corner form (x,y) (x+w, y+h)
    :return: bounding box in center form (cx, cy, w, h)
    """
    return torch.cat([corner[..., 2:] + corner[..., :2] / 2,
corner[..., 2:] - corner[..., :2]], dim=-1)
