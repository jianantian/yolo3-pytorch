import numpy as np
import torch


def get_area(bbox):
    """

    :param bbox: torch.tensor([left, right, bottom, top]) or np.array([left, right, bottom, top])
    :return:
    """
    return (bbox[..., 1] - bbox[..., 0] + 1) * (bbox[..., 3] - bbox[..., 2] + 1)


def from_center_representation(center_bboxes):
    """
    transform bbox from center representation to border representation
    :param center_bboxes: tensor of bbox like torch.tensor([center_x, center_y, length, width])
    :return: tensor of bbox like torch.tensor([left, right, bottom, top])
    """
    shape = center_bboxes.shape
    batch_size = center_bboxes.shape[0]
    x_center = center_bboxes[..., 0]
    y_center = center_bboxes[..., 1]
    half_length = center_bboxes[..., 2] / 2
    half_width = center_bboxes[..., 3] / 2
    left = x_center - half_length
    right = x_center + half_length
    bottom = y_center - half_width
    top = y_center + half_width

    _seq = (left, right, bottom, top)
    if len(shape) == 3:
        return torch.cat(_seq, 1).reshape(batch_size, 4, -1).transpose(2, 1)
    elif len(shape) == 2:
        return torch.cat(_seq, 0).reshape(4, -1).transpose(1, 0)
    else:
        return torch.Tensor(_seq)


def to_center_representation(bboxes):
    """
    transform bbox from center representation to border representation
    :param bboxes: np.array of bbox like torch.tensor([left, right, bottom, top])
    :return: np.array of bbox like torch.tensor([center_x, center_y, length, width])
    """
    shape = bboxes.shape
    batch_size = bboxes.shape[0]

    left = bboxes[..., 0]
    right = bboxes[..., 1]
    bottom = bboxes[..., 2]
    top = bboxes[..., 3]

    x_center = (left + right) / 2
    y_center = (bottom + top) / 2

    length = right - left
    width = top - bottom

    _seq = (x_center, y_center, length, width)

    if len(shape) == 3:
        return np.concatenate(_seq, 1).reshape(batch_size, 4, -1).transpose(2, 1)
    elif len(shape) == 2:
        return np.concatenate(_seq, 0).reshape(4, -1).transpose(1, 0)
    else:
        return np.array(_seq)


def intersection_area(base_bbox, bbox_array: np.ndarray):
    """
    area of intersection bbox
    :param base_bbox: 1d np.array([left, right, bottom, top])
    :param bbox_array: 2d np.array(list of bbox)
    :return:
    """
    # ew is short for element wise
    if isinstance(base_bbox, np.ndarray):
        _ew_max_func = np.maximum
        _ew_min_func = np.minimum
        min_value = 0.0
    elif isinstance(base_bbox, torch.Tensor):
        _ew_max_func = torch.max
        _ew_min_func = torch.min
        min_value = torch.tensor([0.0], device=base_bbox.device).float()
    else:
        raise TypeError(f'THe type of argument base_box: {type(base_bbox)} is illegal.')
    if isinstance(base_bbox, np.ndarray):
        left = _ew_max_func(np.expand_dims(base_bbox[..., 0], -1), bbox_array[..., 0])
        right = _ew_min_func(np.expand_dims(base_bbox[..., 1], -1), bbox_array[..., 1])
        bottom = _ew_max_func(np.expand_dims(base_bbox[..., 2], -1), bbox_array[..., 2])
        top = _ew_min_func(np.expand_dims(base_bbox[..., 3], -1), bbox_array[..., 3])
    elif isinstance(base_bbox, torch.Tensor):
        left = _ew_max_func(base_bbox[..., 0].unsqueeze(-1), bbox_array[..., 0])
        right = _ew_min_func(base_bbox[..., 1].unsqueeze(-1), bbox_array[..., 1])
        bottom = _ew_max_func(base_bbox[..., 2].unsqueeze(-1), bbox_array[..., 2])
        top = _ew_min_func(base_bbox[..., 3].unsqueeze(-1), bbox_array[..., 3])
    else:
        raise ValueError
    w = _ew_max_func(min_value, right - left + 1)
    h = _ew_max_func(min_value, top - bottom + 1)

    return w * h


def get_iou(base_bbox, bbox_array: np.ndarray):
    """

    :param base_bbox:
    :param bbox_array:
    :return:
    """
    left = bbox_array[..., 0]
    right = bbox_array[..., 1]
    bottom = bbox_array[..., 2]
    top = bbox_array[..., 3]
    inter_area = intersection_area(base_bbox, bbox_array)
    areas = (right - left + 1) * (top - bottom + 1)
    base_area = get_area(base_bbox).unsqueeze(-1)
    iou = inter_area / (base_area + areas - inter_area)
    return iou


def nms(bboxes, thresh):
    """
    Non-Maximum Suppression
    Inspired by the code written by Ross Girshick for Fast R-CNN, the original code is MIT licensed.
    :param bboxes: 2d np.array(list of np.array([left, right, bottom, top]))
    :param thresh: thresh value, float type
    :return:
    """
    try:
        left = bboxes[..., 0]
    except IndexError:
        return []
    right = bboxes[..., 1]
    bottom = bboxes[..., 2]
    top = bboxes[..., 3]
    scores = bboxes[..., 4]  # sorted by confidence score

    areas = (right - left + 1) * (top - bottom + 1)
    try:
        idx = scores.argsort()
    except AttributeError:
        _, idx = torch.sort(scores)

    if isinstance(bboxes, np.ndarray):
        argument_type = 'numpy'
        device = 'cpu'
    elif isinstance(bboxes, torch.Tensor):
        argument_type = 'torch'
        device = bboxes.device
    else:
        raise TypeError(f'THe type of argument bboxes: {type(bboxes)} is illegal.')

    while len(idx) > 0:
        i = idx[-1]
        yield i
        if len(idx) <= 1:
            idx = []
            continue
        _area = intersection_area(bboxes[i], bboxes[idx[:-1]])
        iou = _area / (areas[i] + areas[idx[:-1]] - _area)
        if argument_type == 'numpy':
            remain_idx = np.where(iou <= thresh)[0]
        else:
            remain_idx = torch.where(iou <= thresh,
                                     torch.tensor([1], dtype=torch.uint8, device=device),
                                     torch.tensor([0], dtype=torch.uint8, device=device)).nonzero().squeeze(-1)
        idx = idx[remain_idx]
