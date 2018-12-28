import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch

from yolo import boundingbox


class Config(object):
    """

    """
    __slots__ = (
        "net_config",
        "net_weight",
        "version",
        "train_root",
        "val_root",
        "trainval_root",
        "anchor_path",
        "weight_dir",
        "output_dir",
        "class_names",
        "epoch",
        "batch_size",
        "max_objects",
        "image_shape",
        "save_interval",
        "val_interval",
        "log_interval",
        "coordinate_rate",
        "no_object_rate",
        "confidence_thresh",
        "iou_thresh",
        "lr",
        "lr_decay",
        "lr_decay_period",
        "lr_decay_epoch",
        "wd")

    def __init__(self, dct):
        """

        :param dct:
        """
        for k in self.__slots__:
            v = dct.get(k, None)
            if k in {"train_root", "val_root", "trainval_root"}:
                v = Path(v)
            setattr(self, k, v)


def parse_config(config_path):
    """

    :param config_path:
    :return:
    """
    with open(config_path, 'r') as fr:
        dct = json.load(fr)
    return Config(dct)


def get_palette(palette_file):
    """
    load color palette
    :param palette_file:
    :return:
    """
    with open(palette_file, 'rb') as fr:
        color_palette = pickle.load(fr)
    return color_palette


def transform_prediction(prediction,
                         confidence_thresh: float,
                         iou_thresh: float,
                         max_object_num: int,
                         need_nms=True):
    """

    :param prediction: 3d torch.tensor of shape (batch_size * some_num * (num_class + 5))
    :param confidence_thresh: bbox confidence thresh value
    :param iou_thresh:
    :param max_object_num:
    :param need_nms:
    :return:
    """
    multiplier = (prediction[..., 4] > confidence_thresh).float().unsqueeze(2)
    prediction *= multiplier
    center_bboxes = prediction[..., :4]
    bboxes = boundingbox.from_center_representation(center_bboxes)
    prediction[..., :4] = bboxes
    batch_size = prediction.shape[0]

    result = np.zeros(shape=(batch_size, max_object_num, min(7, prediction.shape[-1])))
    for i in range(batch_size):
        output_list = []
        img_predict = prediction[i]

        if need_nms:
            max_conf, max_conf_class = torch.max(img_predict[:, 5:], 1)
            max_conf = max_conf.float().unsqueeze(1)
            max_conf_class = max_conf_class.float().unsqueeze(1)

            _seq = (img_predict[:, :5], max_conf, max_conf_class)
            img_predict = torch.cat(_seq, 1)
            non_zero_ind = torch.nonzero(img_predict[:, 4])
            img_predict = img_predict[non_zero_ind.squeeze(), :].reshape(-1, 7)
            if img_predict.shape[0] == 0:
                continue
            class_set = torch.unique(img_predict[:, -1])
            for cls in class_set:
                _cls_mask = img_predict * (img_predict[:, -1] == cls).float().unsqueeze(1)
                _cls_mask_ind = torch.nonzero(_cls_mask[:, -2]).squeeze()
                img_predict_class = img_predict[_cls_mask_ind].reshape(-1, 7)
                img_predict_class = img_predict_class.cpu().detach().numpy()
                img_predict_idx = list(boundingbox.nms(img_predict_class, iou_thresh))
                img_predict_class = img_predict_class[img_predict_idx]
                if img_predict_class.shape[0] > 0:
                    output_list.append(img_predict_class)
        num = len(output_list)
        if num > 0:
            img_predict = np.vstack(output_list)
            predict_shape = img_predict.shape
            if predict_shape[0] < max_object_num:
                patch_targets = np.zeros((max_object_num - predict_shape[0], predict_shape[1]))
                patch_targets[..., -1] = -1
                img_predict = np.concatenate((img_predict, patch_targets))
            result[i, ...] = img_predict
    return result


def count_prediction(prediction,
                     target,
                     num_class,
                     confidence_thresh,
                     iou_thresh):
    """

    :param prediction: np.array of bbox of shape (left, right, bottom, top)
    :param target: np.array of bbox of shape (left, right, bottom, top)
    :param num_class:
    :param confidence_thresh:
    :param iou_thresh:
    :return:
    """

    multiplier = (prediction[..., 4] > confidence_thresh).float().unsqueeze(2)
    prediction *= multiplier
    center_bboxes = prediction[..., :4]
    bboxes = boundingbox.from_center_representation(center_bboxes)
    prediction[..., :4] = bboxes
    batch_size = prediction.shape[0]

    total_object_num = torch.zeros(num_class, dtype=torch.int)
    target_class_label = target[..., -1].int()
    for cls_i in range(num_class):
        _cls_num_t = torch.sum(target_class_label == cls_i)
        total_object_num[cls_i] = _cls_num_t

    total_detect_num = torch.zeros(num_class, dtype=torch.int)
    true_detect_num = torch.zeros(num_class, dtype=torch.int)
    for i in range(batch_size):
        img_predict = prediction[i]
        img_target = target[i]

        max_conf, max_conf_class = torch.max(img_predict[..., 5:], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_class = max_conf_class.float().unsqueeze(1)

        _seq = (img_predict[..., :5], max_conf, max_conf_class)
        img_predict = torch.cat(_seq, 1)
        non_zero_ind = torch.nonzero(img_predict[..., 4])
        img_predict = img_predict[non_zero_ind.squeeze(), ...].reshape(-1, 7)
        if img_predict.shape[0] == 0:
            logging.debug('No prediction bounding box.')
            continue
        class_set = torch.unique(img_predict[..., -1])
        for cls in class_set:
            _cls_mask = img_predict * (img_predict[..., -1] == cls).float().unsqueeze(1)
            _cls_mask_ind = torch.nonzero(_cls_mask[..., -2]).squeeze()
            _img_predict_class = img_predict[_cls_mask_ind].reshape(-1, 7)
            img_predict_idx = list(boundingbox.nms(_img_predict_class, iou_thresh))
            img_predict_idx = torch.tensor(img_predict_idx)
            total_detect_num[int(cls)] += len(img_predict_idx)

            img_predict_class = _img_predict_class[img_predict_idx]
            predict_box = img_predict_class[..., :4]

            _cls_target_ind = (img_target[..., -1] == cls)
            img_target_class = img_target[_cls_target_ind].reshape(-1, 5)
            target_box = img_target_class[..., :4]
            if predict_box.shape[0] > 0 and target_box.shape[0] > 0:
                iou = boundingbox.get_iou(predict_box, target_box)
                predict_cmp_res = torch.max(iou, dim=-1)[0]
                true_detect_num[int(cls)] += int(torch.sum(predict_cmp_res > iou_thresh))

    return total_object_num, total_detect_num, true_detect_num


def prepare_target(target, anchors, num_class, grid_size, stride):
    """
    prepare target for loss computation
    :param target: tensor of bbox like torch.tensor([left, right, bottom, top, label])
    :param anchors:
    :param num_class:
    :param grid_size:
    :param stride:
    :return:
    """
    device = target.device
    num_anchors = len(anchors)
    batch_size = target.size(0)

    y_target = torch.zeros((batch_size, num_anchors * grid_size * grid_size, num_class + 5),
                           dtype=torch.float, device=device)
    anchors = np.array(anchors) / stride
    anchor_boxes = torch.tensor(np.concatenate((np.zeros((num_anchors, 2)), anchors), 1),
                                dtype=torch.float, device=device)
    anchor_boxes = boundingbox.from_center_representation(anchor_boxes)
    anchors = torch.tensor(anchors, dtype=torch.float, device=device)
    valid_mask = (target[..., 4] > -1)

    for b_index in range(batch_size):
        filtered_target = target[b_index, valid_mask[b_index]]
        if len(filtered_target) == 0:
            continue
        # tensor of bbox of shape (left, right, bottom, top)
        true_labels = filtered_target[..., : 4] * grid_size
        center_x = ((true_labels[..., 1] + true_labels[..., 0]).unsqueeze(-1)) / 2
        center_y = ((true_labels[..., 3] + true_labels[..., 2]).unsqueeze(-1)) / 2

        center_box = true_labels.clone()
        center_box[..., :2] -= center_x
        center_box[..., 2:4] -= center_y
        _iou = boundingbox.get_iou(center_box, anchor_boxes)
        best_anchor_index = torch.argmax(_iou, dim=-1)

        center_i = center_x.long()
        center_j = center_y.long()

        class_label = filtered_target[..., 4].unsqueeze(-1).long()
        width = (true_labels[..., 1] - true_labels[..., 0]).unsqueeze(-1)
        height = (true_labels[..., 3] - true_labels[..., 2]).unsqueeze(-1)
        # true_boxes = torch.cat((center_x, center_y, width, height), dim=-1)
        # some filter can be added
        index = num_anchors * (grid_size * center_j + center_i) + best_anchor_index.unsqueeze(-1)
        y_target[b_index, index, 0] = center_x - center_i.float()
        y_target[b_index, index, 1] = center_y - center_j.float()
        y_target[b_index, index, 2] = torch.log(width / anchors[best_anchor_index][..., 0].unsqueeze(-1))
        y_target[b_index, index, 3] = torch.log(height / anchors[best_anchor_index][..., 1].unsqueeze(-1))
        y_target[b_index, index, 4] = 1.0
        y_target[b_index, index, 5 + class_label] = 1
    return y_target


def parse_class_names(class_names):
    """

    :return:
    """
    classname_dct = {}
    name_list = []
    with open(class_names, 'r', encoding='utf8') as fr:
        for i, line in enumerate(fr.readlines()):
            name = line.strip()
            if name:
                classname_dct[name] = i
                name_list.append(name)
    return classname_dct, tuple(name_list)
