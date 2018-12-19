"""generate 9 anchors"""

import logging
import pickle
import random

import numpy as np
from torch.utils import data

from yolo import dataset


def prepare_data(root_dir: str) -> np.ndarray:
    """
    get bounding box data from label directory
    :param root_dir: root directory for train data
    :return:
    """
    ds = dataset.Dataset(root_dir, augmentation=False)
    image_shape = ds.image_shape
    data_iter = data.DataLoader(ds, batch_size=1, shuffle=False)

    bbox_list = []
    for i, (_, label, _, _) in enumerate(data_iter):
        bbox = (label[..., :4][label[..., 4] > -1]).detach()
        bbox_list.append(bbox)
    bbox_array = np.concatenate(bbox_list)
    width = bbox_array[..., 1] - bbox_array[..., 0]
    height = bbox_array[..., 3] - bbox_array[..., 2]
    return np.vstack((width, height)).T * image_shape[0]


def iou_distance(bbox: np.ndarray, bbox_array: np.ndarray) -> np.ndarray:
    """
    calculate iou, ignore bounding box position
    :param bbox: np.array of shape (num, 2) or (2, )
    :param bbox_array:  np.array of shape (num, 2)
    :return:
    """
    width = bbox[..., 0]
    height = bbox[..., 1]
    area = width * height

    b_width = bbox_array[..., 0]
    b_height = bbox_array[..., 1]
    b_width = np.expand_dims(b_width, axis=-1)
    b_height = np.expand_dims(b_height, axis=-1)
    b_area = b_width * b_height
    intersection_area = np.minimum(b_width, width) * np.minimum(b_height, height)
    return intersection_area / (area + b_area - intersection_area)


def initial_centroid(train_data: np.ndarray, cluster_num: int = 9) -> np.ndarray:
    """
    initial centroid by kmeans++
    :param train_data: np.array of shape (data_num, 2)
    :param cluster_num: cluster number
    :return:
    """
    data_num = train_data.shape[0]
    start_index = random.randint(0, data_num)
    centroid_list = [train_data[start_index]]
    for _ in range(cluster_num - 1):
        centroid = np.array(centroid_list)
        distance = 1 - iou_distance(train_data, centroid)
        min_distance = np.min(distance, axis=0)
        sum_distance = np.sum(min_distance)
        distance_thresh = sum_distance * random.random()

        cul_dis = 0
        for i, bbox in enumerate(train_data):
            cul_dis += min_distance[i]
            if cul_dis > distance_thresh:
                centroid_list.append(bbox)
                break
    return np.array(centroid_list)


def cluster_kmeans(train_data: np.ndarray, cluster_num: int = 9, initial_method: str = 'random'):
    """
    kmeans
    :param train_data: np.array of shape (data_num, 2)
    :param cluster_num: cluster number
    :param initial_method: 'random' or 'kmean++'
    :return:
    """
    data_num, data_dim = train_data.shape

    if initial_method == 'random':
        _ind = np.random.choice(data_num, cluster_num, replace=False)
        centroid = train_data[_ind]
    elif initial_method == 'kmean++':
        centroid = initial_centroid(train_data, cluster_num)
    else:
        raise ValueError(f'initial method {initial_method} is non available.')

    iter_num = 0
    prev_assignment = np.ones(data_num) * (-1)
    prev_distance = np.zeros((cluster_num, data_num))
    while True:
        distance = 1 - iou_distance(train_data, centroid)
        assignment = np.argmin(distance, axis=0)
        if (assignment == prev_assignment).all():
            break
        logging.debug("iteration {}: distance = {}".format(iter_num, np.sum(np.abs(prev_distance - distance))))
        for i in range(cluster_num):
            index = (assignment == i)
            centroid[i] = np.mean(train_data[index], axis=0)
        prev_assignment = assignment.copy()
        prev_distance = distance.copy()
        iter_num += 1

    centroid = centroid.astype(np.int32)
    ind_sort = np.argsort(centroid[..., 0])
    anchors = tuple(tuple(centroid[i]) for i in ind_sort)
    return anchors, assignment


def generate_anchors(root_dir, filename, initial_method='kmean++'):
    """

    :param root_dir: data root directory
    :param filename: anchors file save directory
    :param initial_method: initial method: 'kmean++' or 'random'
    :return:
    """
    train_data = prepare_data(root_dir)
    centroid, _ = cluster_kmeans(train_data, initial_method=initial_method)
    with open(filename, 'wb') as fr:
        pickle.dump(centroid, fr)


if __name__ == '__main__':
    root_dir = '../data/opening_detection/train'
    filename = '../resource/opening.anchors.bak'
    train_data = prepare_data(root_dir)
    generate_anchors(root_dir, filename)

