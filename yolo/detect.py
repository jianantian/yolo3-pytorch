import json
import os
import pickle
import random
import time
from pathlib import Path

import cv2
import numpy as np
from torch import autograd

from yolo import dataset, model, utils

COLOR_PALETTE = utils.get_palette(Path(os.path.abspath(__file__)).parents[1] / 'resource' / 'palette')


def data_iter(img_dirname):
    """
    iter img_file in directory img_dirname
    :param img_dirname:
    :return:
    """
    img_dir = Path(img_dirname)
    if not img_dir.exists():
        raise ValueError(f'{img_dirname} not exist!')
    if img_dir.is_file():
        yield img_dir
    else:
        for filename in img_dir.iter():
            yield filename


def load_classes(filename):
    """

    :param filename:
    :return:
    """
    with open(filename, 'r', encoding='utf8') as fr:
        for line in fr.readlines():
            s = line.strip()
            if s:
                yield s


def draw_box(img, bbox, color=None):
    """

    :param img:
    :param bbox:
    :param color:
    :return:
    """
    bbox = bbox.astype(np.int32)
    pt_0 = (bbox[0], bbox[2])
    pt_1 = (bbox[1], bbox[3])
    if color is None:
        c_ind = random.randint(0, len(COLOR_PALETTE))
        color = COLOR_PALETTE[c_ind % len(COLOR_PALETTE)]
        if np.issubdtype(img.dtype, np.floating):
            color = (color[0] / 255., color[1] / 255., color[2] / 255.)
    cv2.rectangle(img, pt_0, pt_1, color, 2)


def draw_single_prediction(image, prediction, out_filename, input_shape=None):
    """

    :param image:
    :param prediction:
    :param out_filename:
    :param input_shape:
    :return:
    """
    img = np.ascontiguousarray(image, dtype=np.uint8)
    original_size = img.shape[:2]
    for record in prediction:
        bbox = record[:4]
        if (bbox[1] - bbox[0] > 0) and (bbox[3] - bbox[2] > 0):
            if input_shape is not None:
                bbox = bbox.copy()
                new_bbox = dataset.resize_bbox(bbox, input_shape, original_size, is_train=False)
                draw_box(img, new_bbox)
            else:
                draw_box(img, bbox)
    cv2.imwrite(os.fspath(out_filename), img[:, :, ::-1])


def get_detect_result(prediction, original_size, input_shape=None):
    """

    :param prediction:
    :param original_size:
    :param input_shape:
    :return:
    """
    for record in prediction:
        bbox = record[:4]
        label = int(record[-1])
        object_conf = record[4]
        class_prob = record[5]
        if (bbox[1] - bbox[0] > 0) and (bbox[3] - bbox[2] > 0):
            if input_shape is not None:
                bbox = bbox.copy()
                new_bbox = dataset.resize_bbox(bbox, input_shape, original_size, is_train=False)
                yield new_bbox, label, object_conf, class_prob
            else:
                yield bbox, label, object_conf, class_prob


def get_detection_json(prediction, original_size, name_tuple, input_shape=None):
    """

    :param prediction:
    :param original_size:
    :param name_tuple:
    :param input_shape:
    :return:
    """
    res = []
    for box, label_ind, object_conf, class_prob in get_detect_result(prediction, original_size,
                                                                     input_shape=input_shape):
        topleft = {'x': int(box[0]), 'y': int(box[2])}
        bottomright = {'x': int(box[1]), 'y': int(box[3])}

        data = {'label': name_tuple[label_ind],
                'confidence': round(object_conf),
                'topleft': topleft,
                'bottomright': bottomright}

        res.append(data)
    return res


def detect(img_filename, config_path, out_dirname):
    """

    :param img_filename:
    :param config_path:
    :param out_dirname
    :return:
    """
    config = utils.parse_config(config_path)

    max_object_num = config.max_objects
    confidence_thresh = config.confidence_thresh
    iou_thresh = config.iou_thresh
    kwargs = {"confidence_thresh": confidence_thresh,
              "iou_thresh": iou_thresh}
    with open(config.anchor_path, 'rb') as fr:
        train_anchors = pickle.load(fr)

    cfg_filename = config.net_config
    weight_file = config.net_weight
    class_names = config.class_names

    start_time = time.time()
    net = model.load_model(cfg_filename, weight_file, anchors=train_anchors,
                           class_names=class_names, use_cuda=True, **kwargs)
    net.eval()

    model_load_time = time.time() - start_time
    predict_start_time = time.time()
    original_img = dataset.image_loader(img_filename)
    input_size = (config.image_shape, config.image_shape)
    img, _ = dataset.transform_image(original_img, input_size=input_size, new_axis=True, augmentation=False)
    if net.use_cuda:
        img = img.cuda()

    with autograd.no_grad():
        prediction = net(img)

    name = Path(img_filename).stem
    base_dir = Path(out_dirname)
    out_img = base_dir / (name + '_res.jpg')
    out_json = base_dir / (name + '_res.json')
    _, name_tuple = utils.parse_class_names(class_names)
    original_size = original_img.shape[:2]
    prediction_result = utils.transform_prediction(prediction, confidence_thresh, iou_thresh, max_object_num)
    res_dct = get_detection_json(prediction_result[0], original_size, name_tuple, input_size)

    with open(out_json, 'w', encoding='utf8') as fr:
        json.dump(res_dct, fr, ensure_ascii=False)
    draw_single_prediction(original_img, prediction_result[0], out_img, input_shape=input_size)
    print(f'total_time: {time.time() - start_time},'
          f' model_load_time = {model_load_time}, '
          f'infer_time = {time.time() - predict_start_time}')


if __name__ == '__main__':
    config_path = '../config/config.json'
    img_filename = '../data/opening_detection/validation/data/pic_54.jpg'
    detect(img_filename, config_path, '../img')
