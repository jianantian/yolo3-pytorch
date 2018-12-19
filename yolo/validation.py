import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils import data

from yolo import model, dataset, utils, detect


def draw_result(images,
                prediction,
                out_dirname,
                epoch_num,
                name,
                max_object_num: int,
                confidence_thresh: float = 0.8,
                iou_thresh: float = 0.4):
    """

    :param images:
    :param prediction:
    :param out_dirname:
    :param epoch_num:
    :param name:
    :param max_object_num:
    :param confidence_thresh:
    :param iou_thresh:
    :return:
    """
    prediction_res = utils.transform_prediction(prediction, confidence_thresh, iou_thresh, max_object_num)
    input_size = (416, 416)
    base_dir = Path(out_dirname)
    if not base_dir.exists():
        base_dir.mkdir()
    images = images.detach().cpu().numpy()
    images = images * 255
    images = np.einsum('...kij->...ijk', images)
    img_dir = base_dir / str(epoch_num)
    if not img_dir.exists():
        img_dir.mkdir()
    for i, (img, pred) in enumerate(zip(images, prediction_res)):
        detect.draw_single_prediction(img, pred, out_filename=img_dir / (str(name[i]) + '.jpg'), input_shape=input_size)


def validate(net,
             val_data,
             out_dirname,
             batch_size,
             epoch_num,
             max_object_num: int,
             confidence_thresh: float,
             iou_thresh: float,
             save_result=False):
    """

    :param net:
    :param val_data:
    :param out_dirname:
    :param batch_size:
    :param epoch_num:
    :param max_object_num:
    :param confidence_thresh:
    :param iou_thresh:
    :param save_result:
    :return:
    """
    net.eval()

    num_class = val_data.class_num

    val_data_iter = data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    use_cuda = net.use_cuda

    total_val_object_num = torch.zeros(num_class, dtype=torch.int)
    total_val_detect_num = torch.zeros(num_class, dtype=torch.int)
    total_val_true_detect_num = torch.zeros(num_class, dtype=torch.int)
    with torch.no_grad():
        for i, (val_data, label, original_size, img_name) in enumerate(val_data_iter):
            if use_cuda:
                val_data = val_data.cuda()
                label = label.cuda()
            prediction, val_count_result = net(val_data, label)
            _object_num, _detect_num, _true_detect_num = val_count_result
            total_val_object_num += _object_num
            total_val_detect_num += _detect_num
            total_val_true_detect_num += _true_detect_num
            if save_result:
                draw_result(val_data, prediction, out_dirname, epoch_num, img_name, max_object_num,
                            confidence_thresh=confidence_thresh, iou_thresh=iou_thresh)

    # val_precision = total_val_true_detect_num.float() / total_val_detect_num.float()
    # val_recall = total_val_true_detect_num.float() / total_val_object_num.float()

    val_mean_precision = torch.sum(total_val_true_detect_num.float()) / torch.sum(total_val_detect_num.float())
    val_mean_recall = torch.sum(total_val_true_detect_num.float()) / torch.sum(total_val_object_num.float())

    validation_res_dct = {'mean_precision': val_mean_precision,
                          'mean_recall': val_mean_recall}
    return validation_res_dct


if __name__ == '__main__':
    config_path = '../config/config.json'
    config = utils.parse_config(config_path)

    confidence_thresh = config.confidence_thresh
    iou_thresh = config.iou_thresh
    kwargs = {"coordinate_rate": config.coordinate_rate,
              "no_object_rate": config.no_object_rate,
              "confidence_thresh": confidence_thresh,
              "iou_thresh": iou_thresh}
    max_object_num = config.max_objects
    class_names = config.class_names

    val_root = '../data/opening_detection/validation'
    val_data = dataset.Dataset(val_root, max_object=max_object_num, augmentation=False)
    num_class = val_data.class_num
    anchor_path = Path('../resource/opening.anchors')

    with open(anchor_path, 'rb') as fr:
        anchors = pickle.load(fr)

    cfg_filename = '../cfg/yolov3.cfg'
    weight_file = '../weight/opening_420.weights'
    yolo = model.load_model(cfg_filename, weight_file, anchors=anchors,
                            class_names=class_names, use_cuda=True, **kwargs)
    validation_result = validate(yolo,
                                 val_data,
                                 '../validation_result',
                                 4,
                                 'val_0',
                                 max_object_num,
                                 confidence_thresh,
                                 iou_thresh,
                                 save_result=True)
    logging.info(f'{validation_result}')
