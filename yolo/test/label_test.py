import os
from pathlib import Path

import cv2
import numpy as np
from torch.utils import data

from yolo import dataset, detect


def draw_label(image, prediction, out_filename):
    """

    :param image:
    :param prediction:
    :param out_filename:
    :return:
    """
    color_list = detect.COLOR_PALETTE
    img = np.ascontiguousarray(image, dtype=np.uint8)
    for record in prediction:
        bbox = record[:4]
        if bbox[1] - bbox[0] > 0:
            color_index = int(record[4])
            detect.draw_box(img, bbox, color=color_list[color_index])
    cv2.imwrite(out_filename, img[:, :, ::-1])


if __name__ == '__main__':
    root = '../data/opening_detection'
    ds = dataset.Dataset(root, transform=None, target_transform=None)
    num_class = ds.class_num
    batch_size = 1
    data_iter = data.DataLoader(ds, batch_size=1, shuffle=False)
    base_filename = Path('../img/res/label_test')
    if not base_filename.exists():
        base_filename.mkdir()
    for i, (img, label) in enumerate(data_iter):
        for b in range(batch_size):
            image = img[b].numpy()
            target = label[b].numpy()
            image_shape = image.shape[:2]
            target[..., :2] *= image_shape[1]
            target[..., 2:4] *= image_shape[0]
            draw_label(image, target, out_filename=os.path.join(base_filename, str(i) + '.png'))
