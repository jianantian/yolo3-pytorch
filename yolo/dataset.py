import os
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils import data

from yolo import utils


def numpy2tensor(array, dtype='float'):
    """

    :param array:
    :param dtype:
    :return:
    """
    if dtype == 'float':
        array = torch.from_numpy(array.copy()).float()
    elif dtype == 'double':
        array = torch.from_numpy(array.copy()).double()
    elif dtype == 'half':
        array = torch.from_numpy(array.copy()).half()
    else:
        raise ValueError(f'dtype of type {dtype} is not support.')
    return array


def image_loader(img_filename):
    """

    :param img_filename:
    :return:
    """
    if isinstance(img_filename, Path):
        img_filename = os.fspath(img_filename)
    img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
    return img[..., ::-1]


def _resize_img(original_img, input_size, fill_value=0):
    """

    :param original_img:
    :param input_size:
    :param fill_value:
    :return:
    """
    img_h, img_w = original_img.shape[:2]
    w, h = input_size
    scale_rate = min(w / img_w, h / img_h)
    new_w = int(img_w * scale_rate)
    new_h = int(img_h * scale_rate)

    original_img = cv2.resize(original_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((h, w, 3), fill_value=fill_value)
    h_shift = (h - new_h) // 2
    w_shift = (w - new_w) // 2
    canvas[h_shift: h_shift + new_h, w_shift: w_shift + new_w, :] = original_img
    return canvas


def flip_image(img: np.ndarray, hflip: bool = False, vflip: bool = False):
    """

    :param img:
    :param hflip:
    :param vflip:
    :return:
    """
    if hflip:
        img = img[:, ::-1, :]
    if vflip:
        img = img[::-1, :, :]
    return img


def rotate_img(img, angle=90):
    """

    :param img:
    :param angle:
    :return:
    """
    h, w, _ = img.shape
    mat = cv2.getRotationMatrix2D((w / 2, h / 2.), angle, 1)
    return cv2.warpAffine(img, mat, (h, w))


def transform_image(original_img, input_size, fill_value=255, new_axis=False, augmentation=True, dtype='float'):
    """
    transform img to torch.tensor of shape NCWH
    :param original_img:
    :param input_size:
    :param fill_value:
    :param new_axis: bool value, default False
    :param augmentation:
    :param dtype: data type to use: float, double, half
    :return:
    """
    img = _resize_img(original_img, input_size, fill_value=fill_value)
    img = img / 255.

    if augmentation:
        hflip_random = np.random.rand()
        vflip_random = np.random.rand()
        hflip = np.all(hflip_random > 0.5)
        vflip = np.all(vflip_random > 0.5)
        img = flip_image(img, hflip=hflip, vflip=vflip)

        rotate_random = np.random.rand()
        rotate_bool = np.all(rotate_random > 0.5)
        if rotate_bool:
            img = rotate_img(img)
        transform_dct = {'hflip': hflip,
                         'vflip': vflip,
                         'rotate': rotate_bool}
    else:
        transform_dct = {'hflip': False,
                         'vflip': False,
                         'rotate': False}
    img = np.einsum('ijk->kij', img)
    if new_axis:
        img = img[np.newaxis, ...]
    return numpy2tensor(img, dtype=dtype), transform_dct


def _parse_target(label):
    """
    parse a single target from xml to get bounding box
    :param label:
    :return:
    """
    name = label.findtext('name')
    bbox = label.find('bndbox')
    left = int(bbox.findtext('xmin'))
    right = int(bbox.findtext('xmax'))
    bottom = int(bbox.findtext('ymin'))
    top = int(bbox.findtext('ymax'))
    return left, right, bottom, top, name


def label_loader(xml_filename):
    """
    parse label from xml to get bounding box iterator
    :param xml_filename:
    :return:
    """
    if isinstance(xml_filename, Path):
        xml_filename = os.fspath(xml_filename)
    tree = ElementTree.ElementTree(file=xml_filename)
    for x in tree.iter(tag='object'):
        yield _parse_target(x)


def resize_bbox(bbox, bbox_image_size, out_image_size, is_train=True):
    """
    :param bbox: of shape (left, right, bottom, top)
    :param bbox_image_size:
    :param out_image_size:
    :param is_train:
    :return:
    """
    h, w = bbox_image_size
    img_h, img_w = out_image_size
    if is_train:
        scale_rate = min(img_w / float(w), img_h / float(h))
    else:
        scale_rate = max(img_w / float(w), img_h / float(h))
    shift_w = (img_w - w * scale_rate) / 2
    shift_h = (img_h - h * scale_rate) / 2
    bbox[..., : 2] *= scale_rate
    bbox[..., : 2] += shift_w

    bbox[..., 2:4] *= scale_rate
    bbox[..., 2:4] += shift_h

    bbox[..., :2] = np.clip(bbox[..., :2], 0, img_w)
    bbox[..., 2:4] = np.clip(bbox[..., 2:4], 0, img_h)

    return bbox.astype(np.int32)


def flip_bbox(bbox, image_shape, hflip=False, vflip=False):
    """

    :param bbox: of shape (left, right, bottom, top)
    :param image_shape:
    :param hflip:
    :param vflip:
    :return:
    """
    try:
        h, w = image_shape
    except TypeError:
        h = image_shape
        w = image_shape

    if hflip:
        bbox[..., :2] = (w - bbox[..., : 2])[..., ::-1]

    if vflip:
        bbox[..., 2:4] = (h - bbox[..., 2:4])[..., ::-1]

    return bbox


def rotate_bbox(bbox, image_shape, rotate: bool = False):
    """

    :param bbox: of shape (left, right, bottom, top)
    :param image_shape:
    :param rotate
    :return: of shape (left, right, bottom, top)
    """
    try:
        h, w = image_shape
    except TypeError:
        h = image_shape
        w = image_shape

    if rotate:
        mat = np.array([[0, -1.], [1., 0]], dtype=np.float32)
        center = np.array([h / 2., w / 2.])

        pt_0 = np.array([bbox[..., 0], bbox[..., 2]])
        pt_1 = np.array([bbox[..., 1], bbox[..., 3]])

        new_pt_0 = np.einsum('ij, i...', mat, pt_0 - center.reshape(2, 1)) + center
        new_pt_1 = np.einsum('ij, i...', mat, pt_1 - center.reshape(2, 1)) + center

        left = np.minimum(new_pt_0[..., 0], new_pt_1[..., 0])
        right = np.maximum(new_pt_0[..., 0], new_pt_1[..., 0])

        bottom = np.minimum(new_pt_0[..., 1], new_pt_1[..., 1])
        top = np.maximum(new_pt_0[..., 1], new_pt_1[..., 1])

        return np.concatenate((left, right, bottom, top)).reshape(4, -1).transpose(1, 0)
    else:
        return bbox


def transform_label(label, bbox_image_size, out_image_size, dtype='float', **kwargs):
    """

    :param label:
    :param bbox_image_size:
    :param out_image_size:
    :param dtype:
    :return:
    """
    bbox = label[..., :4]
    resize_bbox(bbox, bbox_image_size, out_image_size, is_train=True)
    bbox = flip_bbox(bbox, out_image_size, hflip=kwargs['hflip'], vflip=kwargs['vflip'])
    bbox = rotate_bbox(bbox, out_image_size, rotate=kwargs['rotate'])
    label[..., : 4] = bbox
    label[..., :2] /= out_image_size[1]
    label[..., 2:4] /= out_image_size[0]
    return numpy2tensor(label, dtype=dtype)


def get_data_statistics(train_dataset):
    """
    calculate mean and std of dataset
    :param train_dataset:
    :return:
    """
    r_sum, g_sum, b_sum = 0, 0, 0
    r_square_sum, g_square_sum, b_square_sum = 0, 0, 0

    for img, _, _, _ in train_dataset:
        r_sum += torch.sum(img[0, :, :])
        g_sum += torch.sum(img[1, :, :])
        b_sum += torch.sum(img[2, :, :])

        r_square_sum += torch.sum(img[0, :, :] ** 2)
        g_square_sum += torch.sum(img[1, :, :] ** 2)
        b_square_sum += torch.sum(img[2, :, :] ** 2)

    num = len(train_dataset) * 416 * 416

    r_mean = r_sum / num
    g_mean = g_sum / num
    b_mean = b_sum / num

    print(g_square_sum)
    r_square_mean = r_square_sum / num
    g_square_mean = g_square_sum / num
    b_square_mean = b_square_sum / num

    r_std = torch.sqrt(r_square_mean - r_mean * r_mean)
    g_std = torch.sqrt(g_square_mean - g_mean * g_mean)
    b_std = torch.sqrt(b_square_mean - b_mean * b_mean)

    data_mean = (r_mean.item(), g_mean.item(), b_mean.item())
    data_std = (r_std.item(), g_std.item(), b_std.item())
    return data_mean, data_std


class Dataset(data.Dataset):
    """

    """

    def __init__(self,
                 root,
                 image_shape=416,
                 dtype='float',
                 data_loader=image_loader,
                 label_loader=label_loader,
                 transform=transform_image,
                 target_transform=transform_label,
                 max_object=None,
                 augmentation=True):
        """

        :param root:
        :param image_shape:
        :param dtype:
        :param data_loader:
        :param label_loader:
        :param transform:
        :param target_transform:
        :param augmentation:
        """
        self.root = Path(root)
        self.data_dirname = self.root / 'data'
        self.label_dirname = self.root / 'label'
        self.class_names = self.root / 'class.names'
        self.image_shape = (image_shape, image_shape)
        self.dtype = dtype
        self.label_dct = self.__get_label_dct()
        self.class_dct = self.__get_class_dct()
        self.class_num = len(self.class_dct)
        self.data = list(self.data_dirname.iterdir())
        self.data_loader = data_loader
        self.label_loader = label_loader
        self.transform = transform
        self.target_transform = target_transform
        self.augmentation = augmentation
        self.max_object = max_object

    def __get_label_dct(self):
        """

        :return:
        """
        res = {}
        for filename in self.label_dirname.iterdir():
            k = filename.stem
            v = filename
            res[k] = v
        return res

    def __get_class_dct(self):
        """

        :return:
        """
        classname_dct, _ = utils.parse_class_names(self.class_names)
        return classname_dct

    def __len__(self):
        """

        :return:
        """
        return len(self.data)

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        img_filename = self.data[index]
        img_name = img_filename.stem
        label_filename = self.label_dct[img_name]
        original_image = self.data_loader(img_filename)
        original_size = original_image.shape[:2]
        class_dct = self.class_dct
        _target = []
        for label in self.label_loader(label_filename):
            name = label[-1]
            _target.append([*label[:4], class_dct[name]])
        original_target = np.array(_target, dtype=np.float32)

        image, transform_dct = self.transform(original_image,
                                              self.image_shape,
                                              augmentation=self.augmentation,
                                              dtype=self.dtype)

        target = self.target_transform(original_target,
                                       original_size,
                                       self.image_shape,
                                       dtype=self.dtype,
                                       **transform_dct)

        # patch the target to same shape of self.max_object * 5
        target_shape = target.shape
        if self.max_object is not None:
            if target_shape[0] < self.max_object:
                patch_targets = torch.zeros((self.max_object - target_shape[0], target_shape[1]))
                patch_targets[..., -1] = -2
                target = torch.cat((target, patch_targets))
            elif target_shape[0] > self.max_object:
                target = target[:self.max_object, ...]
        return image, target, original_size, img_name
