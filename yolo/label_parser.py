import os
import xml.etree.ElementTree as ElementTree
from pathlib import Path


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
