import pickle

import cv2
import flask
import numpy as np
from torch import autograd

from yolo import utils, dataset, model, detect

app = flask.Flask(__name__)

net = None
config = None


def model_load_server():
    """
    load model
    :return:
    """
    global net
    global config

    config_path = '../config/config.json'
    config = utils.parse_config(config_path)

    confidence_thresh = config.confidence_thresh
    iou_thresh = config.iou_thresh
    kwargs = {"confidence_thresh": confidence_thresh,
              "iou_thresh": iou_thresh}
    with open(config.anchor_path, 'rb') as fr:
        train_anchors = pickle.load(fr)

    cfg_filename = config.net_config
    weight_file = config.net_weight
    class_names = config.class_names

    net = model.load_model(cfg_filename, weight_file, anchors=train_anchors,
                           class_names=class_names, use_cuda=True, **kwargs)
    net.eval()


@app.route('/detection', methods=['POST'])
def detection():
    """

    :return:
    """
    data = {'success': False}
    if flask.request.method == 'POST':
        if flask.request.files.get('image'):
            input_size = (config.image_shape, config.image_shape)
            max_object_num = config.max_objects
            class_names = config.class_names
            confidence_thresh = config.confidence_thresh
            iou_thresh = config.iou_thresh

            _img_bytes = flask.request.files['image'].read()
            _img_craft = np.frombuffer(_img_bytes, np.uint8)
            original_img = cv2.imdecode(_img_craft, cv2.IMREAD_COLOR)

            img, _ = dataset.transform_image(original_img, input_size=input_size, new_axis=True, augmentation=False)
            if net.use_cuda:
                img = img.cuda()

            with autograd.no_grad():
                prediction = net(img)

            _, name_tuple = utils.parse_class_names(class_names)
            original_size = original_img.shape[:2]
            prediction_result = utils.transform_prediction(prediction, confidence_thresh, iou_thresh, max_object_num)
            res_dct = detect.get_detection_json(prediction_result[0], original_size, name_tuple, input_size)
            data['result'] = res_dct
            data['success'] = True
    return flask.jsonify(data)


if __name__ == '__main__':
    model_load_server()
    app.run(host='0.0.0.0', port=3456)
