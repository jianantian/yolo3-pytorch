import numpy as np
import torch
import torch.nn as nn

from yolo import utils


def _content_parse(value: str):
    """

    :param value:
    :return:
    """
    try:
        res = int(value)
    except ValueError:
        try:
            res = float(value)
        except ValueError:
            res = value
    return res


def cfg_parser(cfg_filename: str):
    """

    :param cfg_filename:
    :return:
    """
    with open(cfg_filename, 'r', encoding='utf8') as fr:
        block_list = []
        block = {}
        for _line in fr.readlines():
            line = _line.strip()
            if not line:
                continue
            if line[0] == '[':
                if block:
                    block_list.append(block)
                block = {}
                name = line[1:-1]
                block['name'] = name
            elif line[0] != '#':
                try:
                    k, v = line.split('=')
                except ValueError:
                    pass
                else:
                    block[k.strip()] = _content_parse(v.strip())
        if block:
            block_list.append(block)
    return block_list


class EmptyLayer(nn.Module):
    """

    """

    def __init__(self, start=None, end=None):
        super().__init__()
        self.start = start
        self.end = end


class DetectionLayer(nn.Module):
    """

    """

    def __init__(self, block, num_class, anchors):
        """

        :param block:
        """
        super().__init__()
        self.name = 'detection'
        _mask = block['mask']
        mask_tuple = tuple(int(_x.strip()) for _x in _mask.split(','))
        if anchors is None:
            _anchors = block['anchors']
            _anchors = (_t.strip() for _t in _anchors.split(', '))

            def anchor_getter(_x):
                """

                :param _x:
                :return:
                """
                _x_list = _x.split(',')
                return int(_x_list[0]), int(_x_list[1])

            anchor_tuple = tuple(anchor_getter(_x) for _x in _anchors)
        else:
            anchor_tuple = anchors
        self.mask = mask_tuple
        self.anchors = anchor_tuple
        self.anchor_num = block['num']
        self.class_num = num_class
        # self.class_num = 5
        self.jitter = block['jitter']
        self.ignore_thresh = block['ignore_thresh']
        self.truth_thresh = block['truth_thresh']
        self.random = bool(block['random'])
        self.used_anchors = tuple(anchor_tuple[i] for i in mask_tuple)


def process_detection(x, grid_size, stride, anchors):
    """

    :param x:
    :param grid_size:
    :param stride:
    :param anchors:
    :return:
    """
    device = x.device
    num_anchors = len(anchors)
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)
    x_offset = torch.tensor(a, dtype=torch.float, device=device).reshape(-1, 1)
    y_offset = torch.tensor(b, dtype=torch.float, device=device).reshape(-1, 1)
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).reshape(-1, 2).unsqueeze(0)
    x[..., :2] += x_y_offset

    anchors = [(_a[0] / stride, _a[1] / stride) for _a in anchors]
    anchors = torch.tensor(anchors, dtype=torch.float, device=device)
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    x[..., 2:4] = torch.exp(x[..., 2:4]) * anchors
    x[..., :4] *= stride


def prepare_detection_result(x, img_width, num_class, anchors):
    """

    :param x:
    :param img_width:
    :param num_class:
    :param anchors:
    :return: tensor of shape batch_size * (grid_size * grid_size * num_anchors) * (5 + num_class),
    and the bbox is of shape (center_x, center_y, width, height)
    """
    batch_size, _, width, _ = x.size()
    stride = img_width // width
    grid_size = img_width // stride
    bbox_attr_num = 5 + num_class
    num_anchors = len(anchors)

    x = x.reshape(batch_size, bbox_attr_num * num_anchors, grid_size * grid_size)
    x = x.transpose(1, 2).contiguous()
    x = x.reshape(batch_size, grid_size * grid_size * num_anchors, bbox_attr_num)

    # predict bbox center
    x[..., 0] = torch.sigmoid(x[..., 0])
    x[..., 1] = torch.sigmoid(x[..., 1])

    # object confidence
    x[..., 4] = torch.sigmoid(x[..., 4])

    # class probability
    x[..., 5:] = torch.sigmoid(x[..., 5:])

    return x, grid_size, stride


class NamedLayer(nn.Sequential):
    """

    """

    def __init__(self, name, start=None, end=None, bn=False):
        super().__init__()
        self.name = name
        self.start = start
        self.end = end
        self.bn = bn


class UpsampleLayer(nn.Module):
    """

    """

    def __init__(self, scale_factor, mode='bilinear'):
        """

        :param scale_factor:
        :param mode:
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        """

        :param x:
        :return:
        """
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)


def get_convolutional_layer(block, prev_out_channels, index, num_class=None):
    """

    :param block:
    :param prev_out_channels:
    :param index:
    :param num_class:
    :return:
    """
    in_channels = prev_out_channels
    if num_class is None:
        out_channels = block['filters']
    else:
        out_channels = 3 * (num_class + 5)
    kernel_size = block['size']
    stride = block['stride']
    padding = block['pad']
    if block.get('batch_normalize'):
        bias = False
        batch_normalize = True
    else:
        bias = True
        batch_normalize = False
    if padding:
        padding = (kernel_size - 1) // 2

    model = NamedLayer('convolutional', bn=batch_normalize)
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    model.add_module(f'conv_{index}: ', conv_layer)
    if batch_normalize:
        bn_layer = nn.BatchNorm2d(out_channels)
        model.add_module(f'bn_{index}', bn_layer)

    activation = block.get('activation')
    if activation == 'leaky':
        activation_layer = nn.LeakyReLU(0.1, inplace=True)
        model.add_module(f'activation_{index}', activation_layer)
    return model, out_channels


def get_shortcut_layer(block, index):
    """

    :param block:
    :param index:
    :return:
    """
    from_index = block['from']
    if from_index > 0:
        from_index = from_index - index

    module = NamedLayer('shortcut', start=from_index)
    shortcut_layer = EmptyLayer(start=from_index)
    module.add_module(f'shortcut_{index}', shortcut_layer)
    return module


def get_upsample_layer(block, index):
    """

    :param block:
    :param index:
    :return:
    """
    module = NamedLayer('upsample')
    stride = block['stride']
    upsample_layer = UpsampleLayer(scale_factor=stride, mode='bilinear')
    module.add_module(f'upsample_{index}', upsample_layer)
    return module


def get_route_layer(block, index):
    """

    :param block:
    :param index:
    :return:
    """
    _layers = block['layers']
    if isinstance(_layers, int):
        layers = [_layers]
    else:
        layers = [int(_x.strip()) for _x in _layers.split(',')]

    for i, x in enumerate(layers):
        if x > 0:
            x -= index
            layers[i] = x

    if len(layers) == 2:
        start_index, end_index = layers
    elif len(layers) == 1:
        start_index = layers[0]
        end_index = None
    else:
        raise ValueError(f'Illegal argument layers {_layers}.')
    module = NamedLayer('route', start=start_index, end=end_index)
    route_layer = EmptyLayer(start=start_index, end=end_index)
    module.add_module(f'route_{index}', route_layer)
    return module


def get_detection_layer(block, index, num_class, anchors):
    """

    :param block:
    :param index:
    :param num_class:
    :param anchors:
    :return:
    """
    module = NamedLayer('detection')
    detection_layer = DetectionLayer(block, num_class, anchors)
    module.add_module(f'detection_{index}', detection_layer)
    return module, detection_layer.used_anchors


def get_model_list(block_list, num_class, anchors):
    """
    """
    net_info = block_list[0]
    module_dct = nn.ModuleDict()
    prev_out_channels = 3
    size_list = []

    used_anchor_list = []
    for index, block in enumerate(block_list[1:]):
        name = block['name']
        try:
            next_block = block_list[index + 2]
        except IndexError:
            next_name = None
        else:
            next_name = next_block['name']
        if name == 'convolutional':
            if next_name == 'yolo':
                _c_num = num_class
            else:
                _c_num = None
            module, out_channels = get_convolutional_layer(block,
                                                           prev_out_channels=prev_out_channels,
                                                           index=index,
                                                           num_class=_c_num)
        elif name == 'upsample':
            module = get_upsample_layer(block, index)
            out_channels = prev_out_channels
        elif name == 'shortcut':
            module = get_shortcut_layer(block, index)
            out_channels = prev_out_channels
        elif name == 'route':
            module = get_route_layer(block, index)
            start_index = module.start
            end_index = module.end
            if end_index is not None:
                out_channels = size_list[start_index + index] + size_list[end_index + index]
            else:
                out_channels = size_list[start_index + index]
        elif name == 'yolo':
            module, sub_anchors = get_detection_layer(block, index, num_class, anchors)
            out_channels = prev_out_channels
            used_anchor_list.append(sub_anchors)
        else:
            raise ValueError
        module_name = module.name + '_' + str(index)
        module_dct[module_name] = module
        size_list.append(out_channels)
        prev_out_channels = out_channels
    return net_info, module_dct, used_anchor_list


def read_weights_file(weight_file):
    """

    :param weight_file:
    :return:
    """
    with open(weight_file, 'rb') as fr:
        head = np.fromfile(fr, dtype=np.uint32, count=5)
        weights = np.fromfile(fr, dtype=np.float32)
    return head, weights


class YOLOv3(nn.Module):
    """
    """

    def __init__(self, cfg_file,
                 name_tuple,
                 anchors=None,
                 use_cuda=torch.cuda.is_available(),
                 **kwargs):
        """

        :param cfg_file:
        :param name_tuple:
        :param anchors:
        :param use_cuda:
        :param kwargs:
        """
        super().__init__()
        self.use_cuda = use_cuda
        if self.use_cuda and (not torch.cuda.is_available()):
            raise ValueError('The value of use_cuda is True, but no compatible device find.')
        # self.use_cuda = False
        self.anchors = anchors
        self.name_tuple = name_tuple
        self.num_class = len(name_tuple)
        block_list = cfg_parser(cfg_file)
        net_info, module_dct, used_anchor_list = get_model_list(block_list, self.num_class, anchors)
        self.anchor_tuple = tuple(used_anchor_list)
        self.__net_info = net_info
        self.img_width = net_info['width']
        self.img_height = net_info['height']
        self.head = None
        self.img_count = 0
        self.coordinate_rate = kwargs.get('coordinate_rate', 5)
        self.no_object_rate = kwargs.get('no_object_rate', 0.2)
        self.confidence_thresh = kwargs.get('confidence_thresh', 0.8)
        self.iou_thresh = kwargs.get('iou_thresh', 0.4)
        for name, module in module_dct.items():
            self.add_module(name, module)

        self.localization_loss = nn.MSELoss(reduction='sum')
        self.classification_loss = nn.BCELoss(reduction='sum')
        self.confidence_loss = nn.BCELoss(reduction='sum')
        self.__bn_attrs = ('bias', 'weight', 'running_mean', 'running_var')
        self.__conv_attrs_without_bias = ('weight',)
        self.__conv_attrs_with_bias = ('bias', 'weight')

    def load_weights(self, weight_file):
        """

        :param weight_file:
        :return:
        """
        bn_attrs = self.__bn_attrs
        conv_attrs_without_bias = self.__conv_attrs_without_bias
        conv_attrs_with_bias = self.__conv_attrs_with_bias

        head, weights = read_weights_file(weight_file)

        def _load_singlelayer_parameter(_layer, _attr_tuple, _start_index):
            """

            :param _layer:
            :param _attr_tuple:
            :param _start_index:
            :return:
            """
            for _attr in _attr_tuple:
                _para = getattr(_layer, _attr)
                _para_count = _para.numel()
                _end_index = _start_index + _para_count
                _new_para = torch.from_numpy(weights[_start_index:_end_index])
                _new_para = _new_para.view_as(_para)
                _para.data.copy_(_new_para)
                _start_index = _end_index
            return _start_index

        def _load_layer_parameter(_layer, _start_index):
            """

            :param _layer:
            :param _start_index:
            :return:
            """
            if _layer.name != 'convolutional':
                return _start_index
            _conv_layer = _layer[0]
            if _layer.bn:
                _bn_layer = _layer[1]
                _start_index = _load_singlelayer_parameter(_bn_layer, bn_attrs, _start_index)
                _start_index = _load_singlelayer_parameter(_conv_layer, conv_attrs_without_bias, _start_index)
            else:
                _start_index = _load_singlelayer_parameter(_conv_layer, conv_attrs_with_bias, _start_index)
            return _start_index

        self.head = head
        self.img_count = head[3]
        start_index = 0
        for layer in self.children():
            try:
                name = layer.name
            except AttributeError:
                continue
            if name != 'convolutional':
                continue
            start_index = _load_layer_parameter(layer, start_index)

    def save_weights(self, filename):
        """

        :param filename:
        :return:
        """
        bn_attrs = self.__bn_attrs
        conv_attrs_without_bias = self.__conv_attrs_without_bias
        conv_attrs_with_bias = self.__conv_attrs_with_bias

        def _get_single_layer_parameter(_layer, _attr_tuple):
            """

            :param _layer:
            :param _attr_tuple:
            :return:
            """
            for _attr in _attr_tuple:
                _para = getattr(_layer, _attr)
                _res = _para.cpu().detach_().numpy().astype(np.float32)
                yield _res
                # _res.tofile(fr)

        def _get_layer_parameter(_layer):
            """

            :param _layer:
            :return:
            """
            _conv_layer = _layer[0]
            if _layer.bn:
                _bn_layer = _layer[1]
                # _get_single_layer_parameter(_bn_layer, bn_attrs)
                # _get_single_layer_parameter(_conv_layer, conv_attrs_without_bias)
                yield from _get_single_layer_parameter(_bn_layer, bn_attrs)
                yield from _get_single_layer_parameter(_conv_layer, conv_attrs_without_bias)
            else:
                # _get_single_layer_parameter(_conv_layer, conv_attrs_with_bias)
                yield from _get_single_layer_parameter(_conv_layer, conv_attrs_with_bias)

        head = self.head
        head[3] = self.img_count
        head = np.array(head, dtype=np.uint32)
        with open(filename, 'wb') as fr:
            head.tofile(fr)
            for layer in self.children():
                try:
                    name = layer.name
                except AttributeError:
                    continue
                if name == 'convolutional':
                    # _get_layer_parameter(layer)
                    for para in _get_layer_parameter(layer):
                        para.tofile(fr)

    def forward(self, x, label=None):
        """

        :param x:
        :param label:
        :return:
        """

        is_train = self.training
        # store output from each layer
        output_list = []

        # store detection result from different stage
        prediction_list = []

        loss_list = []
        num_class = self.num_class
        for index, module in enumerate(self.children()):
            try:
                name = module.name
            except AttributeError:
                continue
            if name == 'convolutional' or name == 'upsample':
                x = module(x)
            elif name == 'shortcut':
                start_index = module.start
                x = output_list[index + start_index] + output_list[index - 1]
            elif name == 'route':
                start_index = module.start
                end_index = module.end
                if end_index is None:
                    x = output_list[index + start_index]
                else:
                    x_1 = output_list[index + start_index]
                    x_2 = output_list[index + end_index]
                    x = torch.cat((x_1, x_2), dim=1)
            elif name == 'detection':
                detection_layer = module[0]
                img_width = self.img_width
                num_class = detection_layer.class_num
                anchors = detection_layer.used_anchors
                x, grid_size, stride = prepare_detection_result(x, img_width, num_class, anchors)

                if is_train and label is not None:
                    target = utils.prepare_target(label, anchors, num_class, grid_size, stride)
                    stage_loss = self.loss(x, target)
                    loss_list.append(stage_loss)

                process_detection(x, grid_size, stride, anchors)
                prediction_list.append(x)
            output_list.append(x)

        prediction_res = torch.cat(prediction_list, 1)
        if label is None:
            return prediction_res
        else:
            true_label = label
            true_label[..., :4] *= self.img_width
            prediction = prediction_res.detach().clone()
            prediction_count_result = utils.count_prediction(prediction,
                                                             label,
                                                             num_class,
                                                             confidence_thresh=self.confidence_thresh,
                                                             iou_thresh=self.iou_thresh)
            if is_train:
                local_loss = sum(x[0] for x in loss_list)
                conf_loss = sum(x[1] for x in loss_list)
                class_loss = sum(x[2] for x in loss_list)
                return local_loss, conf_loss, class_loss
            else:
                return prediction_res, prediction_count_result

    def loss(self, prediction, target):
        """

        :param prediction:
        :param target:
        :return:
        """

        localization_loss = self.localization_loss
        confidence_loss = self.confidence_loss
        classification_loss = self.classification_loss

        if self.use_cuda:
            localization_loss = localization_loss.cuda()
            confidence_loss = confidence_loss.cuda()
            classification_loss = classification_loss.cuda()

        mask = (target[..., 4] > 0)
        negative_mask = 1 - mask

        local_loss = localization_loss(prediction[..., :4][mask], target[..., : 4][mask])

        _pos_conf_loss = confidence_loss(prediction[..., 4][mask], target[..., 4][mask])
        _neg_conf_loss = confidence_loss(prediction[..., 4][negative_mask], target[..., 4][negative_mask])
        conf_loss = _pos_conf_loss + self.no_object_rate * _neg_conf_loss

        class_loss = classification_loss(prediction[..., 5:][mask], target[..., 5:][mask])
        batch_num = prediction.shape[0]
        # grid_size = prediction.shape[1]
        # loss = torch.tensor([local_loss / batch_num, conf_loss / batch_num, class_loss / batch_num])
        # loss = local_loss + conf_loss + class_loss
        loss_tuple = (local_loss / batch_num, conf_loss / batch_num, class_loss / batch_num)
        return loss_tuple


def load_model(cfg_filename=None,
               weight_filename=None,
               model_filename=None,
               anchors=None,
               class_names=None,
               use_cuda=False,
               **kwargs):
    """

    :param cfg_filename:
    :param weight_filename:
    :param model_filename:
    :param anchors:
    :param class_names:
    :param use_cuda:
    :param kwargs:
    :return:
    """
    if model_filename is None:
        _, name_tuple = utils.parse_class_names(class_names)
        module = YOLOv3(cfg_filename, name_tuple, anchors, use_cuda=use_cuda, **kwargs)
        if weight_filename is not None:
            module.load_weights(weight_filename)
    else:
        module = torch.load(model_filename)
    if module.use_cuda:
        module = module.cuda()
    return module
