import logging
import pickle
import random
import time
from pathlib import Path

import torch
from torch.utils import data

from yolo import model, dataset, anchor, visualize, validation, utils

logging.basicConfig(format='%(levelname)s: %(asctime)s %(message)s', level=logging.INFO)


def train(config, name='opening'):
    """

    :param config:
    :param name:
    :return:
    """
    train_root = config.train_root
    val_root = config.val_root
    trainval_root = Path(config.trainval_root)

    filename_list = list(f.name for f in (trainval_root / 'data').iterdir())

    anchor_path = Path(config.anchor_path)
    if not anchor_path.exists():
        anchor.generate_anchors(train_root, anchor_path)
    with open(anchor_path, 'rb') as fr:
        train_anchors = pickle.load(fr)

    weight_dir = Path(config.weight_dir)
    if not weight_dir.exists():
        weight_dir.mkdir(parents=True)

    out_dirname = Path(config.output_dir)
    if not out_dirname.exists():
        out_dirname.mkdir(parents=True)

    epoch = config.epoch
    batch_size = config.batch_size
    max_object_num = config.max_objects

    # train_ds = dataset.Dataset(train_root, max_object=max_object_num, augmentation=True)
    # val_ds = dataset.Dataset(val_root, max_object=max_object_num, augmentation=False)
    # test_ds = dataset.Dataset(train_root, max_object=max_object_num, augmentation=False)

    cfg_filename = config.net_config
    weight_file = None
    class_names = config.class_names

    confidence_thresh = config.confidence_thresh
    iou_thresh = config.iou_thresh
    kwargs = {"coordinate_rate": config.coordinate_rate,
              "no_object_rate": config.no_object_rate,
              "confidence_thresh": confidence_thresh,
              "iou_thresh": iou_thresh}

    yolo = model.load_model(cfg_filename, weight_file, anchors=train_anchors,
                            class_names=class_names, use_cuda=True, **kwargs)

    version = [x.strip() for x in config.version.split('.')]
    img_count = int(0.8 * len(filename_list))
    head = version + [img_count, 0]
    yolo.head = head

    learning_rate = config.lr
    weight_decay = config.wd

    optimizer = torch.optim.Adam(yolo.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # train_data_iter = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    use_cuda = yolo.use_cuda

    visualizer = visualize.Visualizer(name)

    val_interval = config.val_interval
    save_interval = config.save_interval
    log_interval = config.log_interval

    lr_decay_epoch = set(config.lr_decay_epoch)
    lr_decay = config.lr_decay

    best_f1 = 0
    best_filename = name + '_' + 'best.weights'
    saved_count = 0
    for e in range(epoch):
        need_save = False
        start_time = time.time()
        yolo.train()

        random.shuffle(filename_list)

        train_filenames = filename_list[:img_count]
        val_filenames = filename_list[img_count:]
        train_ds = dataset.Dataset(train_root, filenames=train_filenames, max_object=max_object_num, augmentation=True)
        val_ds = dataset.Dataset(val_root, filenames=val_filenames, max_object=max_object_num, augmentation=False)
        test_ds = dataset.Dataset(train_root, filenames=val_filenames, max_object=max_object_num, augmentation=False)

        train_data_iter = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        for i, (train_data, label, original_size, _) in enumerate(train_data_iter):
            if use_cuda:
                train_data = train_data.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            loss = yolo(train_data, label)
            total_loss = yolo.coordinate_rate * loss[0] + loss[1] + loss[2]

            total_loss.backward()
            optimizer.step()
            # logging.info(f'batch num: {i}, train_loss: {loss.item()} \n')

        if e > 1 and e % log_interval == 0:
            logging.info(f'epoch num: {e}')
            loss_dct = {'position_loss': loss[0].item(),
                        'confidence_loss': loss[1].item(),
                        'classification_loss': loss[2].item()}
            logging.info(loss_dct)
            visualizer.plot(loss_dct)

        if e > 1 and e % val_interval == 0:
            if e > 1 and e % save_interval == 0:
                need_save = True

            validation_result = validation.validate(yolo,
                                                    val_ds,
                                                    out_dirname,
                                                    batch_size,
                                                    'val_' + str(e),
                                                    max_object_num,
                                                    confidence_thresh,
                                                    iou_thresh,
                                                    save_result=need_save)
            train_result = validation.validate(yolo,
                                               test_ds,
                                               out_dirname,
                                               batch_size,
                                               'train_' + str(e),
                                               max_object_num,
                                               confidence_thresh,
                                               iou_thresh,
                                               save_result=need_save)

            try:
                f1_score = ((2 * validation_result['mean_precision'] * validation_result['mean_recall']) /
                            (validation_result['mean_precision'] + validation_result['mean_recall']))
            except ZeroDivisionError:
                f1_score = 0
            else:
                if torch.isnan(f1_score):
                    f1_score = 0

            if bool(f1_score > best_f1):
                best_f1 = f1_score
                yolo.save_weights(filename=weight_dir / best_filename)

            logging.info(f'epoch num: {e},\n'
                         f'train_result: {train_result},\n'
                         f'val_result: {validation_result},\n'
                         f'val_f1: {f1_score}')

            logging.info(f'time: {time.time() - start_time}')

            val_dct = {'train_precision': train_result['mean_precision'],
                       'train_recall': train_result['mean_recall'],
                       'val_precision': validation_result['mean_precision'],
                       'val_recall': validation_result['mean_recall'],
                       'val_f1': f1_score}

            visualizer.plot(val_dct)
        #
        # if e in lr_decay_epoch:
        #     learning_rate *= lr_decay
        #     optimizer = torch.optim.Adam(yolo.parameters(), lr=learning_rate)

        if e > 1 and e % save_interval == 0:
            saved_count += 1
            filename = name + '_' + str(e) + '.weights'
            yolo.save_weights(filename=weight_dir / filename)

    logging.info(f'best_f1: {best_f1}')


if __name__ == '__main__':
    config_path = '../config/config.json'
    config = utils.parse_config(config_path)
    # name = 'opening'
    # train(config, name)
