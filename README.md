# yolo3-pytorch
Train and inference yolov3 in pytorch

## Module Description

   1. yolo/anchor.py: generate anchors from dataset, if your dataset is quite different from COCO or PASCALE VOC, 
   this should be the first step.
   1. yolo/train.py: train yolov3.
   1. yolo/detect.py: predict with pretrained model weights, you can use the official weights from: [yolov3.weights].
   1. yolo/validation.py: validate pretrained model on your dataset.
 

[yolov3.weights]: https://pjreddie.com/media/files/yolov3.weights

## Train


## Data Organization


    root_directory: data/opening_detection
    train_data: root_directory / data
    label: root_directory / label


