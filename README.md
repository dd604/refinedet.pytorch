# RefineDet in PyTorch
This is a PyTorch implementation of [Single-Shot Refinement Neural Network for Object Detection](https://arxiv.org/abs/1711.06897) that is a work by Shifeng Zhang, Longyin Wen, Xiao Bian, Zhen Lei and Stan Z. Li in CVPR2018. The official and original Caffe code can be found [here](https://github.com/sfzhang15/RefineDet).
This implementation mainly refers the official RefineDet in Caffe [sfzhang15/RefineDet](https://github.com/sfzhang15/RefineDet) and a PyTorch implementation of SSD [amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch). Its sublibrary(*libs/dataset/datasets*) to process datasets is obtained from [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) with minor modification.


## Repuisites
* Python 2.7
* PyTorch 0.3.1
* CUDA 8.0 or higher

## Data Preparation
* **PASCAL_VOC 07+12**: You can follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) or [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) to prepare VOC datasets, i.e., putting the data or creating soft links in the folder data/.

* **COCO**: You can use COCO2014 to train your model by the same setting as PASCAL_VOC07+12.

The directory trees in data/ in my projects are as following:
```Shell
├── coco -> /root/dataset/coco
├── VOCdevkit2007
│   └── VOC2007 -> /root/dataset/voc/VOCdevkit/VOC2007
└── VOCdevkit2012
    └── VOC2012 -> /root/dataset/voc/VOCdevkit/VOC2012
```

## Train
You can train a RefineDet detector with [VGG16](https://arxiv.org/abs/1409.1556) or [ResNet101](https://arxiv.org/abs/1512.03385) as a base network. The pretrained models can be downloaded from [vgg16_reducedfc.pth](https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth) and [resnet101.pth](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth).
By default, the directories for VGG16 weights is assumed as following:
```Shell
mkdir -p weights/vgg16
cd weights/vgg16
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```
For ResNet101:
```Shell
mkdir -p weights/resnet101
cd weights/resnet101
wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth -O resnet101.pth
```
To train a RefineDet model with PASCAL VOC07+12, run as:
```
python train_refinedet.py --dataset voc --input_size 320 --network vgg16 --basenet vgg16_reducedfc.pth --save_folder "weights/vgg16"
``` 
Change dataset to "coco", if you want to use COCO2014.

## Evaluation
To evaluate a trained model, run as:
```Shell
python eval_refinedet.py --input_size 512 --dataset voc --network vgg16 --model_path your/weights/path
```

## Performance
The project trys to reproduce the performance of RefineDet in Caffe, but there are some gaps.
If you have any suggetion to promote this reproduction, please leave a message in the issues.

1). PASCAL VOC07+12 (Train/Test: VOC07+12/VOC07)

|Method |Backbone | Input Size | mAP | FPS |
|-------|---------|------------|-----|-----|
|SSD      | VGG16| 300 x 300 | 77.2 | 46 |
|SSD      | VGG16| 512 x 512 | 79.8 | 19 |
|RefineDet(Official)| VGG16| 320 x 320 | 80.0 | 40.3 |
|RefineDet(Official)| VGG16| 512 x 512 | 81.8 | 24.1 |
|**RefineDet(Our)**| VGG16| 320 x 320 | 78.9 | ~45 |
|**RefineDet(Our)**| VGG16| 512 x 512 | 79.8 | ~30 |

The speed is evaluted on P40.

## Demo
You can run demo/demo.ipynb with jupyter notebook to visualize detetion results.


## Authors
* [Dongdong Wang](https://github.com/dd604)

## References
- Shifeng Zhang, et al. [Single-Shot Refinement Neural Network for Object Detection](https://arxiv.org/abs/1711.06897), [official Caffe code](https://github.com/sfzhang15/RefineDet)
- [SSD in PyTorch](https://github.com/amdegroot/ssd.pytorch)
- [Faster RCNN in PyTorch](https://github.com/jwyang/faster-rcnn.pytorch)
