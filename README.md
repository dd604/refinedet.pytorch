# RefineDet in PyTorch
This is a PyTorch implementation of [Single-Shot Refinement Neural Network for Object Detection](https://arxiv.org/abs/1711.06897) that is a work by Shifeng Zhang, Longyin Wen, Xiao Bian, Zhen Lei and Stan Z. Li in CVPR2018. The official and original Caffe code can be found [here](https://github.com/sfzhang15/RefineDet).
This implementation mainly refers the official RefineDet in Caffe [sfzhang15/RefineDet](https://github.com/sfzhang15/RefineDet) and a PyTorch implementation of SSD [amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch). Its sublibrary(*libs/datasets*) to process datasets is obtained from [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) with minor modification.
A simple post in Chinese about this project is [here](https://zhuanlan.zhihu.com/p/56800496).

## Repuisites
* Python 3.6.1
* PyTorch 1.2.0
* CUDA 9.0 or higher


## Preparation

First of all, clone the project and create two folders. The "data" is used for pretrained models and datasets. The "output" is used for output models
```
git clone https://github.com/dd604/refinedet.pytorch.git
cd refinedet.pytorch
mkdir data
mkdir output
```
### Compilation
Install all the python dependencies if necessary:
```
pip install -r requirements.txt
```
Compile the dependencies as following:
```
cd libs
sh make.sh
```
It will complie the COCOAPI. The default version is complied with python 3.6.4. They should be re-compiled if you use a different python version.

### Data preparation

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
In details, the VOC datasets are as following:
```Shell
VOCdevkit2007
|__ VOC2007
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
    
VOCdevkit2012
|__ VOC2012
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
```
The coco dataset is as following:
```Shell
coco
|__ annotations
|   |_ instances_valminusminival2014.json
|   |_ instances_minival2014.json
|   |_ instances_train2014.json
|   |_ instances_val2014.json
|   |_ ...
|__ images
    |__ train2014
    |   |_ <im-1-name>.jpg
    |   |_ ...
    |   |_ <im-N-name>.jpg
    |__ val2014
        |_ <im-1-name>.jpg
        |_ ...
        |_ <im-N-name>.jpg
```

## Pre-trained model
You can train a RefineDet detector with [VGG16](https://arxiv.org/abs/1409.1556) or [ResNet101](https://arxiv.org/abs/1512.03385) as a base network. The pretrained models can be downloaded from [vgg16_reducedfc.pth](https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth) and [resnet101.pth](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth).
By default, the directories for VGG16 weights is assumed as following:
```Shell
mkdir data/pretrained_model
cd data/pretrained_model
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```
For ResNet101:
```Shell
mkdir data/pretrained_model
cd data/pretrained_model
wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth -O resnet101.pth
```

## Train
To train a RefineDet model with PASCAL VOC07+12, run as:
```
python -u train_refinedet.py --dataset voc --input_size 320 --batch_size 32 --network vgg16 --base_model vgg16_reducedfc.pth 
``` 
Change dataset to "coco", if you want to use COCO2014.


## Performance
The project trys to reproduce the performance of RefineDet in Caffe, but there are some gaps for VGG16. For resnet101, the results are comparable.
If you have any suggetion to promote this reproduction, please leave a message in the issues.

1). PASCAL VOC07+12 (Train/Test: VOC07+12/VOC07)

|Method |Backbone | Input Size | mAP |
|-------|---------|------------|-----|
|SSD      | VGG16| 300 x 300 | 77.2 |
|SSD      | VGG16| 512 x 512 | 79.8 |
|RefineDet(Official)| VGG16| 320 x 320 | 80.0 |
|RefineDet(Official)| VGG16| 512 x 512 | 81.8 |
|**RefineDet(Our)**| VGG16| 320 x 320 | 78.6 |
|**RefineDet(Our)**| VGG16| 512 x 512 | 79.1 |

The trained models producing the above performance can be downloaded from Dropbox at [vgg16_refinedet320_voc](https://www.dropbox.com/s/gynb405fixwqitv/vgg16_refinedet320_voc_120000.pth?dl=0) and [vgg16_refinedet512_voc](https://www.dropbox.com/s/y527gz2dz4ow0wz/vgg16_refinedet512_voc_120000.pth?dl=0), or from BaiduPan at [vgg16_refinedet320_voc](https://pan.baidu.com/s/1ydhTwuKPONh11NmXXalmuw)(password: d236) and [vgg16_refinedet512_voc](https://pan.baidu.com/s/1e_IPCALi6KvLDT9yv9dMqQ)(password: iejy).

2). COCO2014 (Train/Test: trainval115k/minval5k)

|Method |Backbone | Input Size | mAP |
|-------|---------|------------|-----|
|SSD321      | ResNet101 | 321 x 321 | 28.0 |
|RefineDet(Official)| ResNet101| 320 x 320 | 32.0 |
|RefineDet(Official)| ResNet101| 512 x 512 | 36.4 | 
|**RefineDet(Our)** | ResNet101| 320 x 320 | 31.7 |
|**RefineDet(Our)**| ResNet101| 512 x 512 | 36.6 |

The trained model can be download from Dropbox at [resnet101_refinedet320_coco](https://www.dropbox.com/s/bu8khr18ped59n5/resnet101_refinedet320_coco_400000.pth?dl=0) and [resnet101_refinedet512_coco](https://www.dropbox.com/s/d5wouxm12bp50ke/resnet101_refinedet512_coco_400000.pth?dl=0) or from BaiduPan at [resnet101_refinedet320_coco](https://pan.baidu.com/s/1YIfB2Y4kChpgA4CBJPZ5oA)(password: jgjt) and [resnet101_refinedet512_coco](https://pan.baidu.com/s/1mjO4fv7STQOHwK2JEjOaWw)(password: pk2f).
~~Training is failed with NAN loss when input size is 512x512, and I am seeking reasons.~~ Now the problem of NAN loss is solved.


## Evaluation
To evaluate the trained model, you can run "eval_refinedet.py".
For example, download the trained vgg16_refinedet320_voc model (named vgg16_refinedet320_voc_120000.pth), and put it to "output". Assign parameters as following:
```Shell
python -u eval_refinedet.py --input_size 320 --dataset voc --network vgg16 --model_path "./output/vgg16_refinedet320_voc_120000.pth"
```


## Demo
Using the above trained vgg16_refinedet320_voc model (vgg16_refinedet320_voc_120000.pth) and put it to "output", you can run demos at the folder "demo".
```
cd demo
python demo_simple.py
```
It detects objects on a local image "000004.jpg" and draw results with opencv on the image "000004_result.jpg". You can also run demo.ipynb with jupyter notebook to visualize detection results.

## Miscellaneous
There are some major changes compared with the previous project version. Firstly, the problem of NAN loss is solved. Secondly, the construction of the project is modified for readability. 

## Authors
* [Dongdong Wang](https://github.com/dd604)

## References
- Shifeng Zhang, et al. [Single-Shot Refinement Neural Network for Object Detection](https://arxiv.org/abs/1711.06897), [official Caffe code](https://github.com/sfzhang15/RefineDet)
- [SSD in PyTorch](https://github.com/amdegroot/ssd.pytorch)
- [Faster RCNN in PyTorch](https://github.com/jwyang/faster-rcnn.pytorch)
