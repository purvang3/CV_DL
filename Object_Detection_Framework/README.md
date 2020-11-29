#  Keras based object detection framework

## Introduction

[**Keras based object detection framework**]
####Purvang Lapsiwala

This Framework is a pure keras implementation of *SSD based Object Detection models*.
Framework is build and maintained by me and mainly build for research purpose.Framework is one of its first kind which 
gives users flexibility to choose wide variety of backbones with feature fusion methods
along with flexibility to build feature head network, which contributes to total network capacity.
 
User only need to add parameters in config file. Example of config file [here](https://github.com/purvang3/Object_Detection/blob/main/ODF/config.yaml)

Framework shares some code with [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet) . Information about Data preperation and pre-trained models can be
found from [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet). 

```
# python version
python 3.6

# packages installation
> pip install -r requirements.txt
```

### Key Feature Implementation

- [x] Support for various backbones. [ EfficientDet, Inception, Xception, Vgg, Vgg, Resnet].
- [x] Support for various Feature Fusion methods [NoFPN, FPN, BiFPN, WBiFPN, PANet_FPN (ongoing Research on FPN...))
- [x] Implementation of Flexible Anchor Generation Method.
- [x] Option for user Feed Square Image to Model.
- [x] Flexible framework to use any feature fusion method with any backbone.
- [x] Step, Linear and Polynomial learning rate schedule.
- [x] Feature_Visualization and Weight_Visualization after every n epoch. 

####* one of the key features of this framework is that any backbone can be configured with any feature fusion method by specifying in command line argument based on your your application (Accuracy Vs Inference).

### More backbones and features will be added and Framework will be released once Research is done.

### Usage

For training on a `custom dataset`, a CSV file can be used as a way to pass the data.
See below for more details on the format of these CSV files.
To train using your CSV, run:
```shell
# Running directly from the repository:
python train.py
``` 
