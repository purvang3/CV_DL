This project is modified version of [yolact_edge](https://arxiv.org/abs/2012.12259).

original implementation of project [github](https://github.com/haotian-liu/yolact_edge)

yolact_edge readme file [YOLACT_EDGE_README.md](YOLACT_EDGE_README.md).

If you want to train object detection model and your data doesn't have
segmantation mask and still you want to train 
yolact_edge, I have done some modification in predictor as well as in loss 
calculation to train model with images and bboxes as labels without segmantation mask.

if you want to use project in it's original form 
```
include_mask = True 
```
under yolact_base_config config object in config.py.

