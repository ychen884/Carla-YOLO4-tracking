# YOLOv3 + Deep_SORT

<img src="https://github.com/yehengchen/video_demo/blob/master/video_demo/output.gif" width="40%" height="40%"> <img src="https://github.com/yehengchen/video_demo/blob/master/video_demo/TownCentreXVID_output.gif" width="40%" height="40%">
<img src="https://github.com/yehengchen/Object-Detection-and-Tracking/blob/master/OneStage/yolo/yolo_img/output_person_315_1120_s.gif" width="40%" height="40%"> <img src="https://github.com/yehengchen/video_demo/blob/master/video_demo/output_car_143.gif" width="40%" height="40%">

__Object Tracking & Counting Demo - [[YouTube]](https://www.youtube.com/watch?v=ALw3OfrGWGo) [[BiliBili_V1]](https://www.bilibili.com/video/av55778717) [[BiliBili_V2]](https://www.bilibili.com/video/av59547404)  [[Chinese Version]](https://blog.csdn.net/weixin_38107271/article/details/96741706)__
## Requirement
__Development Environment: [Deep-Learning-Environment-Setup](https://github.com/yehengchen/Ubuntu-16.04-Deep-Learning-Environment-Setup)__ 

* OpenCV
* sklean
* pillow
* numpy 1.15.0
* tensorflow-gpu 1.13.1
* CUDA 10.0
***

It uses:

* __Detection__: [YOLOv3](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3) to detect objects on each of the video frames. - 用自己的数据训练YOLOv3模型

* __Tracking__: [Deep_SORT](https://github.com/nwojke/deep_sort) to track those objects over different frames.

*This repository contains code for Simple Online and Realtime Tracking with a Deep Association Metric (Deep SORT). We extend the original SORT algorithm to integrate appearance information based on a deep appearance descriptor. See the [arXiv preprint](https://arxiv.org/abs/1703.07402) for more information.*

## Quick Start

__0.Requirements__

    pip install -r requirements.txt
    
__1. Download the code to your computer.__
    
    git clone https://github.com/yehengchen/Object-Detection-and-Tracking.git
    
__2. Download [[yolov3.weights]](https://pjreddie.com/media/files/yolov3.weights)__ and place it in `deep_sort_yolov3/model_data/`

*Here you can download my trained [[yolo-spp.h5]](https://pan.baidu.com/s/1DoiifwXrss1QgSQBp2vv8w&shfl=shareset) - `t13k` weights for detecting person/car/bicycle,etc.*

__3. Convert the Darknet YOLO model to a Keras model:__
```
$ python convert.py model_data/yolov3.cfg model_data/yolov3.weights model_data/yolo.h5
``` 
__4. Run the YOLO_DEEP_SORT:__

```
$ python main.py -c [CLASS NAME] -i [INPUT VIDEO PATH]

$ python main.py -c person -i ./test_video/testvideo.avi
```

__5. Can change [deep_sort_yolov3/yolo.py] `__Line 100__` to your tracking object__

*DeepSORT pre-trained weights using people-ReID datasets only for person*
```
    if predicted_class != args["class"]:
               continue
    
    if predicted_class != 'person' and predicted_class != 'car':
               continue
```

## Train on Market1501 & MARS
*People Re-identification model*

[cosine_metric_learning](https://github.com/nwojke/cosine_metric_learning) for training a metric feature representation to be used with the deep_sort tracker.

## Citation

### YOLOv3 :

    @article{yolov3,
    title={YOLOv3: An Incremental Improvement},
    author={Redmon, Joseph and Farhadi, Ali},
    journal = {arXiv},
    year={2018}
    }

### Deep_SORT :

    @inproceedings{Wojke2017simple,
    title={Simple Online and Realtime Tracking with a Deep Association Metric},
    author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
    booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
    year={2017},
    pages={3645--3649},
    organization={IEEE},
    doi={10.1109/ICIP.2017.8296962}
    }

    @inproceedings{Wojke2018deep,
    title={Deep Cosine Metric Learning for Person Re-identification},
    author={Wojke, Nicolai and Bewley, Alex},
    booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
    year={2018},
    pages={748--756},
    organization={IEEE},
    doi={10.1109/WACV.2018.00087}
    }
    
## Reference
#### Github:deep_sort@[Nicolai Wojke nwojke](https://github.com/nwojke/deep_sort)
#### Github:deep_sort_yolov3@[Qidian213 ](https://github.com/Qidian213/deep_sort_yolov3)



