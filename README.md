# Vehicle Tracking

---
### Install
1. Go to working directory
```
cd PhamMinhDuc_STEAM

```
2. Install Anaconda enviroment

```
conda env create --name vehicle_tracking --file=env.yaml
```
3. Activate enviroment
```
conda activate vehicle_tracking
```


### Vehicle Detection
#### Prepare


#### YOLOv5

Vehicle Detection with YOLOv5


```


```

#### YOLOv7

Vehicle Detection with YOLOv7


```


```



### Vehicle Tracking and Speed Estimation


#### Prepare

1. Download pre-trained re-id model checkpoints [ckpt.t7]() and put in `weights/reid`

1. Edit video_fps and distance in `yolo_deepsort.py` line 140

2. Edit position of two lines in `yolo_deepsort.py` line 181, 182

3. Edit `--video_path`, `--detector`, `--cfg_path`, `--save_path` in `run.sh`

#### Run

1. Change working directory to `src`

```
cd src
```

2. Run file `run.sh`

```
sh run.sh
```

---

### References

* [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
* [ZQPei/deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch)
