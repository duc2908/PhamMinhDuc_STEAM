# from yolov4.detector import YOLOv4
from .yolov5 import YOLOv5
# from yolov7.detector import YOLOv7

def build_detector(detector, cfg):
    if detector ==  "yolov5":
        return YOLOv5(weight_path=cfg["YOLOV5"]['WEIGHT'], data_path=cfg["YOLOV5"]['CFG'], img_size=cfg["YOLOV5"]['IMG_SIZE'], conf_thresh=cfg["YOLOV5"]['CONF_THRESH'], iou_thresh=cfg["YOLOV5"]['IOU_THRESH'])
