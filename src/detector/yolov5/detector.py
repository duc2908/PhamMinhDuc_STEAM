import torch

from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh


class YOLOv5(object):
    def __init__(self, weight_path, data_path, img_size=(640, 640), conf_thresh=0.8, iou_thresh=0.3):
        self.device = select_device('0')
        self.net = DetectMultiBackend(
            weights=weight_path, device=self.device, data=data_path)

        self.size = img_size
        self.net.eval()
        self.net.to(self.device)

        self.size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        self.stride = self.net.stride
        self.vid_stride = 1
        self.auto = self.net.pt

        self.class_names = list(self.net.names.values())

    def load_video(self, video_path):
        dataset = LoadImages(path=video_path, img_size=self.size,
                             stride=self.stride, auto=self.auto, vid_stride=self.vid_stride)
        return dataset

    def __call__(self, im, im0s):

        img = torch.from_numpy(im).to(self.device).float()
        img /= 255
        if img.ndimension() == 3:
            img = img[None]
        pred = self.net(img, augment=False)
        pred = non_max_suppression(prediction=pred, conf_thres=self.conf_thresh,
                                   iou_thres=self.iou_thresh, classes=None, agnostic=False)
        bbox = []
        cls_conf = []
        cls_ids = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(
                    img.shape[2:], det[:, :4], im0s.shape).round()

            bbox_xywh = xyxy2xywh(det[:, :4])
            cls_conf = det[:, 4:5]
            cls_ids = det[:, 5:6]
            # for *xyxy, conf, cls in reversed(det):
            #     # label = f'{self.class_names[int(cls)]}'
            #     x1 = xyxy[0].item()
            #     y1 = xyxy[1].item()
            #     x2 = xyxy[2].item()
            #     y2 = xyxy[3].item()
            #     conf = conf.item()
            #     x_center = (x1 + x2) / 2
            #     y_center = (y1 + y2) / 2
            #     w = x2 - x1
            #     h = y2 - y1
            #     bbox.append([x_center, y_center, w, h])
            #     cls_conf.append(conf)
            #     cls_ids.append(int(cls))

        return bbox_xywh, cls_conf, cls_ids


def demo():
    print("=============================================================================")
    print("====================================DEMO=====================================")
    weights = "/workspace/nabang1010/STEAM/PhamMinhDuc_STEAM/src/yolov5/PhamMinhDuc_STEAM/yolov5x_vehicle/weights/last.pt"
    data = "/workspace/nabang1010/STEAM/PhamMinhDuc_STEAM/src/yolov5/vehicle.yaml"
    yolov5 = YOLOv5(weights, data)
    img_path = "/workspace/nabang1010/DATN/src/test.jpg"
    dataset = yolov5.load_video(img_path)
    for _, im, im0s, __, ___ in dataset:
        bbox, cls_conf, cls_ids = yolov5(im, im0s)
        print("bbox: ", bbox)
        print("cls_conf: ", cls_conf)
        print("cls_ids: ", cls_ids)


if __name__ == "__main__":
    demo()
