import torch
import sys
from tqdm import tqdm

sys.path.append('./detector/yolov5')
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_boxes

class YOLOv5(weight_path, data_path, ):
    def __init__(self, weight_path, data_path, img_size, ):
        self.weight = weight_path
        self.data = data_path
        self.device = select_device('0')
        self.img_size = img_size
        self.model = DetectMultiBackend(self.weight, self.device ,self.data)
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        self.vid_stride = 1

    def load_data(self, source_video):
        dataset = LoadImages(source_video, img_size=self.img_size, stride=self.stride, auto=self.pt, vid_stride=self.vid_stride)
        return dataset
        
    
    def detect(self, dataset):
        for path, im, im0s, vid_cap, s in dataset:
            img = torch.from_numpy(im).to(self.device).float() 
            img /=255
            if img.ndimension() == 3:
                img = img[None]
            pred = self.model(img, augment=False)
            pred = non_max_suppression(prediction=pred, conf_thres=0.4, iou_thres=0.5, classes=None, agnostic=False)
            
            for i, det in enumerate(pred):
                if len(det):
                    p, im0, frame =  path, im0s.copy(), getattr(dataset, 'frame', 0)
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        x1 = xyxy[0].item()
                        y1 = xyxy[1].item()
                        x2 = xyxy[2].item()
                        y2 = xyxy[3].item()
                        conf = conf.item()
                        
                        
            
                    
                    
                    