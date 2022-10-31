import cv2
import argparse
import os
import torch
import time
import yaml
import numpy as np
import ast


from detector import build_detector
from deep_sort import build_tracker

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)
        
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1+t_size[0]+3, y1+t_size[1]+4), color, -1)
        cv2.putText(
            img, label, (x1, y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        # draw center point
        cv2.circle(img, (x_center, y_center), 2, color, 2)
    return img


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / \
            shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

class vehicle_speed():
    def __init__(self, vid_fps, distance):
        self.vehilce_id_list = []
        self.frame_start_list = []
        self.speed_list = []
        self.vid_fps = vid_fps
        self.distance = distance/1000
        
    def new_vehicle(self, id, frame_start):
        self.vehilce_id_list.append(id)
        self.frame_start_list.append(frame_start)
        self.speed_list.append(0)
    def calc_speed(self, frame_end, id):
        index = self.vehilce_id_list.index(id)
        frame_start = self.frame_start_list[index]
        time = ((1/self.vid_fps)/60)/60
        speed = self.distance/((frame_end-frame_start)*time)
        self.speed_list[index] = speed
        return speed    
    def delete(self, id):
        index = self.vehilce_id_list.index(id)
        del self.vehilce_id_list[index]
        del self.frame_start_list[index]
        del self.speed_list[index]
    # def save_calc_speed(self, id):
    #     index = self.vehilce_id_list.index(id)
        


class Video_Tracker(object):
    def __init__(self, args, cfg):
        self.detector = build_detector(args.detector, cfg)
        self.deepsort = build_tracker(cfg, use_cuda=True)
        self.video_path = args.video_path
        self.save_path = args.save_path
        self.class_names = self.detector.class_names
        self.vdo = cv2.VideoCapture()

        assert os.path.isfile(self.video_path), "Path error"
        self.vdo.open(self.video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.stride = self.detector.stride
        self.auto = self.detector.auto
        self.img_size = ast.literal_eval(cfg["YOLOV5"]["IMG_SIZE"])

        assert self.vdo.isOpened()

        if args.save_path:
            os.makedirs(args.save_path, exist_ok=True)
            self.save_video_path = os.path.join(
                args.save_path, "results.avi")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(
                self.save_video_path, fourcc, 30, (self.im_width, self.im_height))
        # return self
    def run(self):
        idx_frame = 0
        cal_speed = vehicle_speed(vid_fps=30, distance=20)
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % 1:
                continue

            start = time.time()

            ret_val, im0 = self.vdo.retrieve()
            # print(self.img_size[0])
            im = letterbox(im0, self.img_size,
                           stride=self.stride, auto=self.auto)[0]
            im = im.transpose((2, 0, 1))[::-1]
            im = np.ascontiguousarray(im)
            bbox_xywh, cls_conf, cls_ids = self.detector(im, im0)

            # filter class 0
            # bbox_xywh_final = torch.Tensor()
            # cls_conf_final = torch.Tensor()
            
            # for i, cls_id in enumerate(cls_ids):
            #     if cls_id == 0:
            #         bbox_xywh_final.append(bbox_xywh[i])
            #         cls_conf_final.append(cls_conf[i])

                    
                    
            # reshape to tensor x*4
            
            
            # bbox_xywh_final = np.array(bbox_xywh_final)
            # cls_conf_final = np.array(cls_conf_final)
            # bbox_xywh_final = torch.tensor(bbox_xywh_final).view(-1, 4)
            # cls_conf_final = torch.tensor(cls_conf_final).view(-1, 1)
            
            
            # to tensor
            # bbox_xywh = torch.Tensor(bbox_xywh_final)
            # cls_conf = torch.Tensor(cls_conf_final)

            # deepsort
            line_1 = [(556, 556), (1664, 556)]
            line_2 = [(690, 390), (1288, 390)]
            # draw line
            cv2.line(im0, line_1[0], line_1[1], (0, 255, 0), 2)
            cv2.line(im0, line_2[0], line_2[1], (0, 255, 0), 2)
            
            
        
            
            
            outputs = self.deepsort.update(
                bbox_xywh.cpu(), cls_conf.cpu(), im0)
            # print(outputs)
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                im0 = draw_boxes(
                    im0, bbox_xyxy, identities)
                
                for i, bbox in enumerate(bbox_xyxy):
                    id = int(identities[i]) if identities is not None else 0
                    x_min, y_min, x_max, y_max = bbox
                    y_center = (y_min + y_max)/2
                    x_center = (x_min + x_max)/2
                    
                    # print(id, y_center)
                    if y_center < line_1[0][1] and y_center > line_2[0][1] and x_center > line_1[0][0] and x_center < line_1[1][0] and id not in cal_speed.vehilce_id_list:
                        cal_speed.new_vehicle(id, idx_frame)
                    elif y_center < line_1[0][1] and y_center < line_2[1][1] and x_center > line_2[0][0] and x_center < line_2[1][0] and  id in cal_speed.vehilce_id_list:
                        print("check")
                        index = cal_speed.vehilce_id_list.index(id)
                        if cal_speed.speed_list[index] == 0:
                            speed = cal_speed.calc_speed(idx_frame, id)
                            speed = round(speed, 2)
                            im0 = cv2.putText(im0, str(speed) + "km/h", (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cal_speed.speed_list[index] = speed
                        else:
                            im0 = cv2.putText(im0, str(cal_speed.speed_list[index]) + "km/h", (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            print(str(cal_speed.speed_list[index]) + "km/h")
                        
                    # elif  
                        # cal_speed.delete(id)


                        
                        
                        
                    
            

            if self.save_path:
                self.writer.write(im0)
            # print("time: {:.03f}s, fps: {:.03f}".format(
            #     time.time() - start, 1 / (time.time() - start)))
            
            
            
            
            

        self.writer.release()
        self.vdo.release()


def get_cfg(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str,
                        default='data/video/test.mp4')
    parser.add_argument('--detector', type=str, default='yolov5')
    parser.add_argument('--cfg_path', type=str,
                        default='../cfgs/yolo_deepsort.yaml')
    parser.add_argument('--save_path', type=str,
                        default='/workspace/nabang1010/DATN/results')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg(args.cfg_path)
    print("=============================================================================")
    tracker = Video_Tracker(args, cfg)
    tracker.run()
