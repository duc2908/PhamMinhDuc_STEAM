import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from deep_sort import build_tracker

PALETTE = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


class VehicleSpeed:
    def __init__(self, fps: float, distance_meters: float) -> None:
        self.fps = fps
        self.distance_meters = distance_meters
        self.state: Dict[int, int] = {}

    def start_if_missing(self, track_id: int, frame_idx: int) -> None:
        if track_id not in self.state:
            self.state[track_id] = frame_idx

    def get_speed(self, track_id: int, frame_idx: int) -> float:
        start_frame = self.state.get(track_id)
        if start_frame is None or start_frame >= frame_idx:
            return 0.0
        elapsed_seconds = (frame_idx - start_frame) / max(self.fps, 1e-6)
        if elapsed_seconds <= 0:
            return 0.0
        speed_mps = self.distance_meters / elapsed_seconds
        return speed_mps * 3.6


class VideoTracker:
    def __init__(self, args: argparse.Namespace, cfg: dict) -> None:
        self.cfg = cfg
        self.paths_cfg = cfg.get("PATHS", {})
        self.infer_cfg = cfg.get("INFER", cfg.get("YOLOV8", {}))
        self.tracker_cfg = cfg.get("TRACKER", {})
        self.vis_cfg = cfg.get("VISUALIZATION", {})
        self.speed_cfg = cfg.get("SPEED", {})
        self.cfg_dir = Path(args.cfg_path).resolve().parent if getattr(args, "cfg_path", "") else Path(__file__).resolve().parent.parent

        self.video_path = self._resolve_path(args.video_path or self.paths_cfg["VIDEO_PATH"])
        save_dir = self._resolve_path(args.save_path or self.paths_cfg["SAVE_DIR"])
        output_name = self.paths_cfg["OUTPUT_NAME"]
        save_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self._resolve_output_path(save_dir / output_name)

        model_path = self._resolve_from_cfg(self.infer_cfg["WEIGHTS"])
        self.model = YOLO(str(model_path))
        class_names = self.infer_cfg.get("CLASS_NAMES", [])
        if class_names:
            self.model.model.names = {i: name for i, name in enumerate(class_names)}

        self.device = self.infer_cfg["DEVICE"]
        self.img_size = int(self.infer_cfg["IMG_SIZE"])
        self.img_size_fallbacks = self.infer_cfg.get("IMG_SIZE_FALLBACKS", [512, 416, 320])
        self.conf_thresh = float(self.infer_cfg["CONF_THRESH"])
        self.iou_thresh = float(self.infer_cfg["IOU_THRESH"])
        self.class_ids = self.infer_cfg.get("CLASSES", None)

        tracker_cfg = {
            "DEEPSORT": {
                "REID_CKPT": str(self._resolve_from_cfg(self.paths_cfg["REID_CKPT"])),
                "MAX_DIST": cfg["TRACKER"]["MAX_DIST"],
                "MIN_CONFIDENCE": cfg["TRACKER"]["MIN_CONFIDENCE"],
                "NMS_MAX_OVERLAP": cfg["TRACKER"]["NMS_MAX_OVERLAP"],
                "MAX_IOU_DISTANCE": cfg["TRACKER"]["MAX_IOU_DISTANCE"],
                "MAX_AGE": cfg["TRACKER"]["MAX_AGE"],
                "N_INIT": cfg["TRACKER"]["N_INIT"],
                "NN_BUDGET": cfg["TRACKER"]["NN_BUDGET"],
            }
        }
        self.deepsort = build_tracker(tracker_cfg, use_cuda=torch.cuda.is_available())

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise SystemExit(f"Unable to open video: {self.video_path}")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or self.speed_cfg["FPS_FALLBACK"])

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.video_fps,
            (self.frame_width, self.frame_height),
        )

        self.track_thickness = int(self.vis_cfg["TRACK_THICKNESS"])
        self.line_thickness = int(self.vis_cfg["LINE_THICKNESS"])
        self.text_scale = float(self.vis_cfg["TEXT_SCALE"])
        self.text_thickness = int(self.vis_cfg["TEXT_THICKNESS"])
        self.speed_color = tuple(self.vis_cfg["SPEED_COLOR"])

        self.speed_enabled = bool(self.speed_cfg["ENABLED"])
        self.line_1 = tuple(map(tuple, self.speed_cfg["LINE_1"]))
        self.line_2 = tuple(map(tuple, self.speed_cfg["LINE_2"]))
        self.speed_estimator = VehicleSpeed(self.video_fps, float(self.speed_cfg["DISTANCE_METERS"]))

    @staticmethod
    def _resolve_output_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        for idx in range(1, 1000):
            candidate = path.with_name(f"{stem}_{idx:03d}{suffix}")
            if not candidate.exists():
                return candidate
        raise RuntimeError("Could not determine unique output path")

    def _resolve_path(self, relative_or_abs: str) -> Path:
        path = Path(relative_or_abs)
        if path.is_absolute():
            return path
        return (self.cfg_dir / path).resolve()

    def _resolve_from_cfg(self, relative_or_abs: str) -> Path:
        path = Path(relative_or_abs)
        if path.is_absolute():
            return path
        return (self.cfg_dir / path).resolve()

    @staticmethod
    def _compute_color(track_id: int) -> Tuple[int, int, int]:
        color = [int((p * (track_id ** 2 - track_id + 1)) % 255) for p in PALETTE]
        return tuple(color)

    def _draw_track_box(self, frame: np.ndarray, box: np.ndarray, track_id: int, class_name: str, conf: float) -> None:
        x1, y1, x2, y2 = [int(v) for v in box]
        color = self._compute_color(track_id)
        label = f"{class_name} {conf:.2f} ID:{track_id}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, self.text_scale, self.text_thickness)[0]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.track_thickness)
        cv2.rectangle(frame, (x1, y1), (x1 + text_size[0] + 3, y1 + text_size[1] + 4), color, -1)
        cv2.putText(
            frame,
            label,
            (x1, y1 + text_size[1] + 4),
            cv2.FONT_HERSHEY_PLAIN,
            self.text_scale,
            (255, 255, 255),
            self.text_thickness,
        )
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        cv2.circle(frame, center, 2, color, 2)

    def _draw_speed_lines(self, frame: np.ndarray) -> None:
        if not self.speed_enabled:
            return
        cv2.line(frame, self.line_1[0], self.line_1[1], (0, 255, 0), self.line_thickness)
        cv2.line(frame, self.line_2[0], self.line_2[1], (0, 255, 0), self.line_thickness)

    def _update_speed(self, frame: np.ndarray, box: np.ndarray, track_id: int, frame_idx: int) -> None:
        if not self.speed_enabled:
            return

        x_min, y_min, x_max, y_max = box
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0

        in_start = (
            self.line_2[0][1] < y_center < self.line_1[0][1]
            and self.line_1[0][0] < x_center < self.line_1[1][0]
        )
        in_end = (
            y_center < self.line_2[1][1]
            and self.line_2[0][0] < x_center < self.line_2[1][0]
        )

        if in_start:
            self.speed_estimator.start_if_missing(track_id, frame_idx)
        elif in_end and track_id in self.speed_estimator.state:
            speed = round(self.speed_estimator.get_speed(track_id, frame_idx), 2)
            cv2.putText(
                frame,
                f"{speed} km/h",
                (int(x_min), int(y_min)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.text_scale,
                self.speed_color,
                self.text_thickness,
            )

    def _extract_detections(self, frame: np.ndarray):
        tried_sizes = [self.img_size] + [int(v) for v in self.img_size_fallbacks if int(v) != self.img_size]
        result = None
        last_err = None

        for size in tried_sizes:
            try:
                result = self.model.predict(
                    frame,
                    imgsz=size,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    classes=self.class_ids,
                    device=self.device,
                    verbose=False,
                )[0]
                break
            except RuntimeError as err:
                if "out of memory" in str(err).lower() and str(self.device) != "cpu":
                    torch.cuda.empty_cache()
                    last_err = err
                    continue
                raise

        if result is None:
            raise RuntimeError(f"GPU inference failed for sizes {tried_sizes}. Last error: {last_err}")

        if result.boxes is None or len(result.boxes) == 0:
            return torch.empty((0, 4)), torch.empty((0,)), np.array([], dtype=int)

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        xywh = np.zeros((xyxy.shape[0], 4), dtype=np.float32)
        xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
        xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
        return torch.from_numpy(xywh), torch.from_numpy(confs), classes

    def run(self) -> None:
        frame_idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            frame_idx += 1

            self._draw_speed_lines(frame)
            bbox_xywh, cls_conf, class_ids = self._extract_detections(frame)

            if bbox_xywh.shape[0] > 0:
                outputs = self.deepsort.update(bbox_xywh, cls_conf, class_ids, frame)
            else:
                outputs = []

            if len(outputs) > 0:
                boxes = outputs[:, :4]
                identities = outputs[:, 4].astype(int)
                classes = outputs[:, 5].astype(int)
                confs = outputs[:, 6]

                for i, box in enumerate(boxes):
                    track_id = int(identities[i])
                    class_idx = int(classes[i])
                    class_name = self.model.names.get(class_idx, f"cls_{class_idx}")
                    self._draw_track_box(frame, box, track_id, class_name, float(confs[i]))
                    self._update_speed(frame, box, track_id, frame_idx)

            self.writer.write(frame)

        self.writer.release()
        self.cap.release()
        print(f"Wrote {self.output_path}")


def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8s + DeepSORT tracking")
    parser.add_argument("--cfg_path", type=str, default="../cfgs/yolov8s.yaml")
    parser.add_argument("--video_path", type=str, default="")
    parser.add_argument("--save_path", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_cfg(args.cfg_path)
    tracker = VideoTracker(args, cfg)
    tracker.run()
