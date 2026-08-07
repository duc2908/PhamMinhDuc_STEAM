import argparse
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from deep_sort import build_tracker

PALETTE = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
GroundPosition = Tuple[float, float]
TrackOutput = np.ndarray
REQUIRED_CONFIG = {
    "PATHS": ("VIDEO_PATH", "SAVE_DIR", "OUTPUT_NAME", "REID_CKPT"),
    "INFER": ("WEIGHTS", "DEVICE", "IMG_SIZE", "CONF_THRESH", "IOU_THRESH", "CLASSES", "CLASS_NAMES"),
    "TRACKER": ("MAX_DIST", "MIN_CONFIDENCE", "NMS_MAX_OVERLAP", "MAX_IOU_DISTANCE", "MAX_AGE", "N_INIT", "NN_BUDGET"),
    "SPEED": ("ENABLED", "CAMERA_HEIGHT_M", "CAMERA_ANGLE_DEG", "CAMERA_VFOV_DEG", "CAMERA_HFOV_DEG", "EMA_ALPHA"),
    "VISUALIZATION": ("TRACK_THICKNESS", "TEXT_SCALE", "TEXT_THICKNESS", "SPEED_COLOR"),
}


def _world_pos(
    pixel_x: float, pixel_y: float, camera_height: float, camera_angle_deg: float,
    vertical_fov_deg: float, horizontal_fov_deg: float, frame_width: int, frame_height: int,
) -> Optional[GroundPosition]:
    """Project pixel onto ground plane. Returns (X_right, Y_forward) in metres, or None."""
    angle = math.radians(camera_angle_deg)
    offset_x = pixel_x - frame_width / 2.0
    offset_y = pixel_y - frame_height / 2.0
    focal_x = (frame_width / 2.0) / math.tan(math.radians(horizontal_fov_deg) / 2.0)
    focal_y = (frame_height / 2.0) / math.tan(math.radians(vertical_fov_deg) / 2.0)
    ray_down = focal_y * (math.sin(angle) + (offset_y / focal_y) * math.cos(angle))
    if ray_down <= 0:
        return None
    ground_scale = camera_height / ray_down
    right_ray = offset_x * focal_y / focal_x
    forward_ray = focal_y * (math.cos(angle) - (offset_y / focal_y) * math.sin(angle))
    return (ground_scale * right_ray, ground_scale * forward_ray)


class VehicleSpeed:
    """Estimate each tracked vehicle's ground-plane speed."""

    def __init__(self, fps: float, camera_height: float, camera_angle_deg: float,
                 vertical_fov_deg: float, horizontal_fov_deg: float,
                 frame_width: int, frame_height: int, ema_alpha: float = 0.5) -> None:
        self.fps = fps
        self.camera_height = camera_height
        self.camera_angle_deg = camera_angle_deg
        self.vertical_fov_deg = vertical_fov_deg
        self.horizontal_fov_deg = horizontal_fov_deg
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.ema_alpha = ema_alpha
        self._last_position: Dict[int, Tuple[int, float, float]] = {}
        self._speed: Dict[int, float] = {}

    def update(self, track_id: int, pixel_x: float, pixel_y: float, frame_index: int) -> None:
        """Update speed from the vehicle's ground-contact pixel."""
        position = _world_pos(
            pixel_x, pixel_y, self.camera_height, self.camera_angle_deg,
            self.vertical_fov_deg, self.horizontal_fov_deg, self.frame_width, self.frame_height,
        )
        if position is None:
            return
        ground_x, ground_y = position
        previous = self._last_position.get(track_id)
        if previous is not None:
            previous_frame, previous_x, previous_y = previous
            elapsed_seconds = (frame_index - previous_frame) / max(self.fps, 1e-6)
            if elapsed_seconds > 0:
                instantaneous_kmh = math.hypot(ground_x - previous_x, ground_y - previous_y) / elapsed_seconds * 3.6
                previous_speed = self._speed.get(track_id, instantaneous_kmh)
                self._speed[track_id] = self.ema_alpha * instantaneous_kmh + (1 - self.ema_alpha) * previous_speed
        self._last_position[track_id] = (frame_index, ground_x, ground_y)

    def get_speed(self, track_id: int) -> float:
        return self._speed.get(track_id, 0.0)

    def forget(self, track_id: int) -> None:
        self._last_position.pop(track_id, None)
        self._speed.pop(track_id, None)


class VideoTracker:
    """Run detection, tracking, speed estimation, and output rendering."""

    def __init__(self, args: argparse.Namespace, cfg: dict) -> None:
        self.paths_cfg = cfg["PATHS"]
        self.infer_cfg = cfg["INFER"]
        self.vis_cfg = cfg["VISUALIZATION"]
        self.speed_cfg = cfg["SPEED"]
        self.cfg_dir = Path(args.cfg_path).resolve().parent

        self.video_path = self._resolve_path(args.video_path or self.paths_cfg["VIDEO_PATH"])
        save_dir = self._resolve_path(args.save_path or self.paths_cfg["SAVE_DIR"])
        save_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self._resolve_output_path(save_dir / self.paths_cfg["OUTPUT_NAME"])

        self.model = YOLO(str(self._resolve_path(self.infer_cfg["WEIGHTS"])))
        class_names = self.infer_cfg.get("CLASS_NAMES", [])
        if class_names:
            self.model.model.names = {i: name for i, name in enumerate(class_names)}

        self.device = self.infer_cfg["DEVICE"]
        self.img_size = int(self.infer_cfg["IMG_SIZE"])
        self.conf_thresh = float(self.infer_cfg["CONF_THRESH"])
        self.iou_thresh = float(self.infer_cfg["IOU_THRESH"])
        self.class_ids = self.infer_cfg["CLASSES"]

        self.deepsort = build_tracker(self._tracker_config(cfg["TRACKER"]), use_cuda=torch.cuda.is_available())
        self.cap, self.writer, self.frame_width, self.frame_height, self.video_fps = self._open_video_io()

        self.track_thickness = int(self.vis_cfg["TRACK_THICKNESS"])
        self.text_scale = float(self.vis_cfg["TEXT_SCALE"])
        self.text_thickness = int(self.vis_cfg["TEXT_THICKNESS"])
        self.speed_color = tuple(self.vis_cfg["SPEED_COLOR"])

        self.speed_enabled = bool(self.speed_cfg["ENABLED"])
        camera_height = float(self.speed_cfg["CAMERA_HEIGHT_M"])
        camera_angle_deg = float(self.speed_cfg["CAMERA_ANGLE_DEG"])
        vertical_fov_deg = float(self.speed_cfg["CAMERA_VFOV_DEG"])
        horizontal_fov_deg = float(self.speed_cfg["CAMERA_HFOV_DEG"])
        ema_alpha = float(self.speed_cfg["EMA_ALPHA"])
        self.speed_estimator = VehicleSpeed(
            fps=self.video_fps,
            camera_height=camera_height,
            camera_angle_deg=camera_angle_deg,
            vertical_fov_deg=vertical_fov_deg,
            horizontal_fov_deg=horizontal_fov_deg,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            ema_alpha=ema_alpha,
        )

    def _tracker_config(self, tracker_cfg: dict) -> dict:
        """Build the DeepSORT configuration."""
        return {
            "DEEPSORT": {
                "REID_CKPT": str(self._resolve_path(self.paths_cfg["REID_CKPT"])),
                "MAX_DIST": tracker_cfg["MAX_DIST"],
                "MIN_CONFIDENCE": tracker_cfg["MIN_CONFIDENCE"],
                "NMS_MAX_OVERLAP": tracker_cfg["NMS_MAX_OVERLAP"],
                "MAX_IOU_DISTANCE": tracker_cfg["MAX_IOU_DISTANCE"],
                "MAX_AGE": tracker_cfg["MAX_AGE"],
                "N_INIT": tracker_cfg["N_INIT"],
                "NN_BUDGET": tracker_cfg["NN_BUDGET"],
            }
        }

    def _open_video_io(self) -> Tuple[cv2.VideoCapture, cv2.VideoWriter, int, int, float]:
        """Open input and create an output video with the source frame rate."""
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise SystemExit(f"Unable to open video: {self.video_path}")

        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            capture.release()
            raise SystemExit(f"Unable to read a valid frame rate from video: {self.video_path}")
        writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            capture.release()
            raise SystemExit(f"Unable to create output video: {self.output_path}")
        return capture, writer, frame_width, frame_height, fps

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

    @staticmethod
    def _compute_color(track_id: int) -> Tuple[int, int, int]:
        color = [int((p * (track_id ** 2 - track_id + 1)) % 255) for p in PALETTE]
        return tuple(color)

    def _draw_track_box(self, frame: np.ndarray, box: np.ndarray, track_id: int,
                        class_name: str, confidence: float) -> None:
        x1, y1, x2, y2 = [int(v) for v in box]
        color = self._compute_color(track_id)
        label = f"{class_name} {confidence:.2f} ID:{track_id}"
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

    def _draw_speed_label(self, frame: np.ndarray, box: np.ndarray, track_id: int) -> None:
        if not self.speed_enabled:
            return
        speed = self.speed_estimator.get_speed(track_id)
        if speed > 0:
            cv2.putText(
                frame,
                f"{speed:.1f} km/h",
                (int(box[0]), int(box[3]) + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.text_scale,
                self.speed_color,
                self.text_thickness,
            )

    def _update_speed(self, frame: np.ndarray, box: np.ndarray, track_id: int, frame_index: int) -> None:
        if not self.speed_enabled:
            return
        contact_pixel_x = (box[0] + box[2]) / 2.0
        contact_pixel_y = box[3]  # Vehicle ground-contact pixel.
        self.speed_estimator.update(track_id, contact_pixel_x, contact_pixel_y, frame_index)
        self._draw_speed_label(frame, box, track_id)

    def _detect(self, frame: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Run YOLO and return DeepSORT-ready detections."""
        result = self.model.predict(
            frame,
            imgsz=self.img_size,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=self.class_ids,
            device=self.device,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return torch.empty((0, 4)), torch.empty((0,)), np.array([], dtype=int)

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        boxes_xywh = self._xyxy_to_xywh(boxes_xyxy)
        return torch.from_numpy(boxes_xywh), torch.from_numpy(confidences), class_ids

    @staticmethod
    def _xyxy_to_xywh(boxes_xyxy: np.ndarray) -> np.ndarray:
        """Convert boxes from corner coordinates to centre coordinates."""
        boxes_xywh = np.empty_like(boxes_xyxy, dtype=np.float32)
        boxes_xywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
        boxes_xywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0
        boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        return boxes_xywh

    def _track(self, frame: np.ndarray) -> TrackOutput:
        """Detect vehicles and update DeepSORT."""
        boxes_xywh, confidences, class_ids = self._detect(frame)
        if boxes_xywh.shape[0] == 0:
            return np.empty((0, 7), dtype=float)
        outputs = self.deepsort.update(boxes_xywh, confidences, class_ids, frame)
        return outputs if len(outputs) > 0 else np.empty((0, 7), dtype=float)

    def _retire_missing_tracks(self, outputs: TrackOutput, active_ids: set[int]) -> set[int]:
        """Forget speed state for tracks no longer returned by DeepSORT."""
        current_ids = {int(output[4]) for output in outputs}
        for track_id in active_ids - current_ids:
            self.speed_estimator.forget(track_id)
        return current_ids

    def _annotate_tracks(self, frame: np.ndarray, outputs: TrackOutput, frame_index: int) -> None:
        """Draw tracked vehicles and their current speeds."""
        for output in outputs:
            box = output[:4]
            track_id = int(output[4])
            class_id = int(output[5])
            confidence = float(output[6])
            class_name = self.model.names.get(class_id, f"cls_{class_id}")
            self._draw_track_box(frame, box, track_id, class_name, confidence)
            self._update_speed(frame, box, track_id, frame_index)

    def run(self) -> None:
        """Process the video frame by frame."""
        frame_index = 0
        active_ids: set[int] = set()
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break
                frame_index += 1

                outputs = self._track(frame)
                active_ids = self._retire_missing_tracks(outputs, active_ids)
                self._annotate_tracks(frame, outputs, frame_index)
                self.writer.write(frame)
        finally:
            self.writer.release()
            self.cap.release()

        print(f"Wrote {self.output_path}")


def load_cfg(cfg_path: str) -> dict:
    """Load and validate the required runtime configuration."""
    with open(cfg_path, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise SystemExit(f"Configuration must be a YAML mapping: {cfg_path}")

    missing = []
    for section, keys in REQUIRED_CONFIG.items():
        values = cfg.get(section)
        if not isinstance(values, dict):
            missing.append(section)
            continue
        missing.extend(f"{section}.{key}" for key in keys if values.get(key) is None)
    if missing:
        raise SystemExit(f"Configuration is incomplete: {', '.join(missing)}")
    return cfg


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
