import argparse
from pathlib import Path
import shutil

import yaml
from ultralytics import YOLO


def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 from YAML config")
    parser.add_argument("--cfg_path", type=str, default="../cfgs/yolov8s.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg_path)
    train_cfg = cfg["TRAIN"]
    path_cfg = cfg["PATHS"]

    script_dir = Path(__file__).resolve().parent
    model_name = train_cfg["MODEL"]
    data_path = resolve_path(script_dir, path_cfg["DATASET_YAML"])
    project_path = resolve_path(script_dir, path_cfg["TRAIN_PROJECT"])
    export_best_to = resolve_path(script_dir, path_cfg["WEIGHTS_OUT"])

    model = YOLO(model_name)
    model.train(
        data=str(data_path),
        device=train_cfg["DEVICE"],
        epochs=int(train_cfg["EPOCHS"]),
        imgsz=int(train_cfg["IMG_SIZE"]),
        batch=int(train_cfg["BATCH"]),
        workers=int(train_cfg["WORKERS"]),
        patience=int(train_cfg["PATIENCE"]),
        project=str(project_path),
        name=train_cfg["RUN_NAME"],
        exist_ok=bool(train_cfg["EXIST_OK"]),
        verbose=bool(train_cfg["VERBOSE"]),
    )

    best_path = project_path / train_cfg["RUN_NAME"] / "weights" / "best.pt"
    if best_path.exists():
        export_best_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, export_best_to)
        print(f"Saved best model to {export_best_to}")
    else:
        print(f"Training completed, but best.pt not found at {best_path}")


if __name__ == "__main__":
    main()
