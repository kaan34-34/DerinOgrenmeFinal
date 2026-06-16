import os
from argparse import ArgumentParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_CACHE = PROJECT_ROOT / ".cache"
os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_CACHE / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE))


def parse_args():
    parser = ArgumentParser(description="Train a YOLOv8 brain tumor detector.")
    parser.add_argument("--data", default="brain-tumor.yaml", help="YOLO dataset YAML path.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model, e.g. yolov8n.pt.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--project", default=None, help="Optional output project directory.")
    parser.add_argument("--name", default="brain_tumor_yolov8n", help="Run name.")
    parser.add_argument("--device", default=None, help="Device: cpu, mps, cuda, 0, etc.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing ultralytics. Install dependencies with: "
            "python3 -m pip install -r requirements.txt"
        ) from exc

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset YAML not found: {data_path}")

    model = YOLO(args.model)
    train_args = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "name": args.name,
        "exist_ok": True,
        "plots": True,
    }
    if args.project:
        train_args["project"] = args.project
    if args.device:
        train_args["device"] = args.device

    results = model.train(**train_args)
    print(results)


if __name__ == "__main__":
    main()
