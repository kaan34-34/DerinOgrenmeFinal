import os
from argparse import ArgumentParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_CACHE = PROJECT_ROOT / ".cache"
os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_CACHE / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE))


def parse_args():
    parser = ArgumentParser(description="Evaluate a trained YOLOv8 brain tumor detector.")
    parser.add_argument(
        "--weights",
        default="runs/detect/brain_tumor_yolov8n/weights/best.pt",
        help="Path to trained weights.",
    )
    parser.add_argument("--data", default="brain-tumor.yaml", help="YOLO dataset YAML path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
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

    if not Path(args.weights).exists():
        raise SystemExit(f"Weights not found: {args.weights}")

    model = YOLO(args.weights)
    val_args = {"data": args.data, "imgsz": args.imgsz, "plots": True}
    if args.device:
        val_args["device"] = args.device
    metrics = model.val(**val_args)

    print("Validation metrics")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP75:    {metrics.box.map75:.4f}")


if __name__ == "__main__":
    main()
