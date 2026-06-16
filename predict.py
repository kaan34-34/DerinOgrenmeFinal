import os
from argparse import ArgumentParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_CACHE = PROJECT_ROOT / ".cache"
os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_CACHE / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE))


def parse_args():
    parser = ArgumentParser(description="Run inference with a trained brain tumor detector.")
    parser.add_argument(
        "--weights",
        default="runs/detect/brain_tumor_yolov8n/weights/best.pt",
        help="Path to trained weights.",
    )
    parser.add_argument("--source", default="brain-tumor/images/val", help="Image, folder, or video source.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
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
    predict_args = {
        "source": args.source,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "save": True,
        "save_txt": True,
    }
    if args.device:
        predict_args["device"] = args.device
    model.predict(**predict_args)


if __name__ == "__main__":
    main()
