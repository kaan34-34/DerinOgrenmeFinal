# Brain Tumor Detection with YOLOv8

This project trains a YOLOv8 object detection model on the extracted
`brain-tumor` dataset.

## Dataset

The dataset is already in YOLO format:

- `brain-tumor/images/train`
- `brain-tumor/images/val`
- `brain-tumor/labels/train`
- `brain-tumor/labels/val`

Classes:

- `0`: negative
- `1`: positive

## Setup

Install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Validate the dataset:

```bash
.venv/bin/python validate_dataset.py
```

## Training

The assignment command is:

```bash
.venv/bin/yolo detect train data=brain-tumor.yaml model=yolov8n.pt epochs=50 imgsz=640
```

The same training can be started with the Python script:

```bash
.venv/bin/python train.py --model yolov8n.pt --epochs 50 --imgsz 640
```

On Apple Silicon, this can use Metal when available:

```bash
.venv/bin/python train.py --device mps --model yolov8n.pt --epochs 50 --imgsz 640
```

Outputs are saved under:

```text
runs/detect/brain_tumor_yolov8n
```

Important files after training:

- `weights/best.pt`
- `weights/last.pt`
- `results.png`
- `confusion_matrix.png`
- `PR_curve.png`

## Evaluation

```bash
.venv/bin/python evaluate.py --weights runs/detect/brain_tumor_yolov8n/weights/best.pt
```

## Prediction

```bash
.venv/bin/python predict.py --weights runs/detect/brain_tumor_yolov8n/weights/best.pt --source brain-tumor/images/val
```

Predicted images and labels are saved under `runs/detect/predict`.
