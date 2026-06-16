from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_label_classes(label_path: Path) -> list[int]:
    classes: list[int] = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return classes

    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number} expected 5 YOLO columns, got {len(parts)}")
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
        if class_id not in {0, 1}:
            raise ValueError(f"{label_path}:{line_number} invalid class id {class_id}")
        if any(value < 0 or value > 1 for value in values):
            raise ValueError(f"{label_path}:{line_number} bbox values must be normalized between 0 and 1")
        classes.append(class_id)

    return classes


def validate_split(dataset_root: Path, split: str) -> tuple[int, int, dict[int, int], int]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    class_counts = {0: 0, 1: 0}
    empty_labels = 0

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")

        classes = read_label_classes(label_path)
        if not classes:
            empty_labels += 1
        for class_id in classes:
            class_counts[class_id] += 1

    return len(images), len(list(label_dir.glob("*.txt"))), class_counts, empty_labels


def main() -> None:
    dataset_root = Path("brain-tumor")
    total_boxes = 0

    for split in ("train", "val"):
        image_count, label_count, class_counts, empty_labels = validate_split(dataset_root, split)
        split_boxes = sum(class_counts.values())
        total_boxes += split_boxes
        print(f"{split}: {image_count} images, {label_count} labels, {split_boxes} boxes")
        print(f"  negative boxes: {class_counts[0]}")
        print(f"  positive boxes: {class_counts[1]}")
        print(f"  empty label files: {empty_labels}")

    print(f"Dataset validation completed. Total boxes: {total_boxes}")


if __name__ == "__main__":
    main()
