import csv
import random
from pathlib import Path

from src.config import (
    CLASS_LIST_PATH, DATA_PROCESSED_DIR, DOMAIN_ADAPT_VAL_RATIO, IMAGES_PER_CLASS,
    MANIFEST_JOINT_TRAIN, MANIFEST_JOINT_VAL, MANIFEST_OOD, MANIFEST_OOD_TEST,
    MANIFEST_OOD_TRAIN, MANIFEST_OOD_VAL, MANIFEST_TEST, MANIFEST_TRAIN,
    MANIFEST_VAL, PLANTDOC_OVERSAMPLE_FACTOR, PLANTDOC_RAW_DIR,
    PLANTVILLAGE_RAW_DIR, PLANTVILLAGE_SEGMENTED_RAW_DIR, SEED, SPLIT_RATIOS,
    USE_SEGMENTED_PLANTVILLAGE_AUGMENTATION,
)
from src.utils import save_json

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def scan_classes(raw_dir: Path) -> dict:
    class_to_files = {}
    for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        files = [str(p) for p in class_dir.iterdir() if p.suffix in VALID_EXTENSIONS]
        if files:
            class_to_files[class_dir.name] = files
    return class_to_files


def stratified_subsample(class_to_files: dict, per_class: int, seed: int) -> dict:
    rng = random.Random(seed)
    subsampled = {}
    for class_name, files in class_to_files.items():
        files = files.copy()
        rng.shuffle(files)
        subsampled[class_name] = files[:per_class]
    return subsampled


def split_files(files, ratios, seed: int):
    rng = random.Random(seed)
    files = files.copy()
    rng.shuffle(files)
    n = len(files)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return files[:n_train], files[n_train:n_train + n_val], files[n_train + n_val:]


def write_manifest(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "label_idx"])
        writer.writerows(rows)


def build_plantvillage_manifests(images_per_class: int = IMAGES_PER_CLASS):
    print(f"Scanning PlantVillage raw dir: {PLANTVILLAGE_RAW_DIR}")
    class_to_files = scan_classes(PLANTVILLAGE_RAW_DIR)
    if not class_to_files:
        raise RuntimeError(f"No classes found under {PLANTVILLAGE_RAW_DIR}")
    print(f"Found {len(class_to_files)} classes, "
          f"{sum(len(v) for v in class_to_files.values())} images total")

    subsampled = stratified_subsample(class_to_files, images_per_class, SEED)
    classes = sorted(subsampled.keys())
    class_to_idx = {name: i for i, name in enumerate(classes)}
    save_json(classes, CLASS_LIST_PATH)

    train_rows, val_rows, test_rows = [], [], []
    for class_name, files in subsampled.items():
        idx = class_to_idx[class_name]
        train_files, val_files, test_files = split_files(files, SPLIT_RATIOS, SEED)
        train_rows += [(f, class_name, idx) for f in train_files]
        val_rows += [(f, class_name, idx) for f in val_files]
        test_rows += [(f, class_name, idx) for f in test_files]

    write_manifest(MANIFEST_TRAIN, train_rows)
    write_manifest(MANIFEST_VAL, val_rows)
    write_manifest(MANIFEST_TEST, test_rows)
    print(f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} "
          f"classes={len(classes)}")
    return classes, class_to_idx


PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}


def _scan_plantdoc_split(split_dir: Path, class_to_idx: dict):
    rows = []
    missing_targets = []
    if not split_dir.exists():
        return rows, missing_targets
    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():
            continue
        target_class = PLANTDOC_TO_PLANTVILLAGE.get(class_dir.name)
        if target_class is None:
            continue
        if target_class not in class_to_idx:
            missing_targets.append(target_class)
            continue
        idx = class_to_idx[target_class]
        for f in class_dir.iterdir():
            if f.suffix in VALID_EXTENSIONS:
                rows.append((str(f), target_class, idx))
    return rows, missing_targets


def build_ood_manifest(class_to_idx: dict):
    if not PLANTDOC_RAW_DIR.exists():
        print(f"PlantDoc raw dir not found at {PLANTDOC_RAW_DIR}, skipping OOD manifest")
        return

    train_rows, missing_train = _scan_plantdoc_split(PLANTDOC_RAW_DIR / "train", class_to_idx)
    test_rows, missing_test = _scan_plantdoc_split(PLANTDOC_RAW_DIR / "test", class_to_idx)

    missing_targets = set(missing_train) | set(missing_test)
    if missing_targets:
        print(f"Warning: {len(missing_targets)} mapped classes not in PlantVillage class list: "
              f"{sorted(missing_targets)}")

    by_class = {}
    for row in train_rows:
        by_class.setdefault(row[2], []).append(row)
    ft_train_rows, ft_val_rows = [], []
    for idx, rows in by_class.items():
        tr, va, _ = split_files(rows, (1 - DOMAIN_ADAPT_VAL_RATIO, DOMAIN_ADAPT_VAL_RATIO, 0), SEED)
        ft_train_rows += tr
        ft_val_rows += va

    write_manifest(MANIFEST_OOD_TRAIN, ft_train_rows)
    write_manifest(MANIFEST_OOD_VAL, ft_val_rows)
    write_manifest(MANIFEST_OOD_TEST, test_rows)
    write_manifest(MANIFEST_OOD, train_rows + test_rows)
    print(f"OOD (PlantDoc) manifests: fine-tune-train={len(ft_train_rows)} "
          f"fine-tune-val={len(ft_val_rows)} held-out-test={len(test_rows)} "
          f"across {len(set(r[1] for r in train_rows + test_rows))} mapped classes")


def _read_manifest_rows(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [(row["filepath"], row["label"], int(row["label_idx"])) for row in reader]


def _segmented_lookup_for_class(class_name: str) -> dict:
    class_dir = PLANTVILLAGE_SEGMENTED_RAW_DIR / class_name
    lookup = {}
    if not class_dir.exists():
        return lookup
    for f in class_dir.iterdir():
        if f.suffix not in VALID_EXTENSIONS:
            continue
        stem = f.stem
        if stem.endswith("_final_masked"):
            stem = stem[: -len("_final_masked")]
        lookup[stem] = str(f)
    return lookup


def build_segmented_plantvillage_rows(pv_train_rows):
    rows = []
    lookup_cache = {}
    missing = 0
    for filepath, class_name, idx in pv_train_rows:
        if class_name not in lookup_cache:
            lookup_cache[class_name] = _segmented_lookup_for_class(class_name)
        stem = Path(filepath).stem
        segmented_path = lookup_cache[class_name].get(stem)
        if segmented_path is None:
            missing += 1
            continue
        rows.append((segmented_path, class_name, idx))
    if missing:
        print(f"Warning: {missing} PlantVillage train images had no segmented counterpart")
    return rows


def build_joint_manifests():
    if not MANIFEST_OOD_TRAIN.exists():
        print("No PlantDoc OOD-train manifest yet, skipping joint manifests")
        return

    pv_train = _read_manifest_rows(MANIFEST_TRAIN)
    pv_val = _read_manifest_rows(MANIFEST_VAL)
    pd_train = _read_manifest_rows(MANIFEST_OOD_TRAIN)
    pd_val = _read_manifest_rows(MANIFEST_OOD_VAL)

    joint_train = list(pv_train)

    segmented_rows = []
    if USE_SEGMENTED_PLANTVILLAGE_AUGMENTATION:
        segmented_rows = build_segmented_plantvillage_rows(pv_train)
        joint_train += segmented_rows

    oversampled_pd_train = pd_train * PLANTDOC_OVERSAMPLE_FACTOR
    joint_train += oversampled_pd_train

    write_manifest(MANIFEST_JOINT_TRAIN, joint_train)
    write_manifest(MANIFEST_JOINT_VAL, pv_val + pd_val)
    print(f"Joint manifests: train={len(joint_train)} "
          f"(PlantVillage-color {len(pv_train)} + PlantVillage-segmented {len(segmented_rows)} + "
          f"PlantDoc x{PLANTDOC_OVERSAMPLE_FACTOR} {len(oversampled_pd_train)}), "
          f"val={len(pv_val) + len(pd_val)} (PlantVillage {len(pv_val)} + PlantDoc {len(pd_val)})")


def main(images_per_class: int = IMAGES_PER_CLASS):
    classes, class_to_idx = build_plantvillage_manifests(images_per_class)
    build_ood_manifest(class_to_idx)
    build_joint_manifests()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-per-class", type=int, default=IMAGES_PER_CLASS)
    args = parser.parse_args()
    main(args.images_per_class)
