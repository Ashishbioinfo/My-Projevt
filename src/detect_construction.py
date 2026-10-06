"""Train and run a Sentinel-2 new-construction change detector."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report

BAND_NAMES = ("B02", "B03", "B04", "B08", "B11", "B12")
INDEX_NAMES = ("NDVI", "NDBI", "NDWI")
FEATURE_NAMES = tuple(
    [f"before_{name}" for name in BAND_NAMES]
    + [f"after_{name}" for name in BAND_NAMES]
    + [f"delta_{name}" for name in BAND_NAMES]
    + [f"before_{name}" for name in INDEX_NAMES]
    + [f"after_{name}" for name in INDEX_NAMES]
    + [f"delta_{name}" for name in INDEX_NAMES]
)


def _indices(image: np.ndarray) -> np.ndarray:
    """Calculate NDVI, NDBI, and NDWI from B02/B03/B04/B08/B11/B12 bands."""
    blue, green, red, nir, swir1, _swir2 = image

    def normalized_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        denominator = first + second
        return np.divide(
            first - second,
            denominator,
            out=np.zeros_like(first, dtype=np.float32),
            where=np.abs(denominator) > 1e-6,
        )

    return np.stack(
        (
            normalized_difference(nir, red),
            normalized_difference(swir1, nir),
            normalized_difference(green, nir),
        )
    )


def make_features(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Return an H x W x 27 array of two-date bands, indices, and differences."""
    if before.shape != after.shape or before.ndim != 3 or before.shape[0] != 6:
        raise ValueError("Each image must have matching shape (6, height, width).")

    before = before.astype(np.float32, copy=False)
    after = after.astype(np.float32, copy=False)
    before_indices = _indices(before)
    after_indices = _indices(after)
    channels = np.concatenate(
        (
            before,
            after,
            after - before,
            before_indices,
            after_indices,
            after_indices - before_indices,
        ),
        axis=0,
    )
    return np.moveaxis(channels, 0, -1)


def _check_pair(before: rasterio.io.DatasetReader, after: rasterio.io.DatasetReader) -> None:
    if before.count != 6 or after.count != 6:
        raise ValueError("Before and after GeoTIFFs must each contain 6 bands: B02,B03,B04,B08,B11,B12.")
    if (
        before.width != after.width
        or before.height != after.height
        or before.crs != after.crs
        or before.transform != after.transform
    ):
        raise ValueError("Before and after rasters must have identical dimensions, CRS, and transform.")


def _read_training_data(
    before_path: Path, after_path: Path, labels_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    with rasterio.open(before_path) as before, rasterio.open(after_path) as after, rasterio.open(labels_path) as labels:
        _check_pair(before, after)
        if labels.count != 1 or (
            labels.width != before.width
            or labels.height != before.height
            or labels.crs != before.crs
            or labels.transform != before.transform
        ):
            raise ValueError("Labels must be a one-band raster aligned exactly to the input images.")

        before_data = before.read().astype(np.float32)
        after_data = after.read().astype(np.float32)
        label_data = labels.read(1)
        valid = (
            (before.read_masks().all(axis=0) > 0)
            & (after.read_masks().all(axis=0) > 0)
            & (labels.read_masks(1) > 0)
            & np.isfinite(before_data).all(axis=0)
            & np.isfinite(after_data).all(axis=0)
            & np.isin(label_data, (0, 1))
        )
        height, width = label_data.shape
        features = make_features(before_data, after_data).reshape(-1, len(FEATURE_NAMES))
        labels_flat = label_data.reshape(-1).astype(np.uint8)
        valid_flat = valid.reshape(-1)

        pixel_indices = np.flatnonzero(valid_flat)
        group_columns = (width + 127) // 128
        rows = pixel_indices // width
        columns = pixel_indices % width
        groups = (rows // 128) * group_columns + columns // 128
        return features[pixel_indices], labels_flat[pixel_indices], groups, width


def _spatial_split(features: np.ndarray, labels: np.ndarray, groups: np.ndarray):
    splitter = GroupShuffleSplit(n_splits=30, test_size=0.2, random_state=42)
    for train_indices, test_indices in splitter.split(features, labels, groups):
        if np.unique(labels[train_indices]).size == 2 and np.unique(labels[test_indices]).size == 2:
            return train_indices, test_indices
    raise ValueError(
        "Could not form spatial train/test sets containing both classes. Add labeled pixels "
        "across more 128-pixel blocks or use larger, geographically varied training chips."
    )


def train_model(
    before_path: Path,
    after_path: Path,
    labels_path: Path,
    model_path: Path,
    max_samples_per_class: int,
) -> None:
    features, labels, groups, _width = _read_training_data(before_path, after_path, labels_path)
    rng = np.random.default_rng(42)
    selected: list[np.ndarray] = []
    for class_id in (0, 1):
        class_indices = np.flatnonzero(labels == class_id)
        if class_indices.size == 0:
            raise ValueError(f"Training labels do not contain class {class_id}.")
        if class_indices.size > max_samples_per_class:
            class_indices = rng.choice(class_indices, size=max_samples_per_class, replace=False)
        selected.append(class_indices)

    sample_indices = np.concatenate(selected)
    train_indices, test_indices = _spatial_split(
        features[sample_indices], labels[sample_indices], groups[sample_indices]
    )
    sampled_features = features[sample_indices]
    sampled_labels = labels[sample_indices]
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    classifier.fit(sampled_features[train_indices], sampled_labels[train_indices])
    predictions = classifier.predict(sampled_features[test_indices])
    print("Spatial holdout evaluation:")
    print(classification_report(sampled_labels[test_indices], predictions, target_names=("not_new_construction", "new_construction"), zero_division=0))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": classifier, "feature_names": FEATURE_NAMES, "band_names": BAND_NAMES},
        model_path,
    )
    print(f"Saved model: {model_path}")


def predict(
    before_path: Path,
    after_path: Path,
    model_path: Path,
    output_path: Path,
    threshold: float,
) -> None:
    saved = joblib.load(model_path)
    classifier = saved["model"]
    if tuple(saved["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Model feature schema does not match this script.")
    positive_class = np.flatnonzero(classifier.classes_ == 1)
    if positive_class.size != 1:
        raise ValueError("Saved model does not contain the new-construction class (1).")

    probability_path = output_path.with_name(f"{output_path.stem}_probability.tif")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(before_path) as before, rasterio.open(after_path) as after:
        _check_pair(before, after)
        mask_profile = before.profile.copy()
        mask_profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
        probability_profile = before.profile.copy()
        probability_profile.update(count=1, dtype="float32", nodata=-9999.0, compress="deflate")

        with rasterio.open(output_path, "w", **mask_profile) as mask_output, rasterio.open(
            probability_path, "w", **probability_profile
        ) as probability_output:
            for _block_id, window in before.block_windows(1):
                before_data = before.read(window=window).astype(np.float32)
                after_data = after.read(window=window).astype(np.float32)
                valid = (
                    (before.read_masks(window=window).all(axis=0) > 0)
                    & (after.read_masks(window=window).all(axis=0) > 0)
                    & np.isfinite(before_data).all(axis=0)
                    & np.isfinite(after_data).all(axis=0)
                )
                features = make_features(before_data, after_data)
                mask = np.full(valid.shape, 255, dtype=np.uint8)
                probabilities = np.full(valid.shape, -9999.0, dtype=np.float32)
                if valid.any():
                    pixel_probabilities = classifier.predict_proba(features[valid])[:, positive_class[0]]
                    probabilities[valid] = pixel_probabilities
                    mask[valid] = (pixel_probabilities >= threshold).astype(np.uint8)
                mask_output.write(mask, 1, window=window)
                probability_output.write(probabilities, 1, window=window)

    print(f"Saved detection mask: {output_path} (1=new construction, 0=not detected, 255=nodata)")
    print(f"Saved probability raster: {probability_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sentinel-2 new-construction change detection")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Train from paired images and a 0/1 label raster")
    train_parser.add_argument("--before", type=Path, required=True, help="Earlier six-band Sentinel-2 GeoTIFF")
    train_parser.add_argument("--after", type=Path, required=True, help="Later six-band Sentinel-2 GeoTIFF")
    train_parser.add_argument("--labels", type=Path, required=True, help="Aligned labels: 0=not new construction, 1=new construction")
    train_parser.add_argument("--model", type=Path, default=Path("models/construction_rf.joblib"))
    train_parser.add_argument("--max-samples-per-class", type=int, default=100_000)

    predict_parser = commands.add_parser("predict", help="Predict new construction for a paired image")
    predict_parser.add_argument("--before", type=Path, required=True)
    predict_parser.add_argument("--after", type=Path, required=True)
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True, help="Output binary GeoTIFF mask")
    predict_parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold from 0 to 1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        if args.max_samples_per_class < 1:
            raise SystemExit("--max-samples-per-class must be positive")
        train_model(args.before, args.after, args.labels, args.model, args.max_samples_per_class)
    else:
        if not 0 <= args.threshold <= 1:
            raise SystemExit("--threshold must be between 0 and 1")
        predict(args.before, args.after, args.model, args.output, args.threshold)


if __name__ == "__main__":
    main()
