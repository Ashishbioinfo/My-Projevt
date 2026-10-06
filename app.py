"""Streamlit dashboard for corporation, city, and village monitoring."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
from datetime import date, timedelta
from math import ceil
from pathlib import Path

_rasterio_spec = importlib.util.find_spec("rasterio")
if _rasterio_spec and _rasterio_spec.submodule_search_locations:
    _bundled_proj_data = Path(next(iter(_rasterio_spec.submodule_search_locations))) / "proj_data"
    if _bundled_proj_data.is_dir():
        os.environ["PROJ_LIB"] = str(_bundled_proj_data)

import folium
import numpy as np
import rasterio
import streamlit as st
from PIL import Image as PillowImage, ImageDraw, ImageFont
from folium.plugins import Fullscreen
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from streamlit_folium import st_folium

from src.detect_construction import predict
from src.sentinel_imagery import (
    DEFAULT_PREVIEW_RADIUS_KM,
    analyze_l2a_change,
    analyze_sar_vertical_candidates,
    add_metric_scale_control,
    fetch_sentinel_preview,
    fetch_sentinel1_rtc,
    make_sentinel_preview_map,
    summarize_candidate_patches,
)
from src.zone_report import create_zone_report_pdf

ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "data" / "locations.json"
OUTPUT_DIR = ROOT / "outputs"
SENTINEL_ANALYSIS_CACHE_VERSION = 2
CANDIDATE_CONTEXT_BUFFER_METERS = 200

st.set_page_config(page_title="Construction Watch", page_icon="M", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: #f4f7f4; color: #1c3027; }
      [data-testid="stSidebar"] { background: #e9f0eb; border-right: 1px solid #d5e0d8; }
      [data-testid="stMetric"] { background: #fff; border-left: 3px solid #d85b47; padding: 0.6rem 0.8rem; }
      h1, h2, h3 { color: #183b2d; }
      div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMetric"]) { gap: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_registry(path: Path = REGISTRY_PATH) -> list[dict]:
    """Load and minimally validate the configured location hierarchy."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Location registry not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error.msg}") from error

    corporations = data.get("corporations") if isinstance(data, dict) else None
    if not isinstance(corporations, list):
        raise ValueError('The registry must contain a "corporations" list.')
    for corporation in corporations:
        if not isinstance(corporation, dict) or not corporation.get("name"):
            raise ValueError("Every corporation needs a name and a cities list.")
        if not isinstance(corporation.get("cities"), list):
            raise ValueError(f'Corporation "{corporation["name"]}" needs a cities list.')
        for city in corporation["cities"]:
            if not isinstance(city, dict) or not city.get("name"):
                raise ValueError(f'Every city in "{corporation["name"]}" needs a name.')
            zones = city.get("zones")
            if zones is None:
                area_groups = [{"name": city["name"], "villages": city.get("villages")}]
            elif isinstance(zones, list):
                area_groups = zones
            else:
                raise ValueError(f'Zones in "{city["name"]}" must be a list.')
            for area_group in area_groups:
                if not isinstance(area_group, dict) or not area_group.get("name"):
                    raise ValueError(f'Every zone in "{city["name"]}" needs a name.')
                if not isinstance(area_group.get("villages"), list):
                    raise ValueError(f'Zone "{area_group["name"]}" needs a villages list.')
                for village in area_group["villages"]:
                    if not isinstance(village, dict) or not village.get("name"):
                        raise ValueError(f'Every village in "{area_group["name"]}" needs a name.')
    return corporations


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower() or "location"


def google_satellite_url(latitude: float, longitude: float, zoom: int = 19) -> str:
    return f"https://www.google.com/maps/@{latitude:.6f},{longitude:.6f},{zoom}z/data=!3m1!1e3"


def prediction_paths(
    corporation: str,
    city: str,
    village: str,
    before_path: Path,
    after_path: Path,
    model_path: Path,
    threshold: float,
) -> tuple[Path, Path]:
    inputs = (before_path, after_path, model_path)
    fingerprint = hashlib.sha256(
        json.dumps(
            [(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in inputs]
            + [("threshold", threshold)],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:10]
    location = "_".join(_slug(part) for part in (corporation, city, village))
    mask_path = OUTPUT_DIR / f"{location}_{fingerprint}.tif"
    probability_path = mask_path.with_name(f"{mask_path.stem}_probability.tif")
    return mask_path, probability_path


def ensure_prediction(
    before_path: Path,
    after_path: Path,
    model_path: Path,
    mask_path: Path,
    probability_path: Path,
    threshold: float,
) -> None:
    if mask_path.exists() and probability_path.exists():
        return
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    predict(before_path, after_path, model_path, mask_path, threshold)


def read_detection_overlay(mask_path: Path, max_dimension: int = 1200) -> tuple[np.ndarray, list[list[float]], tuple[float, float], int, int]:
    """Read a downsampled mask and prepare a transparent map overlay."""
    with rasterio.open(mask_path) as dataset:
        scale = min(1.0, max_dimension / max(dataset.width, dataset.height))
        height = max(1, round(dataset.height * scale))
        width = max(1, round(dataset.width * scale))
        mask = dataset.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.nearest,
        )
        bounds = transform_bounds(dataset.crs, "EPSG:4326", *dataset.bounds, densify_pts=21)

    west, south, east, north = bounds
    image_bounds = [[south, west], [north, east]]
    center = ((south + north) / 2, (west + east) / 2)
    detected = mask == 1
    valid = mask != 255
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[detected] = (220, 76, 58, 220)
    return rgba, image_bounds, center, int(detected.sum()), int(valid.sum())


def make_map(overlay: np.ndarray, bounds: list[list[float]], center: tuple[float, float], key: str) -> folium.Map:
    map_view = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap", control_scale=False)
    folium.raster_layers.ImageOverlay(
        image=overlay,
        bounds=bounds,
        name="Detected construction",
        opacity=0.85,
        interactive=False,
        cross_origin=False,
        zindex=2,
    ).add_to(map_view)
    folium.LayerControl(collapsed=False).add_to(map_view)
    add_metric_scale_control(map_view)
    Fullscreen(position="topright").add_to(map_view)
    map_view.fit_bounds(bounds)
    return map_view


def zoom_preview(image: np.ndarray, zoom_factor: float) -> np.ndarray:
    """Center-crop and smoothly resample an RGB preview for map display."""
    height, width = image.shape[:2]
    if zoom_factor > 1:
        crop_height = max(1, round(height / zoom_factor))
        crop_width = max(1, round(width / zoom_factor))
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        image = image[top : top + crop_height, left : left + crop_width]

    crop_height, crop_width = image.shape[:2]
    output_long_edge = max(512, crop_height, crop_width)
    output_height = max(1, round(output_long_edge * crop_height / max(crop_height, crop_width)))
    output_width = max(1, round(output_long_edge * crop_width / max(crop_height, crop_width)))
    return np.asarray(
        PillowImage.fromarray(image).resize(
            (output_width, output_height),
            resample=PillowImage.Resampling.LANCZOS,
        )
    )


def enlarge_candidate_crop(image: np.ndarray, patch: dict[str, int | float | str]) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    patch_height = int(patch["row_end"]) - int(patch["row_start"])
    patch_width = int(patch["column_end"]) - int(patch["column_start"])
    meters_per_row = 2 * DEFAULT_PREVIEW_RADIUS_KM * 1000 / image_height
    meters_per_column = 2 * DEFAULT_PREVIEW_RADIUS_KM * 1000 / image_width
    padding_rows = max(ceil(CANDIDATE_CONTEXT_BUFFER_METERS / meters_per_row), patch_height // 3)
    padding_columns = max(ceil(CANDIDATE_CONTEXT_BUFFER_METERS / meters_per_column), patch_width // 3)
    row_start = max(0, int(patch["row_start"]) - padding_rows)
    row_end = min(image_height, int(patch["row_end"]) + padding_rows)
    column_start = max(0, int(patch["column_start"]) - padding_columns)
    column_end = min(image_width, int(patch["column_end"]) + padding_columns)
    crop = image[
        row_start:row_end,
        column_start:column_end,
    ]
    if crop.size == 0:
        return crop
    crop_height, crop_width = crop.shape[:2]
    output_long_edge = min(1024, max(640, max(crop_height, crop_width) * 4))
    output_height = max(1, round(output_long_edge * crop_height / max(crop_height, crop_width)))
    output_width = max(1, round(output_long_edge * crop_width / max(crop_height, crop_width)))
    enlarged = PillowImage.fromarray(crop).resize(
        (output_width, output_height),
        resample=PillowImage.Resampling.LANCZOS,
    ).convert("RGBA")
    marker = PillowImage.new("RGBA", enlarged.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(marker)
    patch_top = round((int(patch["row_start"]) - row_start) * output_height / crop_height)
    patch_bottom = round((int(patch["row_end"]) - row_start) * output_height / crop_height)
    patch_left = round((int(patch["column_start"]) - column_start) * output_width / crop_width)
    patch_right = round((int(patch["column_end"]) - column_start) * output_width / crop_width)
    bounds = (patch_left, patch_top, max(patch_left, patch_right - 1), max(patch_top, patch_bottom - 1))
    draw.rectangle(bounds, fill=(234, 53, 44, 64), outline=(255, 242, 92, 255), width=max(5, output_long_edge // 128))
    draw.rectangle(bounds, outline=(160, 20, 15, 255), width=max(2, output_long_edge // 256))

    label = f"CHANGE {patch.get('patch_id', '')}".strip()
    font = ImageFont.load_default(size=max(22, output_long_edge // 32))
    text_bounds = draw.textbbox((0, 0), label, font=font)
    badge_width = text_bounds[2] - text_bounds[0] + 24
    badge_height = text_bounds[3] - text_bounds[1] + 16
    badge_left = min(max(0, patch_left), max(0, output_width - badge_width))
    badge_top = max(0, patch_top - badge_height - 8)
    if badge_top == 0 and patch_top < badge_height + 8:
        badge_top = min(output_height - badge_height, patch_top + 8)
    draw.rounded_rectangle(
        (badge_left, badge_top, badge_left + badge_width, badge_top + badge_height),
        radius=6,
        fill=(160, 20, 15, 245),
        outline=(255, 255, 255, 255),
        width=2,
    )
    draw.text((badge_left + 12, badge_top + 8), label, fill=(255, 255, 255, 255), font=font)
    return np.asarray(PillowImage.alpha_composite(enlarged, marker).convert("RGB"))


@st.cache_data(ttl=21600, max_entries=48, show_spinner=False)
def _cached_sentinel_preview(
    latitude: float,
    longitude: float,
    target_date: str,
    search_window_days: int,
    radius_km: float,
    cache_version: int,
) -> tuple[np.ndarray, dict]:
    return fetch_sentinel_preview(
        latitude,
        longitude,
        date.fromisoformat(target_date),
        search_window_days,
        radius_km,
    )


@st.cache_data(ttl=21600, max_entries=48, show_spinner=False)
def _cached_sentinel1_rtc(
    latitude: float,
    longitude: float,
    target_date: str,
    search_window_days: int,
    radius_km: float,
    relative_orbit: int | None,
    cache_version: int,
) -> dict:
    return fetch_sentinel1_rtc(
        latitude,
        longitude,
        date.fromisoformat(target_date),
        search_window_days,
        radius_km,
        relative_orbit,
    )


def add_zone_report_download(
    corporation_name: str,
    city_name: str,
    area_name: str,
    latitude: float,
    longitude: float,
    t1_date: date,
    t2_date: date,
    scene_results: list[tuple[np.ndarray | None, dict | None, str | None]],
    zoom_factor: float,
    detection_summary: dict[str, int | float] | None = None,
    change_summary: dict[str, int | float] | None = None,
    candidate_patches: list[dict[str, object]] | None = None,
    anomaly_image_pairs: list[dict[str, object]] | None = None,
    analysis_note: str | None = None,
    download_label: str = "Download zone report PDF",
) -> None:
    t1_image, t1_metadata, _t1_error = scene_results[0]
    t2_image, t2_metadata, _t2_error = scene_results[1]
    if t1_image is not None and t2_image is not None and candidate_patches:
        anomaly_image_pairs = [
            {
                "patch": patch,
                "t1_image": enlarge_candidate_crop(t1_image, patch),
                "t2_image": enlarge_candidate_crop(t2_image, patch),
            }
            for patch in candidate_patches
        ]
    pdf_bytes = create_zone_report_pdf(
        corporation_name=corporation_name,
        city_name=city_name,
        area_name=area_name,
        latitude=latitude,
        longitude=longitude,
        t1_target=t1_date,
        t1_image=t1_image,
        t1_metadata=t1_metadata,
        t2_target=t2_date,
        t2_image=t2_image,
        t2_metadata=t2_metadata,
        detection_summary=detection_summary,
        change_summary=change_summary,
        candidate_patches=candidate_patches,
        anomaly_image_pairs=anomaly_image_pairs,
        analysis_note=analysis_note,
    )
    report_name = f"{_slug(area_name)}_T1-{t1_date.isoformat()}_T2-{t2_date.isoformat()}_report.pdf"
    st.download_button(
        download_label,
        data=pdf_bytes,
        file_name=report_name,
        mime="application/pdf",
        key=f"zone-report-{_slug(corporation_name)}-{_slug(area_name)}",
        use_container_width=True,
    )


st.title("Construction Watch")
st.caption("Sentinel-2 change monitoring by corporation, zone, and village")

with st.sidebar:
    st.subheader("Monitoring area")
    try:
        corporations = load_registry()
    except ValueError as error:
        st.error(str(error))
        st.stop()

    if not corporations:
        st.selectbox("Corporation", ["Add corporations to locations.json"], disabled=True)
        st.selectbox("City", ["Select a corporation first"], disabled=True)
        st.selectbox("Village", ["Select a city first"], disabled=True)
        st.info("Configure village image paths in data/locations.json to enable monitoring.")
        st.stop()

    corporation_names = [item["name"] for item in corporations]
    corporation_name = st.selectbox("Corporation", corporation_names)
    corporation = next(item for item in corporations if item["name"] == corporation_name)

    cities = corporation["cities"]
    if not cities:
        st.warning("No cities are configured for this corporation.")
        st.stop()
    if corporation.get("hide_city_selector", False):
        if len(cities) != 1:
            st.error("A hidden City selector requires exactly one configured city.")
            st.stop()
        city = cities[0]
        city_name = city["name"]
    else:
        city_names = [item["name"] for item in cities]
        city_name = st.selectbox("City", city_names)
        city = next(item for item in cities if item["name"] == city_name)

    zones = city.get("zones", [])
    if zones:
        zone_names = [item["name"] for item in zones]
        zone_name = st.selectbox("Zone", zone_names)
        area_group = next(item for item in zones if item["name"] == zone_name)
        selected_area_name = zone_name
    else:
        area_group = city
        selected_area_name = city_name
    coordinates = area_group.get("coordinates", city.get("coordinates"))

    village_names = [item["name"] for item in area_group["villages"]]
    if village_names:
        village_name = st.selectbox("Village", village_names)
        village = next(item for item in area_group["villages"] if item["name"] == village_name)
        threshold = st.slider(
            "Detection threshold",
            min_value=0.1,
            max_value=0.9,
            value=float(village.get("threshold", 0.5)),
            step=0.05,
            help="Raise the threshold to show only higher-confidence detections.",
        )
    else:
        st.selectbox("Village", ["No villages configured"], disabled=True)
        st.info("Add village names and their Sentinel-2 raster paths to locations.json to enable model detections.")
        village = None
        village_name = ""
        threshold = 0.5

    if coordinates:
        today = date.today()
        st.subheader("Analysis parameters")
        area_key = (
            corporation_name,
            city_name,
            selected_area_name,
            float(coordinates["latitude"]),
            float(coordinates["longitude"]),
        )
        with st.form("sentinel_image_selection"):
            t1_input = st.date_input("T1 - Past", value=today - timedelta(days=365), max_value=today)
            t2_input = st.date_input("T2 - Recent", value=today - timedelta(days=10), max_value=today)
            search_window_input = st.slider("Scene search window (+/- days)", 5, 30, 15)
            zoom_input = st.slider("Zoom both images", 1.0, 4.0, 1.0, 0.5)
            land_clearing_ndvi_input = st.slider(
                "Land-clearing NDVI drop",
                min_value=-0.50,
                max_value=-0.10,
                value=-0.25,
                step=0.05,
                help="Flag vegetation loss when NDVI change is at or below this value.",
            )
            built_surface_ndbi_input = st.slider(
                "Built-surface NDBI gain",
                min_value=0.05,
                max_value=0.40,
                value=0.15,
                step=0.05,
                help="Minimum NDBI increase required for a built-surface candidate.",
            )
            built_surface_ndvi_input = st.slider(
                "Built-surface NDVI change gate",
                min_value=-0.30,
                max_value=0.00,
                value=-0.10,
                step=0.05,
                help="Also require NDVI to decrease by at least this amount.",
            )
            enter_pressed = st.form_submit_button("Apply parameters and load", type="primary", use_container_width=True)
        if enter_pressed:
            st.session_state["imagery_request"] = {
                "area_key": area_key,
                "t1_date": t1_input.isoformat(),
                "t2_date": t2_input.isoformat(),
                "search_window_days": search_window_input,
                "zoom_factor": zoom_input,
                "land_clearing_ndvi_threshold": land_clearing_ndvi_input,
                "built_surface_ndbi_threshold": built_surface_ndbi_input,
                "built_surface_ndvi_threshold": built_surface_ndvi_input,
            }

if not coordinates:
    st.warning("Coordinates are not configured for this monitoring area.")
    st.stop()

imagery_request = st.session_state.get("imagery_request")
if not imagery_request or imagery_request["area_key"] != area_key:
    st.info("No imagery loaded.")
    add_zone_report_download(
        corporation_name,
        city_name,
        selected_area_name,
        float(coordinates["latitude"]),
        float(coordinates["longitude"]),
        t1_input,
        t2_input,
        [(None, None, "Imagery has not been loaded."), (None, None, "Imagery has not been loaded.")],
        zoom_input,
        analysis_note="Imagery has not been loaded. Apply the analysis parameters in the sidebar to generate a report with Sentinel-2 images.",
        download_label="Download preliminary PDF (no imagery)",
    )
    st.stop()

t1_date = date.fromisoformat(imagery_request["t1_date"])
t2_date = date.fromisoformat(imagery_request["t2_date"])
search_window_days = imagery_request["search_window_days"]
zoom_factor = imagery_request["zoom_factor"]
land_clearing_ndvi_threshold = float(imagery_request.get("land_clearing_ndvi_threshold", -0.25))
built_surface_ndbi_threshold = float(imagery_request.get("built_surface_ndbi_threshold", 0.15))
built_surface_ndvi_threshold = float(imagery_request.get("built_surface_ndvi_threshold", -0.10))

if t1_date >= t2_date:
    st.error("T1 must be earlier than T2. Choose an older date for T1 and a more recent date for T2.")
    st.stop()

latitude = float(coordinates["latitude"])
longitude = float(coordinates["longitude"])
st.markdown(f"### {selected_area_name}")
st.caption(
    f"{corporation_name} / {city_name} / {selected_area_name} · "
    f"{latitude:.4f}, {longitude:.4f} · {DEFAULT_PREVIEW_RADIUS_KM:g} km radius context"
)
st.link_button(
    "Open current Google satellite view",
    google_satellite_url(latitude, longitude),
    help="Opens Google Maps separately. Google imagery here is current context, not historical T1/T2 imagery.",
)

scene_results: list[tuple[np.ndarray | None, dict | None, str | None]] = []
with st.spinner("Searching Sentinel-2 L2A scenes and loading zone previews..."):
    for target_date in (t1_date, t2_date):
        try:
            image, metadata = _cached_sentinel_preview(
                latitude,
                longitude,
                target_date.isoformat(),
                search_window_days,
                DEFAULT_PREVIEW_RADIUS_KM,
                SENTINEL_ANALYSIS_CACHE_VERSION,
            )
            scene_results.append((image, metadata, None))
        except Exception as error:
            scene_results.append((None, None, str(error)))

t1_column, t2_column = st.columns(2)
for column, label, target_date, result in (
    (t1_column, "T1 - Past", t1_date, scene_results[0]),
    (t2_column, "T2 - Recent", t2_date, scene_results[1]),
):
    image, metadata, error_message = result
    with column:
        st.subheader(label)
        st.caption(f"Target date: {target_date.isoformat()}")
        if image is None or metadata is None:
            st.warning(f"Sentinel-2 preview unavailable: {error_message}")
        else:
            cloud_cover = metadata["cloud_cover"]
            cloud_label = "unknown" if cloud_cover is None else f"{cloud_cover:.1f}%"
            processing_level = metadata.get("processing_level", "L2A surface reflectance")
            st.caption(
                f"Sentinel-2 MSI true color (B04/B03/B02) · {processing_level} · 10 m ground sampling distance · "
                f"acquired {metadata['acquired']} · "
                f"cloud cover {cloud_label} · {metadata['scene_id']}"
            )
            st_folium(
                make_sentinel_preview_map(
                    zoom_preview(image, zoom_factor),
                    latitude,
                    longitude,
                    metadata.get("radius_km", DEFAULT_PREVIEW_RADIUS_KM),
                    zoom_factor,
                    overlay_name=label,
                    show_basemap=False,
                ),
                height=420,
                use_container_width=True,
                key=f"sentinel-{_slug(selected_area_name)}-{label.lower().replace(' ', '-')}",
            )

scene_metadata = [result[1] for result in scene_results]
if all(metadata is not None and "analysis_bands" in metadata for metadata in scene_metadata):
    try:
        spectral_change = analyze_l2a_change(
            scene_metadata[0],
            scene_metadata[1],
            land_clearing_ndvi_threshold=land_clearing_ndvi_threshold,
            built_surface_ndbi_threshold=built_surface_ndbi_threshold,
            built_surface_ndvi_threshold=built_surface_ndvi_threshold,
        )
    except (KeyError, ValueError) as error:
        spectral_change = None
        st.warning(f"Could not calculate spectral change candidates: {error}")
else:
    spectral_change = None

sar_change = None
sar_error = None
if spectral_change is not None:
    t1_metadata, t2_metadata = scene_metadata
    with st.spinner("Checking descending Sentinel-1 IW scenes on a matched orbit..."):
        try:
            t1_sar = _cached_sentinel1_rtc(
                latitude,
                longitude,
                t1_date.isoformat(),
                search_window_days,
                DEFAULT_PREVIEW_RADIUS_KM,
                None,
                SENTINEL_ANALYSIS_CACHE_VERSION,
            )
            t2_sar = _cached_sentinel1_rtc(
                latitude,
                longitude,
                t2_date.isoformat(),
                search_window_days,
                DEFAULT_PREVIEW_RADIUS_KM,
                t1_sar["relative_orbit"],
                SENTINEL_ANALYSIS_CACHE_VERSION,
            )
            sar_change = analyze_sar_vertical_candidates(
                t1_sar,
                t2_sar,
                spectral_change["built_surface_gain_mask"],
                vv_gain_threshold_db=4.0,
            )
        except Exception as error:
            sar_error = str(error)

change_summary = None
if spectral_change is not None:
    chip_height, chip_width = spectral_change["valid_mask"].shape
    pixel_area_hectares = (
        (2 * DEFAULT_PREVIEW_RADIUS_KM * 1000 / chip_height)
        * (2 * DEFAULT_PREVIEW_RADIUS_KM * 1000 / chip_width)
        / 10_000
    )
    change_summary = {
        "valid_area_hectares": float(spectral_change["valid_pixels"]) * pixel_area_hectares,
        "land_clearing_area_hectares": float(spectral_change["land_clearing_pixels"]) * pixel_area_hectares,
        "built_surface_gain_area_hectares": float(spectral_change["built_surface_gain_pixels"]) * pixel_area_hectares,
        "mean_ndvi_delta": float(spectral_change["mean_ndvi_delta"]),
        "mean_ndbi_delta": float(spectral_change["mean_ndbi_delta"]),
        "land_clearing_ndvi_threshold": land_clearing_ndvi_threshold,
        "built_surface_ndbi_threshold": built_surface_ndbi_threshold,
        "built_surface_ndvi_threshold": built_surface_ndvi_threshold,
    }
    if sar_change is not None:
        change_summary.update(
            {
                "sar_structure_area_hectares": float(sar_change["candidate_pixels"]) * pixel_area_hectares,
                "sar_mean_vv_change_db": float(sar_change["mean_vv_change_db"]),
                "sar_relative_orbit": int(sar_change["relative_orbit"]),
                "sar_vv_gain_threshold_db": float(sar_change["vv_gain_threshold_db"]),
                "sar_t1_acquired": sar_change["t1_acquired"],
                "sar_t2_acquired": sar_change["t2_acquired"],
            }
        )

if spectral_change is not None:
    st.subheader("Change candidates")
    st.caption(
        f"Built-surface gain means NDBI rose by at least {built_surface_ndbi_threshold:.2f} "
        f"while NDVI fell by at least {abs(built_surface_ndvi_threshold):.2f} from T1 to T2. "
        "It is a surface-change signal, not confirmation of a new building."
    )
    vertical_structure_mask = (
        sar_change["vertical_structure_mask"]
        if sar_change is not None
        else np.zeros_like(spectral_change["built_surface_gain_mask"])
    )
    comparison_layers = ["Land-clearing candidates", "Built-surface gain candidates"]
    if sar_change is not None:
        comparison_layers.append("SAR-backed vertical-structure candidates")
    comparison_layers.append("Combined candidates")
    with st.sidebar:
        change_layer = st.selectbox("Comparison layer", comparison_layers)
    land_clearing_mask = spectral_change["land_clearing_mask"]
    built_surface_mask = spectral_change["built_surface_gain_mask"]
    all_candidate_patches = summarize_candidate_patches(
        land_clearing_mask | built_surface_mask | vertical_structure_mask,
        latitude,
        longitude,
        DEFAULT_PREVIEW_RADIUS_KM,
        land_clearing_mask,
        built_surface_mask,
        min_pixels=2,
        max_patches=None,
        vertical_structure_mask=vertical_structure_mask,
    )
    candidate_overlay = np.zeros((*land_clearing_mask.shape, 4), dtype=np.uint8)
    if change_layer == "Land-clearing candidates":
        active_candidates = land_clearing_mask
        selected_clearing_mask = land_clearing_mask
        selected_built_mask = np.zeros_like(built_surface_mask)
        selected_vertical_mask = np.zeros_like(vertical_structure_mask)
        candidate_overlay[active_candidates] = (239, 171, 67, 220)
        legend = "Amber: vegetation-loss spectral candidates"
    elif change_layer == "Built-surface gain candidates":
        active_candidates = built_surface_mask
        selected_clearing_mask = np.zeros_like(land_clearing_mask)
        selected_built_mask = built_surface_mask
        selected_vertical_mask = np.zeros_like(vertical_structure_mask)
        candidate_overlay[active_candidates] = (220, 76, 58, 220)
        legend = "Red: new built/impervious-surface spectral candidates"
    elif change_layer == "SAR-backed vertical-structure candidates":
        active_candidates = vertical_structure_mask
        selected_clearing_mask = np.zeros_like(land_clearing_mask)
        selected_built_mask = vertical_structure_mask
        selected_vertical_mask = vertical_structure_mask
        candidate_overlay[active_candidates] = (25, 126, 105, 230)
        legend = "Teal: optical built-surface gain with matched-orbit SAR VV increase >= 4 dB"
    else:
        active_candidates = land_clearing_mask | built_surface_mask | vertical_structure_mask
        selected_clearing_mask = land_clearing_mask
        selected_built_mask = built_surface_mask
        selected_vertical_mask = vertical_structure_mask
        candidate_overlay[land_clearing_mask] = (239, 171, 67, 220)
        candidate_overlay[built_surface_mask] = (220, 76, 58, 220)
        candidate_overlay[vertical_structure_mask] = (25, 126, 105, 230)
        legend = "Amber: vegetation loss; red: optical built gain; teal: SAR-backed structure candidates"

    candidate_area_hectares = float(active_candidates.sum()) * pixel_area_hectares
    valid_count = int(spectral_change["valid_pixels"])
    valid_area_hectares = float(valid_count) * pixel_area_hectares
    candidate_share = candidate_area_hectares / valid_area_hectares * 100 if valid_area_hectares else 0.0
    candidate_patches = summarize_candidate_patches(
        active_candidates,
        latitude,
        longitude,
        DEFAULT_PREVIEW_RADIUS_KM,
        selected_clearing_mask,
        selected_built_mask,
        min_pixels=2,
        max_patches=None,
        vertical_structure_mask=selected_vertical_mask,
    )
    metric_columns = st.columns(3)
    metric_columns[0].metric("Anomaly locations", f"{len(candidate_patches):,}")
    metric_columns[1].metric("Approx. candidate area", f"{candidate_area_hectares:.2f} ha")
    metric_columns[2].metric("Clear area analyzed", f"{valid_area_hectares:.2f} ha")
    st_folium(
        make_sentinel_preview_map(
            candidate_overlay,
            latitude,
            longitude,
            DEFAULT_PREVIEW_RADIUS_KM,
            1.0,
            overlay_name=change_layer,
            candidate_patches=candidate_patches[:10],
        ),
        height=420,
        use_container_width=True,
        key=f"change-{_slug(selected_area_name)}-{_slug(change_layer)}",
    )
    st.caption(
        f"{legend}. {candidate_share:.1f}% of clear land area flagged. "
        f"Mean ΔNDVI {spectral_change['mean_ndvi_delta']:+.3f}; "
        f"mean ΔNDBI {spectral_change['mean_ndbi_delta']:+.3f}. "
        "Isolated one-cell speckles are excluded. "
        "These are screening proxies, not building-height measurements or a legal determination."
    )
    if sar_change is not None:
        st.caption(
            f"Sentinel-1 IW GRD-RTC secondary check: descending relative orbit "
            f"#{sar_change['relative_orbit']} ({sar_change['t1_acquired']} / {sar_change['t2_acquired']}), "
            f"VV increase threshold {sar_change['vv_gain_threshold_db']:.1f} dB. "
            "This is a backscatter-change proxy, not a height measurement."
        )
    elif sar_error:
        st.warning(f"Sentinel-1 secondary verification unavailable: {sar_error}")
    if candidate_patches:
        st.markdown("#### Candidate locations")
        st.dataframe(
            [
                {
                    "Patch": patch["patch_id"],
                    "Signal": patch["signal"],
                    "Latitude": round(float(patch["latitude"]), 6),
                    "Longitude": round(float(patch["longitude"]), 6),
                    "Approx. area (ha)": round(float(patch["area_hectares"]), 3),
                    "Google satellite": google_satellite_url(
                        float(patch["latitude"]), float(patch["longitude"])
                    ),
                }
                for patch in candidate_patches
            ],
            column_config={
                "Google satellite": st.column_config.LinkColumn(
                    "Google satellite",
                    display_text="Open current view",
                )
            },
            hide_index=True,
            use_container_width=True,
        )
        t1_rgb, t2_rgb = scene_results[0][0], scene_results[1][0]
        if t1_rgb is not None and t2_rgb is not None:
            st.markdown("#### Before / after appearance")
            for patch in candidate_patches[:10]:
                with st.expander(
                    f"{patch['patch_id']} · {patch['signal']} · "
                    f"{float(patch['area_hectares']):.3f} ha · "
                    f"{float(patch['latitude']):.5f}, {float(patch['longitude']):.5f}"
                ):
                    t1_view, t2_view = st.columns(2)
                    t1_view.image(enlarge_candidate_crop(t1_rgb, patch), caption="T1 - Past")
                    t2_view.image(enlarge_candidate_crop(t2_rgb, patch), caption="T2 - Recent")
                    st.caption("Red outline marks the candidate within its surrounding area. Crops are nearest-neighbor enlarged; Sentinel-2's 10 m ground sampling distance cannot resolve building height or roof details.")
            if len(candidate_patches) > 10:
                st.caption(f"T1/T2 crops shown for 10 largest patches; all {len(candidate_patches)} qualifying patches are listed in the PDF report.")
    else:
        st.info("No candidate areas met the minimum mapping unit for this comparison layer.")

if village is None:
    st.info("Sentinel-2 T1/T2 imagery and spectral change candidates are shown above. Add village-level training data and a trained model for model-based construction detections.")
    add_zone_report_download(
        corporation_name,
        city_name,
        selected_area_name,
        latitude,
        longitude,
        t1_date,
        t2_date,
        scene_results,
        zoom_factor,
        change_summary=change_summary,
        candidate_patches=all_candidate_patches if spectral_change is not None else None,
        analysis_note=(
            "No trained village model is configured. The spectral candidate indicators above are exploratory "
            "screening signals and do not determine whether construction is illegal."
        ),
    )
    st.stop()

before_path = resolve_path(village.get("before", ""))
after_path = resolve_path(village.get("after", ""))
model_path = resolve_path(village.get("model", "models/construction_rf.joblib"))
missing_files = [path for path in (before_path, after_path, model_path) if not path.is_file()]
if missing_files:
    st.warning("The selected village is missing required data files:")
    for missing_path in missing_files:
        st.code(str(missing_path))
    add_zone_report_download(
        corporation_name,
        city_name,
        selected_area_name,
        latitude,
        longitude,
        t1_date,
        t2_date,
        scene_results,
        zoom_factor,
        change_summary=change_summary,
        candidate_patches=all_candidate_patches if spectral_change is not None else None,
        analysis_note="No model-based assessment: required village imagery or the trained model file is missing. Spectral candidates are not legal findings.",
    )
    st.stop()

map_location = f"{city_name}_{selected_area_name}"
mask_path, probability_path = prediction_paths(
    corporation_name, map_location, village_name, before_path, after_path, model_path, threshold
)
with st.spinner(f"Loading the detection map for {village_name}..."):
    try:
        ensure_prediction(before_path, after_path, model_path, mask_path, probability_path, threshold)
        overlay, bounds, center, detected_count, valid_count = read_detection_overlay(mask_path)
    except Exception as error:
        st.error(f"Could not create the village detection map: {error}")
        add_zone_report_download(
            corporation_name,
            city_name,
            selected_area_name,
            latitude,
            longitude,
            t1_date,
            t2_date,
            scene_results,
            zoom_factor,
            change_summary=change_summary,
            candidate_patches=all_candidate_patches if spectral_change is not None else None,
            analysis_note=f"No model-based assessment because the configured detection run failed: {error}. Spectral candidates are not legal findings.",
        )
        st.stop()

st.markdown(f"### Detections: {village_name}")
detection_share = detected_count / valid_count * 100 if valid_count else 0.0
st.metric("Share of valid area flagged", f"{detection_share:.2f}%")
st.markdown("**Map key:** <span style='color:#dc4c3a'>■</span> candidate construction areas", unsafe_allow_html=True)
map_key = "_".join(_slug(part) for part in (corporation_name, city_name, selected_area_name, village_name))
st_folium(make_map(overlay, bounds, center, map_key), height=680, use_container_width=True, key=map_key)
st.caption(f"Detection mask: {mask_path}")
add_zone_report_download(
    corporation_name,
    city_name,
    selected_area_name,
    latitude,
    longitude,
    t1_date,
    t2_date,
    scene_results,
    zoom_factor,
    detection_summary={
        "detected_pixels": detected_count,
        "valid_pixels": valid_count,
        "detected_share_percent": detection_share,
    },
    change_summary=change_summary,
    candidate_patches=all_candidate_patches if spectral_change is not None else None,
)
