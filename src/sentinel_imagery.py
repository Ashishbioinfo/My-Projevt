"""Fetch small Sentinel-2 L2A RGB previews from Microsoft Planetary Computer."""

from __future__ import annotations

from collections import deque
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import folium
import planetary_computer
import pystac_client
import rasterio
from branca.element import MacroElement
from folium.plugins import Fullscreen
from jinja2 import Template
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform as transform_coordinates
from scipy.ndimage import label as connected_components

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SENTINEL2_L2A_COLLECTION = "sentinel-2-l2a"
SENTINEL1_RTC_COLLECTION = "sentinel-1-rtc"
DEFAULT_PREVIEW_RADIUS_KM = 1.0
PROJ_DATA = Path(rasterio.__file__).resolve().parent / "proj_data"


def add_metric_scale_control(map_view: folium.Map) -> None:
    scale_control = MacroElement()
    scale_control._template = Template(
        """
        {% macro script(this, kwargs) %}
        L.control.scale({position: 'bottomleft', maxWidth: 200, metric: true, imperial: false})
            .addTo({{ this._parent.get_name() }});
        {% endmacro %}
        """
    )
    map_view.add_child(scale_control)


def _read_window_bands(
    href: str,
    latitude: float,
    longitude: float,
    radius_km: float,
    indexes: list[int],
    output_shape: tuple[int, int],
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    env_options = {"PROJ_LIB": str(PROJ_DATA)} if PROJ_DATA.is_dir() else {}
    with rasterio.Env(**env_options):
        with rasterio.open(href) as dataset:
            if dataset.crs is None:
                raise ValueError("Sentinel-2 image has no CRS.")
            x_coords, y_coords = transform_coordinates(
                "EPSG:4326", dataset.crs, [longitude], [latitude]
            )
            center_x, center_y = x_coords[0], y_coords[0]
            radius_m = radius_km * 1000
            window = from_bounds(
                center_x - radius_m,
                center_y - radius_m,
                center_x + radius_m,
                center_y + radius_m,
                dataset.transform,
            )
            try:
                window = window.intersection(Window(0, 0, dataset.width, dataset.height))
            except rasterio.errors.WindowError as error:
                raise ValueError("The selected Sentinel-2 tile does not cover this zone.") from error
            window = window.round_offsets().round_lengths()
            if window.width < 1 or window.height < 1:
                raise ValueError("The selected Sentinel-2 tile does not cover this zone.")

            return dataset.read(
                indexes=indexes,
                window=window,
                out_shape=(len(indexes), *output_shape),
                resampling=resampling,
            )


def _stretch_rgb(bands: np.ndarray) -> np.ndarray:
    rgb = np.zeros((bands.shape[1], bands.shape[2], 3), dtype=np.uint8)
    if bands.dtype == np.uint8:
        return np.moveaxis(bands[:3], 0, -1)

    for channel_index, band in enumerate(bands[:3]):
        valid = np.isfinite(band) & (band > 0)
        if not valid.any():
            continue
        low, high = np.percentile(band[valid], (2, 98))
        if high <= low:
            high = low + 1
        scaled = np.clip((band.astype(np.float32) - low) / (high - low), 0, 1)
        rgb[:, :, channel_index] = (scaled * 255).astype(np.uint8)
    return rgb


def analyze_l2a_change(
    before_metadata: dict[str, Any],
    after_metadata: dict[str, Any],
    *,
    land_clearing_ndvi_threshold: float = -0.25,
    built_surface_ndbi_threshold: float = 0.15,
    built_surface_ndvi_threshold: float = -0.10,
) -> dict[str, Any]:
    """Return cloud-masked spectral-change candidates, not building/legal classifications."""
    before = np.asarray(before_metadata["analysis_bands"], dtype=np.float32)
    after = np.asarray(after_metadata["analysis_bands"], dtype=np.float32)
    before_clear = np.asarray(before_metadata["clear_land"], dtype=bool)
    after_clear = np.asarray(after_metadata["clear_land"], dtype=bool)
    if before.shape != after.shape or before.ndim != 3 or before.shape[0] != 6:
        raise ValueError("Before and after L2A data must align as (6, height, width).")
    if before_clear.shape != after_clear.shape or before_clear.shape != before.shape[1:]:
        raise ValueError("Before and after SCL clear-land masks must align with the L2A bands.")

    valid = (
        before_clear
        & after_clear
        & np.isfinite(before).all(axis=0)
        & np.isfinite(after).all(axis=0)
    )

    def normalized_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        denominator = first + second
        return np.divide(
            first - second,
            denominator,
            out=np.zeros_like(first, dtype=np.float32),
            where=np.abs(denominator) > 1e-6,
        )

    before_ndvi = normalized_difference(before[3], before[2])
    after_ndvi = normalized_difference(after[3], after[2])
    before_ndbi = normalized_difference(before[4], before[3])
    after_ndbi = normalized_difference(after[4], after[3])
    ndvi_delta = after_ndvi - before_ndvi
    ndbi_delta = after_ndbi - before_ndbi

    land_clearing = valid & (ndvi_delta <= land_clearing_ndvi_threshold)
    built_surface_gain = (
        valid
        & (ndbi_delta >= built_surface_ndbi_threshold)
        & (ndvi_delta <= built_surface_ndvi_threshold)
    )
    raw_candidates = land_clearing | built_surface_gain
    component_labels, _component_count = connected_components(
        raw_candidates,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    component_sizes = np.bincount(component_labels.ravel())
    keep_component = component_sizes >= 2
    keep_component[0] = False
    keep_pixels = keep_component[component_labels]
    single_pixel_noise_removed = int(raw_candidates.sum() - keep_pixels.sum())
    land_clearing &= keep_pixels
    built_surface_gain &= keep_pixels
    valid_count = int(valid.sum())
    return {
        "valid_mask": valid,
        "land_clearing_mask": land_clearing,
        "built_surface_gain_mask": built_surface_gain,
        "valid_pixels": valid_count,
        "land_clearing_pixels": int(land_clearing.sum()),
        "built_surface_gain_pixels": int(built_surface_gain.sum()),
        "single_pixel_noise_removed": single_pixel_noise_removed,
        "mean_ndvi_delta": float(np.mean(ndvi_delta[valid])) if valid_count else 0.0,
        "mean_ndbi_delta": float(np.mean(ndbi_delta[valid])) if valid_count else 0.0,
        "thresholds": {
            "land_clearing_ndvi": land_clearing_ndvi_threshold,
            "built_surface_ndbi": built_surface_ndbi_threshold,
            "built_surface_ndvi": built_surface_ndvi_threshold,
        },
    }


def fetch_sentinel1_rtc(
    latitude: float,
    longitude: float,
    target_date: date,
    search_window_days: int = 15,
    radius_km: float = DEFAULT_PREVIEW_RADIUS_KM,
    relative_orbit: int | None = None,
) -> dict[str, Any]:
    """Fetch descending Sentinel-1 IW RTC VV/VH linear backscatter for a point chip."""
    latitude_delta = radius_km / 111.0
    longitude_delta = radius_km / (111.0 * np.cos(np.deg2rad(latitude)))
    bbox = [
        longitude - longitude_delta,
        latitude - latitude_delta,
        longitude + longitude_delta,
        latitude + latitude_delta,
    ]
    date_start = target_date - timedelta(days=search_window_days)
    date_end = min(target_date + timedelta(days=search_window_days), date.today())
    catalog = pystac_client.Client.open(
        STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=[SENTINEL1_RTC_COLLECTION],
        bbox=bbox,
        datetime=f"{date_start.isoformat()}/{date_end.isoformat()}",
        max_items=100,
    )

    def is_matching_scene(candidate: Any) -> bool:
        bounds = candidate.bbox
        return bool(
            candidate.properties.get("sar:instrument_mode") == "IW"
            and candidate.properties.get("sat:orbit_state") == "descending"
            and (relative_orbit is None or candidate.properties.get("sat:relative_orbit") == relative_orbit)
            and bounds
            and bounds[0] <= longitude <= bounds[2]
            and bounds[1] <= latitude <= bounds[3]
            and "vv" in candidate.assets
            and "vh" in candidate.assets
        )

    items = [candidate for candidate in search.items() if is_matching_scene(candidate)]
    if not items:
        orbit_note = "" if relative_orbit is None else f" on descending relative orbit {relative_orbit}"
        raise ValueError(
            f"No Sentinel-1 RTC IW scene covering these coordinates{orbit_note} was found "
            f"within {search_window_days} days of {target_date.isoformat()}."
        )
    item = min(
        items,
        key=lambda candidate: abs((candidate.datetime.date() - target_date).days),
    )
    chip_size = max(64, min(256, round(200 * radius_km)))
    output_shape = (chip_size, chip_size)
    vv = _read_window_bands(
        item.assets["vv"].href,
        latitude,
        longitude,
        radius_km,
        [1],
        output_shape,
    )[0].astype(np.float32)
    vh = _read_window_bands(
        item.assets["vh"].href,
        latitude,
        longitude,
        radius_km,
        [1],
        output_shape,
    )[0].astype(np.float32)
    valid = np.isfinite(vv) & np.isfinite(vh) & (vv > 0) & (vh > 0)
    return {
        "vv": vv,
        "vh": vh,
        "valid_mask": valid,
        "scene_id": item.id,
        "acquired": item.datetime.date().isoformat(),
        "orbit_direction": item.properties.get("sat:orbit_state"),
        "relative_orbit": int(item.properties["sat:relative_orbit"]),
        "instrument_mode": item.properties.get("sar:instrument_mode"),
        "polarizations": item.properties.get("sar:polarizations", []),
        "collection": SENTINEL1_RTC_COLLECTION,
        "backscatter_units": "linear sigma0; relative change reported in dB",
    }


def analyze_sar_vertical_candidates(
    before_sar: dict[str, Any],
    after_sar: dict[str, Any],
    optical_built_surface_mask: np.ndarray,
    *,
    vv_gain_threshold_db: float = 4.0,
) -> dict[str, Any]:
    """Use matched descending-IW VV gain to screen optical built-surface candidates."""
    if before_sar["relative_orbit"] != after_sar["relative_orbit"]:
        raise ValueError("Sentinel-1 T1/T2 scenes must use the same relative orbit.")
    if before_sar["orbit_direction"] != "descending" or after_sar["orbit_direction"] != "descending":
        raise ValueError("Sentinel-1 T1/T2 scenes must both be descending passes.")
    if before_sar["instrument_mode"] != "IW" or after_sar["instrument_mode"] != "IW":
        raise ValueError("Sentinel-1 T1/T2 scenes must both use IW mode.")

    before_vv = np.asarray(before_sar["vv"], dtype=np.float32)
    after_vv = np.asarray(after_sar["vv"], dtype=np.float32)
    optical_mask = np.asarray(optical_built_surface_mask, dtype=bool)
    valid = (
        np.asarray(before_sar["valid_mask"], dtype=bool)
        & np.asarray(after_sar["valid_mask"], dtype=bool)
        & (before_vv > 0)
        & (after_vv > 0)
    )
    if before_vv.shape != after_vv.shape or optical_mask.shape != before_vv.shape or valid.shape != before_vv.shape:
        raise ValueError("Sentinel-1 and optical candidate chips must use matching dimensions.")

    vv_change_db = np.zeros(before_vv.shape, dtype=np.float32)
    vv_change_db[valid] = 10 * np.log10(after_vv[valid] / before_vv[valid])
    sar_confirmed_candidate = valid & optical_mask & (vv_change_db >= vv_gain_threshold_db)
    valid_count = int(valid.sum())
    return {
        "vv_change_db": vv_change_db,
        "valid_mask": valid,
        "vertical_structure_mask": sar_confirmed_candidate,
        "candidate_pixels": int(sar_confirmed_candidate.sum()),
        "mean_vv_change_db": float(vv_change_db[valid].mean()) if valid_count else 0.0,
        "valid_pixels": valid_count,
        "vv_gain_threshold_db": vv_gain_threshold_db,
        "relative_orbit": before_sar["relative_orbit"],
        "t1_acquired": before_sar["acquired"],
        "t2_acquired": after_sar["acquired"],
    }


def summarize_candidate_patches(
    candidate_mask: np.ndarray,
    latitude: float,
    longitude: float,
    radius_km: float,
    land_clearing_mask: np.ndarray,
    built_surface_gain_mask: np.ndarray,
    *,
    min_pixels: int = 2,
    max_patches: int | None = None,
    vertical_structure_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Group adjacent candidate pixels and estimate each patch's center and area."""
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    land_clearing_mask = np.asarray(land_clearing_mask, dtype=bool)
    built_surface_gain_mask = np.asarray(built_surface_gain_mask, dtype=bool)
    if vertical_structure_mask is None:
        vertical_structure_mask = np.zeros_like(candidate_mask)
    else:
        vertical_structure_mask = np.asarray(vertical_structure_mask, dtype=bool)
    if candidate_mask.ndim != 2 or land_clearing_mask.shape != candidate_mask.shape:
        raise ValueError("Candidate masks must be matching two-dimensional arrays.")
    if built_surface_gain_mask.shape != candidate_mask.shape:
        raise ValueError("Built-surface mask must match the candidate mask.")
    if vertical_structure_mask.shape != candidate_mask.shape:
        raise ValueError("Vertical-structure mask must match the candidate mask.")

    height, width = candidate_mask.shape
    visited = np.zeros(candidate_mask.shape, dtype=bool)
    pixel_area_hectares = (2 * radius_km * 1000 / height) * (2 * radius_km * 1000 / width) / 10_000
    latitude_delta = radius_km / 111.0
    longitude_delta = radius_km / (111.0 * np.cos(np.deg2rad(latitude)))
    degree_height = 2 * latitude_delta / height
    degree_width = 2 * longitude_delta / width
    patches: list[dict[str, Any]] = []

    for start_row, start_column in np.argwhere(candidate_mask):
        if visited[start_row, start_column]:
            continue
        queue = deque([(int(start_row), int(start_column))])
        visited[start_row, start_column] = True
        pixels: list[tuple[int, int]] = []
        while queue:
            pixel_row, pixel_column = queue.popleft()
            pixels.append((pixel_row, pixel_column))
            for row_offset in (-1, 0, 1):
                for column_offset in (-1, 0, 1):
                    if row_offset == 0 and column_offset == 0:
                        continue
                    neighbor_row = pixel_row + row_offset
                    neighbor_column = pixel_column + column_offset
                    if (
                        0 <= neighbor_row < height
                        and 0 <= neighbor_column < width
                        and candidate_mask[neighbor_row, neighbor_column]
                        and not visited[neighbor_row, neighbor_column]
                    ):
                        visited[neighbor_row, neighbor_column] = True
                        queue.append((neighbor_row, neighbor_column))

        if len(pixels) < min_pixels:
            continue
        rows = np.fromiter((pixel[0] for pixel in pixels), dtype=np.float32)
        columns = np.fromiter((pixel[1] for pixel in pixels), dtype=np.float32)
        clearing_count = sum(bool(land_clearing_mask[row, column]) for row, column in pixels)
        built_count = sum(bool(built_surface_gain_mask[row, column]) for row, column in pixels)
        vertical_count = sum(bool(vertical_structure_mask[row, column]) for row, column in pixels)
        if clearing_count and vertical_count:
            signal = "Land clearing + SAR-backed structure candidate"
        elif vertical_count:
            signal = "SAR-backed structure candidate"
        elif clearing_count and built_count:
            signal = "Land clearing + built-surface gain"
        elif built_count:
            signal = "Built-surface gain"
        else:
            signal = "Land clearing"

        patches.append(
            {
                "patch_id": "",
                "signal": signal,
                "pixels": len(pixels),
                "area_hectares": len(pixels) * pixel_area_hectares,
                "latitude": latitude + latitude_delta - (float(rows.mean()) + 0.5) * degree_height,
                "longitude": longitude - longitude_delta + (float(columns.mean()) + 0.5) * degree_width,
                "row_start": min(pixel[0] for pixel in pixels),
                "row_end": max(pixel[0] for pixel in pixels) + 1,
                "column_start": min(pixel[1] for pixel in pixels),
                "column_end": max(pixel[1] for pixel in pixels) + 1,
            }
        )

    patches.sort(key=lambda patch: patch["area_hectares"], reverse=True)
    selected_patches = patches if max_patches is None else patches[:max_patches]
    for patch_number, patch in enumerate(selected_patches, start=1):
        patch["patch_id"] = f"P{patch_number}"
    return selected_patches


def make_sentinel_preview_map(
    image: np.ndarray,
    latitude: float,
    longitude: float,
    radius_km: float = DEFAULT_PREVIEW_RADIUS_KM,
    zoom_factor: float = 1.0,
    overlay_name: str = "Sentinel-2 L2A preview",
    candidate_patches: list[dict[str, Any]] | None = None,
    show_basemap: bool = True,
) -> folium.Map:
    """Display a georeferenced RGB chip with a metric 200 px Leaflet scale bar."""
    view_radius_km = radius_km / max(1.0, zoom_factor)
    latitude_delta = view_radius_km / 111.0
    longitude_delta = view_radius_km / (111.0 * np.cos(np.deg2rad(latitude)))
    bounds = [
        [latitude - latitude_delta, longitude - longitude_delta],
        [latitude + latitude_delta, longitude + longitude_delta],
    ]
    map_view = folium.Map(
        location=(latitude, longitude),
        zoom_start=16,
        tiles="OpenStreetMap" if show_basemap else None,
        control_scale=False,
    )
    map_view.options["minZoom"] = 13
    map_view.options["maxZoom"] = 17
    folium.raster_layers.ImageOverlay(
        image=image,
        bounds=bounds,
        name=overlay_name,
        opacity=1.0,
        interactive=False,
        cross_origin=False,
        zindex=2,
    ).add_to(map_view)
    folium.CircleMarker(
        location=(latitude, longitude),
        radius=4,
        color="#dc4c3a",
        fill=True,
        fill_opacity=1,
        tooltip=f"Selected coordinates: {latitude:.5f}, {longitude:.5f}",
    ).add_to(map_view)
    for patch in candidate_patches or []:
        folium.CircleMarker(
            location=(patch["latitude"], patch["longitude"]),
            radius=7,
            color="#a83224",
            fill=True,
            fill_color="#dc4c3a",
            fill_opacity=0.9,
            tooltip=(
                f"{patch['patch_id']} · {patch['signal']} · "
                f"{patch['area_hectares']:.2f} ha · "
                f"{patch['latitude']:.5f}, {patch['longitude']:.5f}"
            ),
        ).add_to(map_view)
    add_metric_scale_control(map_view)
    Fullscreen(position="topright").add_to(map_view)
    map_view.fit_bounds(bounds, padding=(16, 16), max_zoom=16)
    return map_view


def fetch_sentinel_preview(
    latitude: float,
    longitude: float,
    target_date: date,
    search_window_days: int = 15,
    radius_km: float = DEFAULT_PREVIEW_RADIUS_KM,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fetch the closest low-cloud Sentinel-2 scene and return a local RGB chip."""
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * np.cos(np.deg2rad(latitude)))
    bbox = [
        longitude - lon_delta,
        latitude - lat_delta,
        longitude + lon_delta,
        latitude + lat_delta,
    ]
    date_start = target_date - timedelta(days=search_window_days)
    date_end = min(target_date + timedelta(days=search_window_days), date.today())
    catalog = pystac_client.Client.open(
        STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=[SENTINEL2_L2A_COLLECTION],
        bbox=bbox,
        datetime=f"{date_start.isoformat()}/{date_end.isoformat()}",
        query={"eo:cloud_cover": {"lt": 80}},
        max_items=60,
    )
    items = list(search.items())
    if not items:
        search = catalog.search(
            collections=[SENTINEL2_L2A_COLLECTION],
            bbox=bbox,
            datetime=f"{date_start.isoformat()}/{date_end.isoformat()}",
            max_items=60,
        )
        items = list(search.items())

    def covers_selected_coordinate(candidate: Any) -> bool:
        bounds = candidate.bbox
        return bool(
            bounds
            and len(bounds) >= 4
            and bounds[0] <= longitude <= bounds[2]
            and bounds[1] <= latitude <= bounds[3]
        )

    items = [candidate for candidate in items if covers_selected_coordinate(candidate)]
    if not items:
        search = catalog.search(
            collections=[SENTINEL2_L2A_COLLECTION],
            bbox=bbox,
            datetime=f"{date_start.isoformat()}/{date_end.isoformat()}",
            max_items=60,
        )
        items = [candidate for candidate in search.items() if covers_selected_coordinate(candidate)]
    if not items:
        raise ValueError(
            f"No Sentinel-2 L2A tile covering these coordinates was found within "
            f"{search_window_days} days of {target_date.isoformat()}."
        )

    item = min(
        items,
        key=lambda candidate: (
            abs((candidate.datetime.date() - target_date).days),
            candidate.properties.get("eo:cloud_cover", 100),
        ),
    )
    band_keys = ("B02", "B03", "B04", "B08", "B11", "B12")
    missing = [key for key in (*band_keys, "SCL") if key not in item.assets]
    if missing:
        raise ValueError(f"Sentinel-2 L2A scene is missing required analysis assets: {', '.join(missing)}")

    pixels_per_side = max(64, min(256, round(200 * radius_km)))
    output_shape = (pixels_per_side, pixels_per_side)
    analysis_bands = np.concatenate(
        [
            _read_window_bands(
                item.assets[key].href,
                latitude,
                longitude,
                radius_km,
                [1],
                output_shape,
            )
            for key in band_keys
        ],
        axis=0,
    )
    scene_classification = _read_window_bands(
        item.assets["SCL"].href,
        latitude,
        longitude,
        radius_km,
        [1],
        output_shape,
        Resampling.nearest,
    )[0]
    clear_land = np.isin(scene_classification, (4, 5))
    rgb = _stretch_rgb(analysis_bands[[2, 1, 0]])

    cloud_cover = item.properties.get("eo:cloud_cover")
    return rgb, {
        "scene_id": item.id,
        "acquired": item.datetime.date().isoformat(),
        "cloud_cover": None if cloud_cover is None else float(cloud_cover),
        "target_date": target_date.isoformat(),
        "collection": SENTINEL2_L2A_COLLECTION,
        "processing_level": "L2A surface reflectance",
        "radius_km": radius_km,
        "analysis_bands": analysis_bands.astype(np.float32),
        "clear_land": clear_land,
    }
