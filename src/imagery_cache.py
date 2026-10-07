"""Persistent on-disk cache for fetched Sentinel imagery and metadata.

Streamlit's ``st.cache_data`` already avoids repeat network calls within a
single running process, but that cache is lost on restart and expires after
its TTL. This module adds a small disk-backed cache so that once a scene has
been fetched for a given zone/date, it is reused on later app runs instead of
re-querying the Planetary Computer STAC API.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "imagery_cache"


def _cache_key(name: str, fields: dict[str, Any]) -> str:
    """Build a stable filename-safe key from the request parameters."""
    serialized = json.dumps(fields, sort_keys=True, default=str)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]
    return f"{name}_{digest}"


def cached_fetch(
    fields: dict[str, Any],
    loader: Callable[[], T],
    *,
    name: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> T:
    """Return a cached result for ``fields`` if present, otherwise call ``loader`` and store it.

    ``fields`` should uniquely identify the request (coordinates, target date,
    search window, radius, and a cache-format version) so that unrelated
    requests never collide and format changes can invalidate old entries.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_cache_key(name, fields)}.pkl"
    if cache_path.exists():
        try:
            with cache_path.open("rb") as cache_file:
                return pickle.load(cache_file)
        except (pickle.UnpicklingError, EOFError, ValueError):
            # Corrupt or partially written cache entry; refetch and overwrite it.
            pass

    result = loader()
    tmp_path = cache_path.with_suffix(".pkl.tmp")
    with tmp_path.open("wb") as cache_file:
        pickle.dump(result, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(cache_path)
    return result
