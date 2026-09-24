"""Dependency-light geographic distance utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


EARTH_RADIUS_M = 6_371_008.8


def haversine_m(
    latitude_1: np.ndarray | float,
    longitude_1: np.ndarray | float,
    latitude_2: np.ndarray | float,
    longitude_2: np.ndarray | float,
) -> np.ndarray:
    """Vectorized great-circle distance in metres for WGS84 lon/lat values."""
    lat1 = np.radians(np.asarray(latitude_1, dtype=float))
    lon1 = np.radians(np.asarray(longitude_1, dtype=float))
    lat2 = np.radians(np.asarray(latitude_2, dtype=float))
    lon2 = np.radians(np.asarray(longitude_2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def nearest_mrt_exits(
    properties: pd.DataFrame,
    exits: pd.DataFrame,
    *,
    batch_size: int = 2_000,
) -> pd.DataFrame:
    """Find the closest physical MRT exit for unique property coordinates."""
    required_properties = {"location_key", "latitude", "longitude"}
    required_exits = {"station_name", "exit_code", "latitude", "longitude"}
    if not required_properties <= set(properties.columns):
        raise ValueError(f"Missing property fields: {sorted(required_properties - set(properties.columns))}")
    if not required_exits <= set(exits.columns):
        raise ValueError(f"Missing MRT fields: {sorted(required_exits - set(exits.columns))}")
    if exits.empty:
        raise ValueError("MRT exit reference is empty")

    valid = properties.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    exit_lat = exits["latitude"].to_numpy(dtype=float)[None, :]
    exit_lon = exits["longitude"].to_numpy(dtype=float)[None, :]
    chunks: list[pd.DataFrame] = []
    for start in range(0, len(valid), batch_size):
        chunk = valid.iloc[start : start + batch_size]
        distances = haversine_m(
            chunk["latitude"].to_numpy(dtype=float)[:, None],
            chunk["longitude"].to_numpy(dtype=float)[:, None],
            exit_lat,
            exit_lon,
        )
        nearest_index = distances.argmin(axis=1)
        nearest_distance = distances[np.arange(len(chunk)), nearest_index]
        nearest = exits.iloc[nearest_index].reset_index(drop=True)
        chunks.append(pd.DataFrame({
            "location_key": chunk["location_key"].to_numpy(),
            "nearest_mrt_station": nearest["station_name"].to_numpy(),
            "nearest_mrt_exit": nearest["exit_code"].to_numpy(),
            "nearest_mrt_distance_m": nearest_distance,
        }))
    if not chunks:
        return pd.DataFrame(columns=[
            "location_key", "nearest_mrt_station", "nearest_mrt_exit", "nearest_mrt_distance_m"
        ])
    return pd.concat(chunks, ignore_index=True)
