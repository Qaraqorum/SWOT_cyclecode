#!/usr/bin/env python3
"""全パスのSSHを2 km共通格子で平均し、SSH画像と3方向勾配を作成する。

既定入力
    /home/kuwabara/SWOTCode/SSHJapan/JapanSSHdata002.nc
既定出力
    /home/kuwabara/SWOTCode/Gradient/gradient_sshJp3.nc
    /home/kuwabara/SWOTCode/Gradient/gradient_sshJp3.png
    /home/kuwabara/SWOTCode/Gradient/ssh_heightJp3.png

入力内の有限な全SSH観測を、EPSG:6933（全球正積円筒図法）の2 kmセルへ
割り当てて算術平均する。平滑化や観測点の間引きは行わない。同じセルへ入る
複数パスの観測値はすべて平均へ使用する。

平均後の共通格子において、Crossは右隣、Alongは前方、Obliqueは右斜め前の
セルとの直接差分で求める。ObliqueをAlongとCrossの和から作ることはしない。
勾配m/mを1e6倍し、微小角近似によるマイクロラジアン（µrad）として保存する。

注意
----
複数軌道を共通地理格子へ平均した後なので、本ファイルのAlong（+y）とCross
（+x）は元衛星軌道固有の方向ではない。軌道方位を用いるLSA3インバージョン
では、この平均格子製品ではなくパス別のネイティブ勾配を使用すること。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import matplotlib
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
import netCDF4
import numpy as np
from pyproj import CRS, Geod, Transformer

# GUIのないLinux計算機でもPNGを生成できるバックエンドを明示する。
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from global_land_mask import globe
except ImportError as error:
    raise ImportError(
        "陸海判定に global-land-mask が必要です。"
        " `python -m pip install global-land-mask` を実行してください。"
    ) from error


DATA_DIRECTORY = Path("/home/kuwabara/SWOTCode")
DEFAULT_INPUT_NC = DATA_DIRECTORY / "SSHJapan" / "JapanSSHdata002.nc"
DEFAULT_OUTPUT_NC = DATA_DIRECTORY / "Gradient" / "gradient_sshJp3.nc"
DEFAULT_GRADIENT_PNG = DATA_DIRECTORY / "Gradient" / "gradient_sshJp3.png"
DEFAULT_SSH_PNG = DATA_DIRECTORY / "Gradient" / "ssh_heightJp3.png"

DEFAULT_LAT_MIN = 20.0
DEFAULT_LAT_MAX = 50.0
DEFAULT_LON_MIN = 110.0
DEFAULT_LON_MAX = 160.0
DEFAULT_GRID_RESOLUTION_M = 2000.0
DEFAULT_GRADIENT_LIMIT_URAD = 10.0
DEFAULT_BLOCK_LINES = 4096

GRID_CRS = CRS.from_epsg(6933)
GEOGRAPHIC_CRS = CRS.from_epsg(4326)
TO_GRID = Transformer.from_crs(GEOGRAPHIC_CRS, GRID_CRS, always_xy=True)
FROM_GRID = Transformer.from_crs(GRID_CRS, GEOGRAPHIC_CRS, always_xy=True)
WGS84 = Geod(ellps="WGS84")


def _normalize_longitude(longitude: np.ndarray) -> np.ndarray:
    """0～360度表現を含む経度を[-180, 180)へ統一する。"""
    return (longitude + 180.0) % 360.0 - 180.0


def _filled_float(values: np.ndarray) -> np.ndarray:
    """NetCDFのMaskedArrayをNaN表現可能なfloat64配列へ変換する。"""
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.asarray(values, dtype=np.float64)


def _projected_bounds(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> tuple[float, float, float, float]:
    """指定緯度経度範囲を完全に含むEPSG:6933上の矩形を返す。"""
    corner_lon = np.array([lon_min, lon_min, lon_max, lon_max], dtype=float)
    corner_lat = np.array([lat_min, lat_max, lat_min, lat_max], dtype=float)
    corner_x, corner_y = TO_GRID.transform(corner_lon, corner_lat)
    return (
        float(np.min(corner_x)),
        float(np.max(corner_x)),
        float(np.min(corner_y)),
        float(np.max(corner_y)),
    )


def _build_grid(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    resolution_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """2 km格子の投影座標中心と1次元緯度・経度座標を作成する。"""
    x_min, x_max, y_min, y_max = _projected_bounds(
        lat_min, lat_max, lon_min, lon_max
    )
    x_origin = np.floor(x_min / resolution_m) * resolution_m
    y_origin = np.floor(y_min / resolution_m) * resolution_m
    nx = int(np.ceil((x_max - x_origin) / resolution_m))
    ny = int(np.ceil((y_max - y_origin) / resolution_m))
    x = x_origin + (np.arange(nx, dtype=np.float64) + 0.5) * resolution_m
    y = y_origin + (np.arange(ny, dtype=np.float64) + 0.5) * resolution_m

    # EPSG:6933は円筒図法なので、経度はx、緯度はyだけから決まる。
    longitude, _ = FROM_GRID.transform(x, np.full_like(x, y[ny // 2]))
    _, latitude = FROM_GRID.transform(np.full_like(y, x[nx // 2]), y)
    return x, y, latitude, longitude, x_origin, y_origin


def _accumulate_by_cell(
    flat_sum: np.ndarray,
    flat_count: np.ndarray,
    flat_index: np.ndarray,
    values: np.ndarray,
) -> None:
    """同じセルの値をブロック内でまとめ、全体の合計・個数へ加算する。"""
    if flat_index.size == 0:
        return
    unique_index, inverse = np.unique(flat_index, return_inverse=True)
    block_sum = np.bincount(inverse, weights=values)
    block_count = np.bincount(inverse)
    flat_sum[unique_index] += block_sum
    flat_count[unique_index] += block_count.astype(flat_count.dtype)


def average_all_observations(
    input_nc: str | Path,
    ssh_name: str,
    latitude_name: str,
    longitude_name: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    resolution_m: float,
    block_lines: int,
) -> dict[str, np.ndarray | float | int | str]:
    """全観測を2 kmセルへ割り当て、全域平均と海洋観測平均を計算する。"""
    input_nc = Path(input_nc).expanduser()
    if not input_nc.is_file():
        raise FileNotFoundError(f"入力NetCDFがありません: {input_nc}")

    x, y, latitude, longitude, x_origin, y_origin = _build_grid(
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        resolution_m,
    )
    ny, nx = y.size, x.size
    cell_count = ny * nx
    all_sum = np.zeros(cell_count, dtype=np.float64)
    all_count = np.zeros(cell_count, dtype=np.int64)
    ocean_sum = np.zeros(cell_count, dtype=np.float64)
    ocean_count = np.zeros(cell_count, dtype=np.int64)

    observations_used = 0
    ocean_observations_used = 0
    with netCDF4.Dataset(input_nc, "r") as source:
        for variable_name in (ssh_name, latitude_name, longitude_name):
            if variable_name not in source.variables:
                raise KeyError(
                    f"必須変数 {variable_name!r} が {input_nc} にありません"
                )

        ssh_variable = source.variables[ssh_name]
        latitude_variable = source.variables[latitude_name]
        longitude_variable = source.variables[longitude_name]
        if ssh_variable.ndim != 2:
            raise ValueError(
                f"{ssh_name}は2次元である必要があります: {ssh_variable.dimensions}"
            )
        if (
            latitude_variable.dimensions != ssh_variable.dimensions
            or longitude_variable.dimensions != ssh_variable.dimensions
        ):
            raise ValueError("SSH・緯度・経度の次元が一致していません")

        line_count = ssh_variable.shape[0]
        for start in range(0, line_count, block_lines):
            stop = min(start + block_lines, line_count)
            ssh = _filled_float(ssh_variable[start:stop, :])
            lat = _filled_float(latitude_variable[start:stop, :])
            lon = _normalize_longitude(
                _filled_float(longitude_variable[start:stop, :])
            )
            valid = (
                np.isfinite(ssh)
                & np.isfinite(lat)
                & np.isfinite(lon)
                & (lat >= lat_min)
                & (lat <= lat_max)
                & (lon >= lon_min)
                & (lon <= lon_max)
            )
            if not np.any(valid):
                continue

            valid_ssh = ssh[valid]
            valid_lat = lat[valid]
            valid_lon = lon[valid]
            projected_x, projected_y = TO_GRID.transform(valid_lon, valid_lat)
            column = np.floor(
                (projected_x - x_origin) / resolution_m
            ).astype(np.int64)
            row = np.floor(
                (projected_y - y_origin) / resolution_m
            ).astype(np.int64)
            inside = (
                (row >= 0)
                & (row < ny)
                & (column >= 0)
                & (column < nx)
            )
            row = row[inside]
            column = column[inside]
            valid_ssh = valid_ssh[inside]
            valid_lat = valid_lat[inside]
            valid_lon = valid_lon[inside]
            flat_index = row * nx + column

            _accumulate_by_cell(all_sum, all_count, flat_index, valid_ssh)
            observations_used += valid_ssh.size

            is_ocean = np.asarray(
                globe.is_ocean(valid_lat, valid_lon), dtype=bool
            )
            _accumulate_by_cell(
                ocean_sum,
                ocean_count,
                flat_index[is_ocean],
                valid_ssh[is_ocean],
            )
            ocean_observations_used += int(np.count_nonzero(is_ocean))

            if start == 0 or stop == line_count or stop % (block_lines * 10) == 0:
                print(f"SSH平均化: {stop}/{line_count} lines")

    all_mean = np.full(cell_count, np.nan, dtype=np.float64)
    ocean_mean = np.full(cell_count, np.nan, dtype=np.float64)
    np.divide(
        all_sum,
        all_count,
        out=all_mean,
        where=all_count > 0,
    )
    np.divide(
        ocean_sum,
        ocean_count,
        out=ocean_mean,
        where=ocean_count > 0,
    )

    # 共通格子中心の陸海区分。海洋未観測セルと陸域を画像上で区別するため、
    # 観測数とは独立にGLOBE陸海マスクを評価する。
    grid_ocean_mask = np.empty((ny, nx), dtype=bool)
    mask_block_rows = 256
    for row_start in range(0, ny, mask_block_rows):
        row_stop = min(row_start + mask_block_rows, ny)
        latitude_block = np.broadcast_to(
            latitude[row_start:row_stop, np.newaxis],
            (row_stop - row_start, nx),
        )
        longitude_block = np.broadcast_to(
            longitude[np.newaxis, :],
            (row_stop - row_start, nx),
        )
        grid_ocean_mask[row_start:row_stop, :] = globe.is_ocean(
            latitude_block,
            longitude_block,
        )

    return {
        "x": x,
        "y": y,
        "latitude": latitude,
        "longitude": longitude,
        "ssh_mean_all": all_mean.reshape(ny, nx),
        "observation_count": all_count.reshape(ny, nx),
        "ssh_mean_ocean": ocean_mean.reshape(ny, nx),
        "ocean_observation_count": ocean_count.reshape(ny, nx),
        "grid_ocean_mask": grid_ocean_mask,
        "resolution_m": float(resolution_m),
        "observations_used": int(observations_used),
        "ocean_observations_used": int(ocean_observations_used),
        "source_file": str(input_nc),
    }


def compute_directional_gradients_urad(
    ssh: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> dict[str, np.ndarray]:
    """共通格子の右・前・右斜め前への直接差分をµradで計算する。"""
    ny, nx = ssh.shape
    if latitude.shape != (ny,) or longitude.shape != (nx,):
        raise ValueError("SSH形状と1次元緯度・経度座標が一致しません")

    along = np.full((ny, nx), np.nan, dtype=np.float32)
    cross = np.full((ny, nx), np.nan, dtype=np.float32)
    oblique = np.full((ny, nx), np.nan, dtype=np.float32)

    # Cross: (row, col) -> (row, col+1)。距離は緯度ごとに実測地距離を使う。
    _, _, cross_distance = WGS84.inv(
        np.full(ny, longitude[0]),
        latitude,
        np.full(ny, longitude[1]),
        latitude,
    )
    cross_difference = ssh[:, 1:] - ssh[:, :-1]
    cross[:, :-1] = (
        cross_difference / np.asarray(cross_distance)[:, np.newaxis] * 1.0e6
    ).astype(np.float32)

    # Along: (row, col) -> (row+1, col)。最終行は前方点がないためNaN。
    _, _, along_distance = WGS84.inv(
        np.full(ny - 1, longitude[0]),
        latitude[:-1],
        np.full(ny - 1, longitude[0]),
        latitude[1:],
    )
    along_difference = ssh[1:, :] - ssh[:-1, :]
    along[:-1, :] = (
        along_difference / np.asarray(along_distance)[:, np.newaxis] * 1.0e6
    ).astype(np.float32)

    # Oblique: (row, col) -> (row+1, col+1)のSSHを直接引く。
    # AlongとCrossの勾配値を加算して作る処理は一切行わない。
    _, _, oblique_distance = WGS84.inv(
        np.full(ny - 1, longitude[0]),
        latitude[:-1],
        np.full(ny - 1, longitude[1]),
        latitude[1:],
    )
    oblique_difference = ssh[1:, 1:] - ssh[:-1, :-1]
    oblique[:-1, :-1] = (
        oblique_difference
        / np.asarray(oblique_distance)[:, np.newaxis]
        * 1.0e6
    ).astype(np.float32)

    return {
        "slope_along_urad": along,
        "slope_cross_urad": cross,
        "slope_oblique_urad": oblique,
    }


def _compression_kwargs() -> dict[str, object]:
    """NetCDF4変数に共通する可逆圧縮設定を返す。"""
    return {
        "zlib": True,
        "complevel": 4,
        "shuffle": True,
    }


def _write_float_grid(
    dataset: netCDF4.Dataset,
    name: str,
    values: np.ndarray,
    long_name: str,
    units: str,
    chunksizes: tuple[int, int],
    extra_attributes: dict[str, object] | None = None,
) -> netCDF4.Variable:
    """圧縮float32格子を作成し、値と属性を書き込む。"""
    variable = dataset.createVariable(
        name,
        "f4",
        ("y", "x"),
        fill_value=np.float32(9.96921e36),
        chunksizes=chunksizes,
        **_compression_kwargs(),
    )
    variable[:, :] = np.asarray(values, dtype=np.float32)
    variable.long_name = long_name
    variable.units = units
    variable.coordinates = "latitude longitude"
    variable.grid_mapping = "spatial_ref"
    if extra_attributes:
        variable.setncatts(extra_attributes)
    return variable


def save_gradient_netcdf(
    output_nc: str | Path,
    grid: dict[str, np.ndarray | float | int | str],
    gradients: dict[str, np.ndarray],
    gradient_limit_urad: float,
) -> Path:
    """平均SSH、観測数、陸海別SSH、3方向µrad勾配を1つのNCへ保存する。"""
    output_nc = Path(output_nc).expanduser()
    output_nc.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(grid["x"])
    y = np.asarray(grid["y"])
    latitude = np.asarray(grid["latitude"])
    longitude = np.asarray(grid["longitude"])
    ssh_mean_all = np.asarray(grid["ssh_mean_all"])
    ssh_mean_ocean = np.asarray(grid["ssh_mean_ocean"])
    observation_count = np.asarray(grid["observation_count"])
    ocean_observation_count = np.asarray(grid["ocean_observation_count"])
    grid_ocean_mask = np.asarray(grid["grid_ocean_mask"], dtype=bool)
    ocean_range = np.where(
        (ssh_mean_ocean >= -100.0) & (ssh_mean_ocean <= 100.0),
        ssh_mean_ocean,
        np.nan,
    )
    chunksizes = (min(y.size, 512), min(x.size, 512))

    with netCDF4.Dataset(output_nc, "w", format="NETCDF4") as output:
        output.createDimension("y", y.size)
        output.createDimension("x", x.size)

        x_variable = output.createVariable("x", "f8", ("x",))
        x_variable[:] = x
        x_variable.standard_name = "projection_x_coordinate"
        x_variable.long_name = "2 km grid-cell centre x coordinate"
        x_variable.units = "m"

        y_variable = output.createVariable("y", "f8", ("y",))
        y_variable[:] = y
        y_variable.standard_name = "projection_y_coordinate"
        y_variable.long_name = "2 km grid-cell centre y coordinate"
        y_variable.units = "m"

        latitude_variable = output.createVariable("latitude", "f8", ("y",))
        latitude_variable[:] = latitude
        latitude_variable.standard_name = "latitude"
        latitude_variable.units = "degrees_north"

        longitude_variable = output.createVariable("longitude", "f8", ("x",))
        longitude_variable[:] = longitude
        longitude_variable.standard_name = "longitude"
        longitude_variable.units = "degrees_east"

        spatial_ref = output.createVariable("spatial_ref", "i4")
        spatial_ref.spatial_ref = GRID_CRS.to_wkt()
        spatial_ref.crs_wkt = GRID_CRS.to_wkt()
        spatial_ref.epsg_code = "EPSG:6933"

        _write_float_grid(
            output,
            "ssh_mean_all",
            ssh_mean_all,
            "mean SSH from all finite observations, including land",
            "m",
            chunksizes,
            {
                "cell_method": "all observations in each 2 km cell: mean",
                "land_handling": "land observations retained",
            },
        )
        _write_float_grid(
            output,
            "ssh_mean_ocean",
            ssh_mean_ocean,
            "mean SSH from observations classified as ocean",
            "m",
            chunksizes,
            {
                "cell_method": "ocean observations in each 2 km cell: mean",
                "land_mask": "global-land-mask GLOBE classification",
            },
        )
        _write_float_grid(
            output,
            "ssh_ocean_minus100_100",
            ocean_range,
            "ocean SSH restricted to the inclusive range -100 to 100 m",
            "m",
            chunksizes,
            {
                "valid_min": np.float32(-100.0),
                "valid_max": np.float32(100.0),
            },
        )

        count_variable = output.createVariable(
            "observation_count",
            "i4",
            ("y", "x"),
            fill_value=np.int32(-1),
            chunksizes=chunksizes,
            **_compression_kwargs(),
        )
        count_variable[:, :] = observation_count.astype(np.int32)
        count_variable.long_name = "number of all SSH observations averaged in cell"
        count_variable.units = "1"

        ocean_count_variable = output.createVariable(
            "ocean_observation_count",
            "i4",
            ("y", "x"),
            fill_value=np.int32(-1),
            chunksizes=chunksizes,
            **_compression_kwargs(),
        )
        ocean_count_variable[:, :] = ocean_observation_count.astype(np.int32)
        ocean_count_variable.long_name = (
            "number of ocean-classified SSH observations averaged in cell"
        )
        ocean_count_variable.units = "1"

        ocean_mask_variable = output.createVariable(
            "grid_ocean_mask",
            "u1",
            ("y", "x"),
            chunksizes=chunksizes,
            **_compression_kwargs(),
        )
        ocean_mask_variable[:, :] = grid_ocean_mask.astype(np.uint8)
        ocean_mask_variable.long_name = (
            "ocean mask at common-grid cell centres from global-land-mask"
        )
        ocean_mask_variable.flag_values = np.array([0, 1], dtype=np.uint8)
        ocean_mask_variable.flag_meanings = "land ocean"

        slope_descriptions = {
            "slope_along_urad": (
                "forward +y adjacent-cell SSH difference divided by geodesic distance"
            ),
            "slope_cross_urad": (
                "right +x adjacent-cell SSH difference divided by geodesic distance"
            ),
            "slope_oblique_urad": (
                "direct forward-right diagonal SSH difference divided by geodesic distance"
            ),
        }
        for name, description in slope_descriptions.items():
            raw = gradients[name]
            _write_float_grid(
                output,
                name,
                raw,
                description,
                "microradian",
                chunksizes,
                {
                    "display_units": "µrad",
                    "conversion_from_slope": "m/m multiplied by 1e6",
                    "source_ssh": "ssh_mean_all",
                    "difference_method": "direct adjacent-cell forward difference",
                },
            )

            within_name = name.replace("_urad", "_within_10_urad")
            within_limit = np.where(
                np.abs(raw) < gradient_limit_urad,
                raw,
                np.nan,
            )
            _write_float_grid(
                output,
                within_name,
                within_limit,
                description + "; values outside display limit set to missing",
                "microradian",
                chunksizes,
                {
                    "display_units": "µrad",
                    "absolute_display_limit": np.float32(gradient_limit_urad),
                },
            )

            exceedance_name = name.replace("_urad", "_over_10_mask")
            exceedance = output.createVariable(
                exceedance_name,
                "u1",
                ("y", "x"),
                fill_value=np.uint8(255),
                chunksizes=chunksizes,
                **_compression_kwargs(),
            )
            finite = np.isfinite(raw)
            mask_values = np.full(raw.shape, 255, dtype=np.uint8)
            mask_values[finite] = (
                np.abs(raw[finite]) >= gradient_limit_urad
            ).astype(np.uint8)
            exceedance[:, :] = mask_values
            exceedance.long_name = (
                f"1 where absolute {name} is at least {gradient_limit_urad:g} µrad"
            )
            exceedance.flag_values = np.array([0, 1], dtype=np.uint8)
            exceedance.flag_meanings = "within_limit exceeds_limit"

        output.setncatts(
            {
                "title": "Cycle 002 Japan SSH mean grid and directional gradients",
                "summary": (
                    "All finite observations are averaged in nominal 2 km EPSG:6933 "
                    "cells; gradients use direct right, forward, and forward-right "
                    "adjacent-cell differences"
                ),
                "Conventions": "CF-1.8, ACDD-1.3",
                "source_file": str(grid["source_file"]),
                "grid_crs": "EPSG:6933",
                "grid_resolution_m": np.float64(grid["resolution_m"]),
                "observations_used": np.int64(grid["observations_used"]),
                "ocean_observations_used": np.int64(
                    grid["ocean_observations_used"]
                ),
                "overlap_method": "arithmetic mean of every observation in each cell",
                "smoothing": "none",
                "gradient_units": "microradian (m/m multiplied by 1e6)",
                "direction_warning": (
                    "Along means +y and Cross means +x on the common grid; these are "
                    "not original per-pass satellite-track directions"
                ),
                "history": (
                    f"{datetime.now(timezone.utc).isoformat()}: generated by "
                    "gradient_sshJp3.py"
                ),
            }
        )

    return output_nc


def _finite_symmetric_limit(values: np.ndarray, percentile: float = 98.0) -> float:
    """有限値絶対値のパーセンタイルからゼロ対称の表示範囲を決める。"""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    limit = float(np.nanpercentile(np.abs(finite), percentile))
    return limit if np.isfinite(limit) and limit > 0.0 else 1.0


def plot_ssh_height(
    output_png: str | Path,
    longitude: np.ndarray,
    latitude: np.ndarray,
    ssh_mean_all: np.ndarray,
    ssh_mean_ocean: np.ndarray,
    grid_ocean_mask: np.ndarray,
    dpi: int,
) -> Path:
    """全域平均SSHと海洋-100～100 m SSHを1つのPNGへ描画する。"""
    output_png = Path(output_png).expanduser()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    colormap = plt.colormaps["RdBu_r"].copy()
    colormap.set_bad("white")

    ocean_range = np.where(
        (ssh_mean_ocean >= -100.0) & (ssh_mean_ocean <= 100.0),
        ssh_mean_ocean,
        np.nan,
    )
    all_limit = _finite_symmetric_limit(ssh_mean_all)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(17, 7),
        constrained_layout=True,
    )

    all_mesh = axes[0].pcolormesh(
        longitude,
        latitude,
        np.ma.masked_invalid(ssh_mean_all),
        shading="auto",
        cmap=colormap,
        norm=TwoSlopeNorm(vmin=-all_limit, vcenter=0.0, vmax=all_limit),
        rasterized=True,
    )
    axes[0].set_title("Mean SSH: all observations including land")
    all_colorbar = figure.colorbar(all_mesh, ax=axes[0], shrink=0.88)
    all_colorbar.set_label("SSH (m)")

    # 海洋観測のないセルを灰色の下地で示し、海洋SSHだけを上に描く。
    non_ocean_layer = np.ma.masked_where(
        grid_ocean_mask,
        np.ones_like(grid_ocean_mask, dtype=float),
    )
    axes[1].pcolormesh(
        longitude,
        latitude,
        non_ocean_layer,
        shading="auto",
        cmap=ListedColormap(["0.82"]),
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )
    ocean_mesh = axes[1].pcolormesh(
        longitude,
        latitude,
        np.ma.masked_invalid(ocean_range),
        shading="auto",
        cmap=colormap,
        norm=TwoSlopeNorm(vmin=-100.0, vcenter=0.0, vmax=100.0),
        rasterized=True,
    )
    axes[1].set_title("Ocean SSH only: -100 to 100 m")
    ocean_colorbar = figure.colorbar(ocean_mesh, ax=axes[1], shrink=0.88)
    ocean_colorbar.set_label("SSH (m)")

    for axis in axes:
        axis.set_xlabel("Longitude (degrees east)")
        axis.set_ylabel("Latitude (degrees north)")
        axis.set_xlim(float(longitude[0]), float(longitude[-1]))
        axis.set_ylim(float(latitude[0]), float(latitude[-1]))
        axis.set_aspect("equal", adjustable="box")

    figure.suptitle(
        "Cycle 002 Japan SSH: all-pass 2 km cell means",
        fontsize=14,
    )
    figure.savefig(output_png, dpi=dpi, facecolor="white")
    plt.close(figure)
    return output_png


def plot_gradients(
    output_png: str | Path,
    longitude: np.ndarray,
    latitude: np.ndarray,
    gradients: dict[str, np.ndarray],
    gradient_limit_urad: float,
    dpi: int,
) -> Path:
    """3方向µrad勾配を±10で着色し、範囲外を灰色表示する。"""
    output_png = Path(output_png).expanduser()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    colormap = plt.colormaps["RdBu_r"].copy()
    colormap.set_bad("white")
    gray_colormap = ListedColormap(["0.55"])
    norm = TwoSlopeNorm(
        vmin=-gradient_limit_urad,
        vcenter=0.0,
        vmax=gradient_limit_urad,
    )

    panels = (
        ("slope_along_urad", "Along: forward +y"),
        ("slope_cross_urad", "Cross: right +x"),
        ("slope_oblique_urad", "Oblique: direct forward-right diagonal"),
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(21, 7),
        constrained_layout=True,
    )
    for axis, (name, title) in zip(axes, panels):
        values = gradients[name]
        within = np.ma.masked_where(
            ~np.isfinite(values) | (np.abs(values) >= gradient_limit_urad),
            values,
        )
        beyond = np.ma.masked_where(
            ~np.isfinite(values) | (np.abs(values) < gradient_limit_urad),
            np.ones_like(values, dtype=float),
        )
        mesh = axis.pcolormesh(
            longitude,
            latitude,
            within,
            shading="auto",
            cmap=colormap,
            norm=norm,
            rasterized=True,
        )
        axis.pcolormesh(
            longitude,
            latitude,
            beyond,
            shading="auto",
            cmap=gray_colormap,
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )
        axis.set_title(title)
        axis.set_xlabel("Longitude (degrees east)")
        axis.set_ylabel("Latitude (degrees north)")
        axis.set_xlim(float(longitude[0]), float(longitude[-1]))
        axis.set_ylim(float(latitude[0]), float(latitude[-1]))
        axis.set_aspect("equal", adjustable="box")
        colorbar = figure.colorbar(mesh, ax=axis, shrink=0.88)
        colorbar.set_label("Gradient (µrad)")

    figure.legend(
        handles=[
            Patch(
                facecolor="0.55",
                edgecolor="none",
                label=f"|gradient| ≥ {gradient_limit_urad:g} µrad",
            )
        ],
        loc="lower center",
        frameon=False,
    )
    figure.suptitle(
        "Directional SSH gradients on all-observation mean grid",
        fontsize=14,
    )
    figure.savefig(output_png, dpi=dpi, facecolor="white")
    plt.close(figure)
    return output_png


def create_products(
    input_nc: str | Path = DEFAULT_INPUT_NC,
    output_nc: str | Path = DEFAULT_OUTPUT_NC,
    ssh_png: str | Path = DEFAULT_SSH_PNG,
    gradient_png: str | Path = DEFAULT_GRADIENT_PNG,
    ssh_name: str = "ssh_karin_2",
    latitude_name: str = "latitude",
    longitude_name: str = "longitude",
    lat_min: float = DEFAULT_LAT_MIN,
    lat_max: float = DEFAULT_LAT_MAX,
    lon_min: float = DEFAULT_LON_MIN,
    lon_max: float = DEFAULT_LON_MAX,
    resolution_m: float = DEFAULT_GRID_RESOLUTION_M,
    gradient_limit_urad: float = DEFAULT_GRADIENT_LIMIT_URAD,
    block_lines: int = DEFAULT_BLOCK_LINES,
    dpi: int = 220,
) -> tuple[Path, Path, Path]:
    """平均化、勾配計算、NC保存、2種類のPNG作成を連続実行する。"""
    if resolution_m <= 0.0:
        raise ValueError("resolution-mは正の値にしてください")
    if gradient_limit_urad <= 0.0:
        raise ValueError("gradient-limitは正の値にしてください")
    if block_lines <= 0:
        raise ValueError("block-linesは正の整数にしてください")
    if lat_min >= lat_max or lon_min >= lon_max:
        raise ValueError("緯度・経度範囲の最小値は最大値より小さくしてください")

    print("全パスの観測SSHを2 kmセルへ平均しています...")
    grid = average_all_observations(
        input_nc,
        ssh_name,
        latitude_name,
        longitude_name,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        resolution_m,
        block_lines,
    )
    if int(grid["observations_used"]) == 0:
        raise RuntimeError("指定範囲に有効なSSH観測がありません")
    print(
        "平均に使用した観測数: "
        f"{int(grid['observations_used']):,}（海洋: "
        f"{int(grid['ocean_observations_used']):,}）"
    )

    print("右・前・右斜め前への直接差分でmicro-radian勾配を計算しています...")
    gradients = compute_directional_gradients_urad(
        np.asarray(grid["ssh_mean_all"]),
        np.asarray(grid["latitude"]),
        np.asarray(grid["longitude"]),
    )
    saved_nc = save_gradient_netcdf(
        output_nc,
        grid,
        gradients,
        gradient_limit_urad,
    )
    print(f"NetCDFを保存しました: {saved_nc}")

    saved_ssh_png = plot_ssh_height(
        ssh_png,
        np.asarray(grid["longitude"]),
        np.asarray(grid["latitude"]),
        np.asarray(grid["ssh_mean_all"]),
        np.asarray(grid["ssh_mean_ocean"]),
        np.asarray(grid["grid_ocean_mask"]),
        dpi,
    )
    print(f"SSH画像を保存しました: {saved_ssh_png}")

    saved_gradient_png = plot_gradients(
        gradient_png,
        np.asarray(grid["longitude"]),
        np.asarray(grid["latitude"]),
        gradients,
        gradient_limit_urad,
        dpi,
    )
    print(f"勾配画像を保存しました: {saved_gradient_png}")
    return saved_nc, saved_ssh_png, saved_gradient_png


def _parser() -> argparse.ArgumentParser:
    """コマンドライン引数を定義する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_NC)
    parser.add_argument("--output-nc", type=Path, default=DEFAULT_OUTPUT_NC)
    parser.add_argument("--ssh-png", type=Path, default=DEFAULT_SSH_PNG)
    parser.add_argument(
        "--gradient-png",
        type=Path,
        default=DEFAULT_GRADIENT_PNG,
    )
    parser.add_argument("--ssh-var", default="ssh_karin_2")
    parser.add_argument("--lat-var", default="latitude")
    parser.add_argument("--lon-var", default="longitude")
    parser.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX)
    parser.add_argument("--lon-min", type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon-max", type=float, default=DEFAULT_LON_MAX)
    parser.add_argument(
        "--resolution-m",
        type=float,
        default=DEFAULT_GRID_RESOLUTION_M,
        help="共通格子の投影座標上のセル幅（m）",
    )
    parser.add_argument(
        "--gradient-limit",
        type=float,
        default=DEFAULT_GRADIENT_LIMIT_URAD,
        help="勾配画像の絶対値上限（µrad）。超過値は灰色",
    )
    parser.add_argument(
        "--block-lines",
        type=int,
        default=DEFAULT_BLOCK_LINES,
        help="一度に読む入力沿軌道ライン数",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def main() -> None:
    """コマンドライン指定に従い全出力を作成する。"""
    args = _parser().parse_args()
    try:
        create_products(
            input_nc=args.input,
            output_nc=args.output_nc,
            ssh_png=args.ssh_png,
            gradient_png=args.gradient_png,
            ssh_name=args.ssh_var,
            latitude_name=args.lat_var,
            longitude_name=args.lon_var,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
            resolution_m=args.resolution_m,
            gradient_limit_urad=args.gradient_limit,
            block_lines=args.block_lines,
            dpi=args.dpi,
        )
    except Exception as error:
        print(f"エラー: {error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
