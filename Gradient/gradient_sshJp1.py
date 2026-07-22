#!/usr/bin/env python3
"""日本近海の結合SSHから3方向勾配NetCDFと確認用PNGを作成する。

入力は ``combine_japan_ssh_cycle.py`` で作成したCycle 002の
``JapanSSHdata002.nc`` を想定する。既定では次の3ファイルを使用する。

入力NetCDF
    /home/kuwabara/SwotData/Japan/JapanSSHdata002.nc
出力NetCDF
    /home/kuwabara/SwotData/Japan/gradient_sshJp1.nc
出力PNG
    /home/kuwabara/SwotData/Japan/gradient_sshJp1.png

このファイルだけで勾配計算、NetCDF保存、PNG描画まで実行できる。元の2 km
スワスグリッドを維持し、再グリッド化、平滑化、間引き、欠損値補間は行わない。
Along、Cross、Oblique（正の45度）の各勾配は、隣接SSH差をWGS84楕円体上の
実距離で割って求める。結合ファイルに ``source_file_index`` が存在する場合、
異なる元ファイルの境界を跨ぐAlong・Oblique差分は計算しない。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr
from pyproj import Geod

# GUIやXサーバーのないLinux計算機でもPNGを保存できる描画バックエンド。
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Kuwabara環境における既定の入出力先。
DATA_DIRECTORY = Path("/home/kuwabara/SwotData/Japan")
DEFAULT_INPUT_NC = DATA_DIRECTORY / "JapanSSHdata002.nc"
DEFAULT_OUTPUT_NC = DATA_DIRECTORY / "gradient_sshJp1.nc"
DEFAULT_OUTPUT_PNG = DATA_DIRECTORY / "gradient_sshJp1.png"

# 隣接ピクセル間の距離は固定値2 kmではなく、各点の緯度・経度から求める。
WGS84 = Geod(ellps="WGS84")


def _distance_m(
    lon0: np.ndarray,
    lat0: np.ndarray,
    lon1: np.ndarray,
    lat1: np.ndarray,
) -> np.ndarray:
    """2点間のWGS84楕円体上の測地距離をメートル単位で返す。"""
    _, _, distance = WGS84.inv(lon0, lat0, lon1, lat1)
    return np.asarray(distance, dtype=np.float64)


def _geodesic_distance(
    lon0: xr.DataArray,
    lat0: xr.DataArray,
    lon1: xr.DataArray,
    lat1: xr.DataArray,
) -> xr.DataArray:
    """NumPy配列とDask配列の両方に対応して実距離を計算する。"""
    return xr.apply_ufunc(
        _distance_m,
        lon0,
        lat0,
        lon1,
        lat1,
        dask="parallelized",
        output_dtypes=[np.float64],
    )


def _directional_slope(
    ssh: xr.DataArray,
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    shifts: dict[str, int],
    segment_index: xr.DataArray | None = None,
) -> xr.DataArray:
    """指定方向の直接隣接ピクセルからSSH勾配を計算する。

    通常は正方向の前方差分を使用し、配列終端または元入力ファイルの終端では
    後方差分を使用する。欠損値を越える内挿や遠方点への置換は行わない。
    """
    # xarray.shiftでは-1を指定すると、現在点へ正方向の隣接値が配置される。
    ssh_next = ssh.shift(shifts)
    lat_next = latitude.shift(shifts)
    lon_next = longitude.shift(shifts)
    distance_forward = _geodesic_distance(
        longitude, latitude, lon_next, lat_next
    )
    forward = (ssh_next - ssh) / distance_forward

    reverse_shifts = {dimension: -step for dimension, step in shifts.items()}
    ssh_previous = ssh.shift(reverse_shifts)
    lat_previous = latitude.shift(reverse_shifts)
    lon_previous = longitude.shift(reverse_shifts)
    distance_backward = _geodesic_distance(
        lon_previous, lat_previous, longitude, latitude
    )
    backward = (ssh - ssh_previous) / distance_backward

    # 結合前の入力ファイルが異なるライン同士は、隣接点として使用しない。
    same_forward = xr.ones_like(ssh, dtype=bool)
    same_backward = xr.ones_like(ssh, dtype=bool)
    if segment_index is not None:
        segment_shifts = {
            dimension: step
            for dimension, step in shifts.items()
            if dimension in segment_index.dims
        }
        if segment_shifts:
            same_forward = segment_index.shift(segment_shifts) == segment_index
            reverse_segment_shifts = {
                dimension: -step
                for dimension, step in segment_shifts.items()
            }
            same_backward = (
                segment_index.shift(reverse_segment_shifts) == segment_index
            )
            forward = forward.where(same_forward)
            backward = backward.where(same_backward)

    # 配列の正方向終端を検出する。Obliqueでは2軸のどちらかが終端なら後方差分。
    terminal = xr.zeros_like(ssh, dtype=bool)
    for dimension, step in shifts.items():
        if step != -1:
            raise ValueError("正方向のシフトには-1を指定してください")
        index = xr.DataArray(
            np.arange(ssh.sizes[dimension]),
            dims=(dimension,),
            coords={dimension: ssh[dimension]},
        )
        terminal = terminal | (index == ssh.sizes[dimension] - 1)

    # 元ファイルの最終ラインも、同じファイル内の直前点による後方差分とする。
    terminal = terminal | ~same_forward
    result = xr.where(terminal, backward, forward)
    return result.where(np.isfinite(result))


def _pick_spatial_dims(ssh: xr.DataArray) -> tuple[str, str]:
    """SSH変数から沿軌道・横軌道の次元名を決定する。"""
    if "num_lines" in ssh.dims and "num_pixels" in ssh.dims:
        return "num_lines", "num_pixels"
    if ssh.ndim == 2:
        return ssh.dims[0], ssh.dims[1]
    raise ValueError(
        f"SSH変数は2次元である必要があります。入力次元: {ssh.dims}"
    )


def compute_gradients(
    input_nc: str | Path,
    output_nc: str | Path,
    ssh_name: str = "ssh_karin_2",
    latitude_name: str = "latitude",
    longitude_name: str = "longitude",
    chunks: str | int | None = "auto",
) -> Path:
    """日本近海SSHから3方向の勾配を計算してNetCDFへ保存する。"""
    input_nc = Path(input_nc)
    output_nc = Path(output_nc)

    # FillValue、scale_factor、add_offsetをデコードし、欠損値をNaNとして読む。
    dataset = xr.open_dataset(
        input_nc,
        decode_cf=True,
        mask_and_scale=True,
        chunks=chunks,
    )
    try:
        for variable_name in (ssh_name, latitude_name, longitude_name):
            if variable_name not in dataset:
                raise KeyError(
                    f"必須変数 {variable_name!r} が {input_nc} にありません"
                )

        ssh = dataset[ssh_name].astype(np.float64)
        latitude = dataset[latitude_name].astype(np.float64)
        longitude = dataset[longitude_name].astype(np.float64)
        along_dim, cross_dim = _pick_spatial_dims(ssh)
        if latitude.dims != ssh.dims or longitude.dims != ssh.dims:
            raise ValueError(
                "latitude、longitude、SSHは同一の2次元グリッドが必要です"
            )

        # combine_japan_ssh_cycle.pyが付けた元ファイル索引を境界判定に使う。
        segment_index = dataset.get("source_file_index")
        if segment_index is not None and segment_index.dims != (along_dim,):
            raise ValueError(
                "source_file_indexは沿軌道次元だけを持つ必要があります"
            )

        slope_along = _directional_slope(
            ssh,
            latitude,
            longitude,
            {along_dim: -1},
            segment_index=segment_index,
        )
        slope_cross = _directional_slope(
            ssh,
            latitude,
            longitude,
            {cross_dim: -1},
            segment_index=segment_index,
        )
        # 正の45度方向は(line, pixel)から(line+1, pixel+1)への対角差分。
        slope_oblique = _directional_slope(
            ssh,
            latitude,
            longitude,
            {along_dim: -1, cross_dim: -1},
            segment_index=segment_index,
        )

        # SSH、座標、時刻、元ファイル追跡情報を勾配と同じNCへ保持する。
        keep_variables = {ssh_name, latitude_name, longitude_name}
        keep_variables.update(dataset.coords)
        keep_variables.update(
            variable_name
            for variable_name in (
                "time",
                "source_file_index",
                "source_line_index",
            )
            if variable_name in dataset
        )
        output = dataset[
            [
                variable_name
                for variable_name in dataset.variables
                if variable_name in keep_variables
            ]
        ].copy()
        output[ssh_name] = dataset[ssh_name]
        output["slope_along"] = slope_along.astype(np.float32)
        output["slope_cross"] = slope_cross.astype(np.float32)
        output["slope_oblique"] = slope_oblique.astype(np.float32)

        common_attributes = {
            "units": "m m-1",
            "coordinates": f"{latitude_name} {longitude_name}",
            "source_ssh": ssh_name,
            "distance_method": "WGS84 ellipsoidal inverse geodesic",
            "difference_method": (
                "adjacent-pixel forward difference; backward at array or "
                "source-file terminal edge"
            ),
            "resolution_note": (
                "native grid retained; no resampling, smoothing, or interpolation"
            ),
            "file_boundary_note": (
                "differences never cross source_file_index boundaries"
            ),
        }
        output["slope_along"].attrs = {
            **common_attributes,
            "long_name": "sea surface slope in positive along-track direction",
            "direction": f"increasing {along_dim}",
        }
        output["slope_cross"].attrs = {
            **common_attributes,
            "long_name": "sea surface slope in positive cross-track direction",
            "direction": f"increasing {cross_dim}",
        }
        output["slope_oblique"].attrs = {
            **common_attributes,
            "long_name": "sea surface slope along positive 45-degree diagonal",
            "direction": (
                f"simultaneously increasing {along_dim} and {cross_dim}"
            ),
            "geometry_note": (
                "distance is measured between each pair of diagonal neighbours"
            ),
        }

        output.attrs.update(dataset.attrs)
        previous_history = dataset.attrs.get("history", "")
        output.attrs.update(
            {
                "title": (
                    "Native-grid SWOT SSH and adjacent-pixel directional slopes"
                ),
                "history": (
                    (previous_history + "\n" if previous_history else "")
                    + f"{datetime.now(timezone.utc).isoformat()}: generated by "
                    "gradient_sshJp1.py from "
                    + input_nc.name
                ),
                "slope_sign_convention": (
                    "(SSH at positive-direction neighbour - SSH at current "
                    "pixel) / distance"
                ),
                "Conventions": dataset.attrs.get("Conventions", "CF-1.8"),
            }
        )

        output_nc.parent.mkdir(parents=True, exist_ok=True)
        encoding = {
            variable_name: {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "dtype": "float32",
                "_FillValue": np.float32(9.96921e36),
                "chunksizes": tuple(
                    min(output.sizes[dimension], 512)
                    for dimension in ssh.dims
                ),
            }
            for variable_name in (
                "slope_along",
                "slope_cross",
                "slope_oblique",
            )
        }
        output.to_netcdf(
            output_nc,
            engine="netcdf4",
            format="NETCDF4",
            encoding=encoding,
        )
    finally:
        dataset.close()

    return output_nc


def _robust_symmetric_limit(
    values: np.ndarray,
    percentile: float,
) -> float:
    """勾配PNG用のゼロ対称な色範囲を外れ値に強い方法で決める。"""
    finite_values = np.asarray(values)[np.isfinite(values)]
    if finite_values.size == 0:
        return 1.0
    limit = float(np.nanpercentile(np.abs(finite_values), percentile))
    return limit if np.isfinite(limit) and limit > 0.0 else 1.0


def plot_gradients(
    input_nc: str | Path,
    output_png: str | Path,
    ssh_name: str = "ssh_karin_2",
    latitude_name: str = "latitude",
    longitude_name: str = "longitude",
    slope_percentile: float = 98.0,
    dpi: int = 220,
) -> Path:
    """保存した勾配NetCDFを読み直し、SSHと3方向勾配をPNGへ描画する。"""
    input_nc = Path(input_nc)
    output_png = Path(output_png)

    with xr.open_dataset(input_nc, mask_and_scale=True) as dataset:
        for variable_name in (
            ssh_name,
            latitude_name,
            longitude_name,
            "slope_along",
            "slope_cross",
            "slope_oblique",
        ):
            if variable_name not in dataset:
                raise KeyError(
                    f"描画に必要な変数 {variable_name!r} がありません"
                )
        latitude = dataset[latitude_name].values
        longitude = dataset[longitude_name].values
        ssh = dataset[ssh_name].values
        slopes = [
            dataset[variable_name].values
            for variable_name in (
                "slope_along",
                "slope_cross",
                "slope_oblique",
            )
        ]

    ssh_colormap = plt.colormaps["viridis"].copy()
    slope_colormap = plt.colormaps["RdBu_r"].copy()
    ssh_colormap.set_bad("white")
    slope_colormap.set_bad("white")

    finite_ssh = ssh[np.isfinite(ssh)]
    ssh_limits = (
        np.nanpercentile(finite_ssh, [2.0, 98.0])
        if finite_ssh.size
        else (-1.0, 1.0)
    )
    panels = [
        (
            ssh,
            "SSH",
            ssh_colormap,
            float(ssh_limits[0]),
            float(ssh_limits[1]),
        )
    ]
    for values, title in zip(
        slopes,
        (
            "Along-track slope",
            "Cross-track slope",
            "Oblique +45-degree slope",
        ),
    ):
        limit = _robust_symmetric_limit(values, slope_percentile)
        panels.append((values, title, slope_colormap, -limit, limit))

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15, 10),
        constrained_layout=True,
    )
    for axis, (values, title, colormap, value_min, value_max) in zip(
        axes.flat,
        panels,
    ):
        # pcolormeshはX・Y座標のNaNを受け付けない。日本近海への切り出しで
        # 緯度・経度もNaNにしたセルが存在するため、有限な座標・値を持つ
        # 観測点をすべて抽出して描画する。ここで間引きは行っていない。
        valid = (
            np.isfinite(longitude)
            & np.isfinite(latitude)
            & np.isfinite(values)
        )
        mesh = axis.scatter(
            longitude[valid],
            latitude[valid],
            c=values[valid],
            marker="s",
            s=0.18,
            linewidths=0.0,
            cmap=colormap,
            vmin=value_min,
            vmax=value_max,
            rasterized=True,
        )
        axis.set_title(title)
        axis.set_xlabel("Longitude (degrees east)")
        axis.set_ylabel("Latitude (degrees north)")
        axis.set_aspect("equal", adjustable="box")
        colorbar = figure.colorbar(mesh, ax=axis, shrink=0.88)
        colorbar.set_label("m" if title == "SSH" else "m/m")

    figure.suptitle(
        "SWOT native 2-km swath: SSH and adjacent-pixel slopes",
        fontsize=14,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=dpi, facecolor="white")
    plt.close(figure)
    return output_png


def create_japan_gradients(
    input_nc: str | Path = DEFAULT_INPUT_NC,
    output_nc: str | Path = DEFAULT_OUTPUT_NC,
    output_png: str | Path = DEFAULT_OUTPUT_PNG,
    ssh_name: str = "ssh_karin_2",
    latitude_name: str = "latitude",
    longitude_name: str = "longitude",
    slope_percentile: float = 98.0,
    dpi: int = 220,
    use_dask: bool = True,
) -> tuple[Path, Path]:
    """日本近海SSHの勾配NetCDFを作成し、そのファイルからPNGを描画する。

    Parameters
    ----------
    input_nc
        ``JapanSSHdata002.nc`` のパス。
    output_nc, output_png
        作成する勾配NetCDFと確認用PNGのパス。
    ssh_name, latitude_name, longitude_name
        入力ファイル内のSSH・緯度・経度変数名。
    slope_percentile
        PNGの勾配表示範囲を決める絶対値パーセンタイル。NetCDF内の数値は
        変更・クリッピングされない。
    dpi
        PNGの出力解像度。
    use_dask
        Trueの場合、勾配計算をDaskの遅延・チャンク処理で実行する。

    Returns
    -------
    tuple[Path, Path]
        作成したNetCDFとPNGのパス。
    """
    input_nc = Path(input_nc).expanduser()
    output_nc = Path(output_nc).expanduser()
    output_png = Path(output_png).expanduser()

    if not input_nc.is_file():
        raise FileNotFoundError(
            f"入力NetCDFがありません: {input_nc}\n"
            "先に combine_japan_ssh_cycle.py を実行してください。"
        )
    if not 0.0 < slope_percentile <= 100.0:
        raise ValueError("percentileは0より大きく100以下で指定してください")
    if dpi <= 0:
        raise ValueError("dpiは正の整数で指定してください")

    print(f"入力SSH: {input_nc}")
    print("Along・Cross・Obliqueの3方向勾配を計算しています...")
    gradient_nc = compute_gradients(
        input_nc=input_nc,
        output_nc=output_nc,
        ssh_name=ssh_name,
        latitude_name=latitude_name,
        longitude_name=longitude_name,
        chunks="auto" if use_dask else None,
    )
    print(f"勾配NetCDFを保存しました: {gradient_nc}")

    # 必ず保存済みの勾配NetCDFを読み直してPNGを作成する。
    print("保存したNetCDFを読み込み、確認用PNGを作成しています...")
    gradient_png = plot_gradients(
        input_nc=gradient_nc,
        output_png=output_png,
        ssh_name=ssh_name,
        latitude_name=latitude_name,
        longitude_name=longitude_name,
        slope_percentile=slope_percentile,
        dpi=dpi,
    )
    print(f"確認用PNGを保存しました: {gradient_png}")
    return gradient_nc, gradient_png


def _parser() -> argparse.ArgumentParser:
    """コマンドライン引数を定義する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_NC,
        help="入力JapanSSHdata002.ncのパス",
    )
    parser.add_argument(
        "--output-nc",
        type=Path,
        default=DEFAULT_OUTPUT_NC,
        help="出力gradient_sshJp1.ncのパス",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=DEFAULT_OUTPUT_PNG,
        help="出力gradient_sshJp1.pngのパス",
    )
    parser.add_argument("--ssh-var", default="ssh_karin_2", help="SSH変数名")
    parser.add_argument("--lat-var", default="latitude", help="緯度変数名")
    parser.add_argument("--lon-var", default="longitude", help="経度変数名")
    parser.add_argument(
        "--percentile",
        type=float,
        default=98.0,
        help="PNGの勾配色範囲を決める絶対値パーセンタイル",
    )
    parser.add_argument("--dpi", type=int, default=220, help="PNGの解像度")
    parser.add_argument(
        "--no-dask",
        action="store_true",
        help="Daskを使わずに勾配計算する",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="勾配を再計算せず、既存の--output-ncからPNGだけを作成する",
    )
    return parser


def main() -> None:
    """既定値またはコマンドライン指定に従って勾配計算と描画を実行する。"""
    args = _parser().parse_args()
    if args.plot_only:
        gradient_nc = args.output_nc.expanduser()
        if not gradient_nc.is_file():
            raise FileNotFoundError(
                f"PNG描画元の勾配NetCDFがありません: {gradient_nc}"
            )
        gradient_png = plot_gradients(
            input_nc=gradient_nc,
            output_png=args.output_png,
            ssh_name=args.ssh_var,
            latitude_name=args.lat_var,
            longitude_name=args.lon_var,
            slope_percentile=args.percentile,
            dpi=args.dpi,
        )
        print(f"確認用PNGを保存しました: {gradient_png}")
        return

    create_japan_gradients(
        input_nc=args.input,
        output_nc=args.output_nc,
        output_png=args.output_png,
        ssh_name=args.ssh_var,
        latitude_name=args.lat_var,
        longitude_name=args.lon_var,
        slope_percentile=args.percentile,
        dpi=args.dpi,
        use_dask=not args.no_dask,
    )


if __name__ == "__main__":
    main()
