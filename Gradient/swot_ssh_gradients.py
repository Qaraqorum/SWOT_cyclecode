#!/usr/bin/env python3
"""SWOT L2_LR_SSHのネイティブグリッド上で海面勾配を計算・可視化する。

リサンプリング、平滑化、内挿、間引きは一切行わない。各方向の勾配は、
元グリッド上で直接隣り合う2ピクセルのSSH差を、両点間のWGS84楕円体上の
実距離で割って計算する。Along、Cross、Obliqueの3方向を独立に求め、
元のSSH・座標とともにNetCDFへ保存し、そのNetCDFから確認用PNGを作成する。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pyproj import Geod

WGS84 = Geod(ellps="WGS84")


def _distance_m(
    lon0: np.ndarray, lat0: np.ndarray, lon1: np.ndarray, lat1: np.ndarray
) -> np.ndarray:
    """2点間のWGS84測地距離（m）を配列単位で計算する。

    ``pyproj.Geod.inv`` は入力配列をベクトル化して処理する。いずれかの
    端点座標がNaNの場合は、対応する距離もNaNのまま返す。
    """
    _, _, distance = WGS84.inv(lon0, lat0, lon1, lat1)
    return np.asarray(distance, dtype=np.float64)


def _geodesic_distance(
    lon0: xr.DataArray,
    lat0: xr.DataArray,
    lon1: xr.DataArray,
    lat1: xr.DataArray,
) -> xr.DataArray:
    """Xarray/Dask対応のWGS84測地距離計算を適用する。

    ``xr.apply_ufunc`` を使うことで、ピクセルごとのPythonループを避け、
    Dask配列の場合はチャンクごとに並列実行できるようにする。
    """
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
    """指定方向の隣接ピクセル差分から海面勾配を計算する。

    通常点では正方向の隣接ピクセルを使った前方差分を採用する。出力配列の
    形状を入力と完全に一致させるため、正方向の隣接点が存在しない配列終端
    だけは同じ軸方向の後方差分を使用する。欠損点を跨ぐ内挿は行わない。

    Parameters
    ----------
    ssh
        2次元の海面高度。
    latitude, longitude
        ``ssh`` と同じグリッドを持つ2次元緯度・経度。
    shifts
        正方向の隣接点を表すXarrayのシフト指定。沿軌道は1軸、横軌道は
        1軸、斜め方向は2軸を同時に ``-1`` とする。
    segment_index
        結合前の入力ファイルを示す沿軌道ライン単位の索引。指定された場合、
        異なる元ファイル間を跨ぐAlong・Oblique差分を無効にする。
    """
    # 正方向に隣接するSSH・緯度・経度を、座標を変えずにシフトして取得する。
    ssh_next = ssh.shift(shifts)
    lat_next = latitude.shift(shifts)
    lon_next = longitude.shift(shifts)
    d_forward = _geodesic_distance(longitude, latitude, lon_next, lat_next)
    forward = (ssh_next - ssh) / d_forward

    # 配列終端で使用する、逆側の直接隣接点による後方差分を用意する。
    reverse_shifts = {dim: -step for dim, step in shifts.items()}
    ssh_prev = ssh.shift(reverse_shifts)
    lat_prev = latitude.shift(reverse_shifts)
    lon_prev = longitude.shift(reverse_shifts)
    d_backward = _geodesic_distance(lon_prev, lat_prev, longitude, latitude)
    backward = (ssh - ssh_prev) / d_backward

    # 複数スワスの結合データでは、ファイル境界を隣接ピクセルとみなさない。
    # 横軌道差分だけの場合、segment_indexに横軌道次元がないため常に同一
    # セグメントとして扱う。
    same_forward = xr.ones_like(ssh, dtype=bool)
    same_backward = xr.ones_like(ssh, dtype=bool)
    if segment_index is not None:
        segment_shifts = {
            dim: step for dim, step in shifts.items() if dim in segment_index.dims
        }
        if segment_shifts:
            same_forward = segment_index.shift(segment_shifts) == segment_index
            reverse_segment_shifts = {
                dim: -step for dim, step in segment_shifts.items()
            }
            same_backward = (
                segment_index.shift(reverse_segment_shifts) == segment_index
            )
            forward = forward.where(same_forward)
            backward = backward.where(same_backward)

    # xarray.shiftは output[i] = input[i - shift] と定義されるため、-1が
    # 正方向の隣接点を表す。後方差分を選択するのは各軸の最終端だけである。
    edge = xr.zeros_like(ssh, dtype=bool)
    for dim, step in shifts.items():
        if step != -1:
            raise ValueError("正方向のシフトには -1 を指定してください")
        idx = xr.DataArray(
            np.arange(ssh.sizes[dim]), dims=(dim,), coords={dim: ssh[dim]}
        )
        edge = edge | (idx == ssh.sizes[dim] - 1)

    # 元ファイルの最終ラインでは、次ファイルを跨ぐ前方差分ではなく、同じ
    # ファイル内の直接隣接ラインを使った後方差分へ切り替える。
    edge = edge | ~same_forward

    result = xr.where(edge, backward, forward)
    # 距離ゼロや無効な座標による非有限値を除き、SSH・位置情報の欠損を保持する。
    return result.where(np.isfinite(result))


def _pick_spatial_dims(ssh: xr.DataArray) -> tuple[str, str]:
    """SSH配列から沿軌道・横軌道に対応する2つの空間次元を決定する。

    SWOT標準の ``num_lines``・``num_pixels`` を優先する。別名の2次元配列も
    利用できるよう、標準次元がない場合は配列の第1・第2次元を使用する。
    """
    preferred = ("num_lines", "num_pixels")
    if all(dim in ssh.dims for dim in preferred):
        return preferred
    if ssh.ndim == 2:
        return ssh.dims[0], ssh.dims[1]
    raise ValueError(
        f"SSH変数は2次元である必要があります。入力次元: {ssh.dims}。"
        "空間以外の次元をあらかじめ選択またはsqueezeしてください。"
    )


def compute_gradients(
    input_nc: str | Path,
    output_nc: str | Path = "swot_ssh_gradients_2km.nc",
    ssh_name: str = "ssh_karin_2",
    latitude_name: str = "latitude",
    longitude_name: str = "longitude",
    chunks: str | int | None = "auto",
) -> Path:
    """3方向の海面勾配を計算し、座標とともに1つのNetCDFへ保存する。

    入力SSHと2次元緯度・経度は元の形状・解像度のまま使用する。Alongは
    ``num_lines`` 方向、Crossは ``num_pixels`` 方向、Obliqueは両軸を
    同時に1ピクセル進む正の45度対角方向として計算する。

    Parameters
    ----------
    input_nc, output_nc
        入力SWOT NetCDFと出力NetCDFのパス。
    ssh_name, latitude_name, longitude_name
        入力ファイル内のSSH・緯度・経度変数名。
    chunks
        Xarray/Daskのチャンク指定。``"auto"`` で自動チャンク、``None`` で
        Daskを使わずに読み込む。

    Returns
    -------
    Path
        作成したNetCDFファイルのパス。
    """
    input_nc, output_nc = Path(input_nc), Path(output_nc)
    # CFスケール・オフセットとFillValueをデコードし、欠損値をNaNとして読む。
    ds = xr.open_dataset(input_nc, decode_cf=True, mask_and_scale=True, chunks=chunks)
    for name in (ssh_name, latitude_name, longitude_name):
        if name not in ds:
            raise KeyError(f"必須変数 {name!r} が {input_nc} にありません")

    ssh = ds[ssh_name].astype(np.float64)
    lat = ds[latitude_name].astype(np.float64)
    lon = ds[longitude_name].astype(np.float64)
    along_dim, cross_dim = _pick_spatial_dims(ssh)
    if lat.dims != ssh.dims or lon.dims != ssh.dims:
        raise ValueError("latitude、longitude、SSHは同一の2次元グリッドが必要です")

    segment_index = ds.get("source_file_index")
    if segment_index is not None and segment_index.dims != (along_dim,):
        raise ValueError(
            "source_file_indexは沿軌道次元だけを持つ1次元変数である必要があります"
        )

    # 3方向をそれぞれ独立に計算する。各結果の形状は元SSHと同一になる。
    along = _directional_slope(
        ssh, lat, lon, {along_dim: -1}, segment_index=segment_index
    )
    cross = _directional_slope(
        ssh, lat, lon, {cross_dim: -1}, segment_index=segment_index
    )
    # 正の45度対角方向: (line, pixel) -> (line+1, pixel+1)。
    oblique = _directional_slope(
        ssh,
        lat,
        lon,
        {along_dim: -1, cross_dim: -1},
        segment_index=segment_index,
    )

    # 次元座標に加え、2次元位置情報とライン時刻を出力へ引き継ぐ。
    keep = {ssh_name, latitude_name, longitude_name}
    keep.update(ds.coords)
    if "time" in ds:
        keep.add("time")
    keep.update(
        name
        for name in ("source_file_index", "source_line_index")
        if name in ds
    )
    out = ds[[name for name in ds.variables if name in keep]].copy()
    out[ssh_name] = ds[ssh_name]  # 元SSHのデータ型と属性を維持する。
    out["slope_along"] = along.astype(np.float32)
    out["slope_cross"] = cross.astype(np.float32)
    out["slope_oblique"] = oblique.astype(np.float32)

    # NetCDF属性はPanoply・GMT等との相互運用性を考慮して英語で記録する。
    common = {
        "units": "m m-1",
        "coordinates": f"{latitude_name} {longitude_name}",
        "source_ssh": ssh_name,
        "distance_method": "WGS84 ellipsoidal inverse geodesic",
        "difference_method": (
            "adjacent-pixel forward difference; backward at array or source-file terminal edge"
        ),
        "resolution_note": "native grid retained; no resampling, smoothing, or interpolation",
        "file_boundary_note": "differences never cross source_file_index boundaries",
    }
    out["slope_along"].attrs = {
        **common,
        "long_name": "sea surface slope in positive along-track grid direction",
        "direction": f"increasing {along_dim}",
    }
    out["slope_cross"].attrs = {
        **common,
        "long_name": "sea surface slope in positive cross-track grid direction",
        "direction": f"increasing {cross_dim}",
    }
    out["slope_oblique"].attrs = {
        **common,
        "long_name": "sea surface slope along positive 45-degree grid diagonal",
        "direction": f"simultaneously increasing {along_dim} and {cross_dim}",
        "geometry_note": "distance is measured between diagonal neighbours, not assumed to be 2828 m",
    }
    out.attrs.update(ds.attrs)
    out.attrs.update(
        {
            "title": "Native-grid SWOT SSH and adjacent-pixel directional slopes",
            "history": (ds.attrs.get("history", "") + "\n" if ds.attrs.get("history") else "") + (
                f"{datetime.now(timezone.utc).isoformat()}: generated by "
                "swot_ssh_gradients.py from " + input_nc.name
            ),
            "slope_sign_convention": "(SSH at positive-direction neighbour - SSH at current pixel) / distance",
            "Conventions": ds.attrs.get("Conventions", "CF-1.8"),
        }
    )

    output_nc.parent.mkdir(parents=True, exist_ok=True)
    # 勾配はfloat32で保存し、NetCDF-4の可逆圧縮を適用する。
    encoding = {
        name: {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "dtype": "float32",
            "_FillValue": np.float32(9.96921e36),
            "chunksizes": tuple(min(out.sizes[d], 512) for d in ssh.dims),
        }
        for name in ("slope_along", "slope_cross", "slope_oblique")
    }
    out.to_netcdf(output_nc, engine="netcdf4", format="NETCDF4", encoding=encoding)
    ds.close()
    return output_nc


def _robust_symmetric_limit(values: np.ndarray, percentile: float = 98.0) -> float:
    """外れ値に強い、ゼロ対称の描画上限絶対値を求める。

    有限値の絶対値パーセンタイルを採用する。これはPNGの色範囲だけに使い、
    NetCDFへ保存する勾配値のクリッピングや変更は行わない。
    """
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    limit = float(np.nanpercentile(np.abs(finite), percentile))
    return limit if np.isfinite(limit) and limit > 0 else 1.0


def plot_gradients(
    input_nc: str | Path,
    output_png: str | Path = "swot_gradients_map.png",
    ssh_name: str = "ssh_karin_2",
    latitude_name: str = "latitude",
    longitude_name: str = "longitude",
    slope_percentile: float = 98.0,
    dpi: int = 220,
) -> Path:
    """保存済みNetCDFを読み、全ネイティブセルを4パネルのPNGへ描画する。

    SSH、Along、Cross、Obliqueを2×2に配置する。NaNは白抜き表示とし、
    勾配はゼロを中心とする ``RdBu_r`` カラーマップを使用する。描画のための
    間引きは行わず、入力NetCDF内のすべてのグリッドセルを渡す。

    Returns
    -------
    Path
        作成したPNGファイルのパス。
    """
    input_nc, output_png = Path(input_nc), Path(output_png)
    # 描画段階では保存済みNetCDFを改めて開き、出力ファイル単独で再現できるようにする。
    with xr.open_dataset(input_nc, mask_and_scale=True) as ds:
        lat = ds[latitude_name].values
        lon = ds[longitude_name].values
        ssh = ds[ssh_name].values
        slopes = [ds[n].values for n in ("slope_along", "slope_cross", "slope_oblique")]

    # 欠損域（スワス中央ギャップ、陸域など）は白色で表示する。
    cmap_ssh = plt.colormaps["viridis"].copy()
    cmap_slope = plt.colormaps["RdBu_r"].copy()
    cmap_ssh.set_bad("white")
    cmap_slope.set_bad("white")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    # SSHと勾配の表示範囲だけをロバストに決める。元データ値は変更しない。
    ssh_finite = ssh[np.isfinite(ssh)]
    ssh_limits = (
        np.nanpercentile(ssh_finite, [2, 98]) if ssh_finite.size else (-1.0, 1.0)
    )
    panels = [(ssh, "SSH", cmap_ssh, float(ssh_limits[0]), float(ssh_limits[1]))]
    for data, title in zip(slopes, ("Along-track slope", "Cross-track slope", "Oblique +45° slope")):
        lim = _robust_symmetric_limit(data, slope_percentile)
        panels.append((data, title, cmap_slope, -lim, lim))

    for ax, (data, title, cmap, vmin, vmax) in zip(axes.flat, panels):
        mesh = ax.pcolormesh(lon, lat, np.ma.masked_invalid(data), shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Longitude (degrees east)")
        ax.set_ylabel("Latitude (degrees north)")
        ax.set_aspect("equal", adjustable="box")
        cb = fig.colorbar(mesh, ax=ax, shrink=0.88)
        cb.set_label("m" if title == "SSH" else "m/m")

    fig.suptitle("SWOT native 2-km swath: SSH and adjacent-pixel slopes", fontsize=14)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, facecolor="white")
    plt.close(fig)
    return output_png


def _parser() -> argparse.ArgumentParser:
    """日本語のコマンドライン引数パーサーを作成する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    compute = sub.add_parser("compute", help="勾配を計算してNetCDFへ保存する")
    both = sub.add_parser("all", help="NetCDF作成とPNG描画を連続実行する")
    plot = sub.add_parser("plot", help="作成済みNetCDFからPNGを描画する")
    for p in (compute, both):
        p.add_argument("input", type=Path, help="入力SWOT L2_LR_SSH NetCDF")
        p.add_argument("--output-nc", type=Path, default=Path("swot_ssh_gradients_2km.nc"), help="出力NetCDFのパス")
        p.add_argument("--ssh-var", default="ssh_karin_2", help="SSH変数名")
        p.add_argument("--lat-var", default="latitude", help="緯度変数名")
        p.add_argument("--lon-var", default="longitude", help="経度変数名")
        p.add_argument("--no-dask", action="store_true", help="Daskによる遅延・チャンク処理を無効化する")
    for p in (plot, both):
        if p is plot:
            p.add_argument("input", type=Path, help="勾配を保存したNetCDF")
            p.add_argument("--ssh-var", default="ssh_karin_2", help="SSH変数名")
            p.add_argument("--lat-var", default="latitude", help="緯度変数名")
            p.add_argument("--lon-var", default="longitude", help="経度変数名")
        p.add_argument("--output-png", type=Path, default=Path("swot_gradients_map.png"), help="出力PNGのパス")
        p.add_argument("--percentile", type=float, default=98.0, help="勾配の色範囲を決める絶対値パーセンタイル")
        p.add_argument("--dpi", type=int, default=220, help="出力PNGの解像度（dpi）")
    return parser


def main() -> None:
    """コマンドライン指定に従って計算処理と描画処理を実行する。"""
    args = _parser().parse_args()
    if args.command in ("compute", "all"):
        nc = compute_gradients(
            args.input,
            args.output_nc,
            args.ssh_var,
            args.lat_var,
            args.lon_var,
            chunks=None if args.no_dask else "auto",
        )
        print(f"NetCDFを保存しました: {nc}")
    else:
        nc = args.input
    if args.command in ("plot", "all"):
        png = plot_gradients(
            nc,
            args.output_png,
            args.ssh_var,
            args.lat_var,
            args.lon_var,
            args.percentile,
            args.dpi,
        )
        print(f"PNGを保存しました: {png}")


if __name__ == "__main__":
    main()
