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

勾配の計算本体は ``swot_ssh_gradients.py`` を呼び出す。元の2 kmスワス
グリッドを維持し、再グリッド化、平滑化、間引き、欠損値補間は行わない。
Along、Cross、Oblique（正の45度）の各勾配は、隣接SSH差をWGS84楕円体上の
実距離で割って求める。結合ファイルに ``source_file_index`` が存在する場合、
異なる元ファイルの境界を跨ぐAlong・Oblique差分は計算しない。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from swot_ssh_gradients import compute_gradients, plot_gradients


# Kuwabara環境における既定の入出力先。
DATA_DIRECTORY = Path("/home/kuwabara/SwotData/Japan")
DEFAULT_INPUT_NC = DATA_DIRECTORY / "JapanSSHdata002.nc"
DEFAULT_OUTPUT_NC = DATA_DIRECTORY / "gradient_sshJp1.nc"
DEFAULT_OUTPUT_PNG = DATA_DIRECTORY / "gradient_sshJp1.png"


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
    return parser


def main() -> None:
    """既定値またはコマンドライン指定に従って勾配計算と描画を実行する。"""
    args = _parser().parse_args()
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
