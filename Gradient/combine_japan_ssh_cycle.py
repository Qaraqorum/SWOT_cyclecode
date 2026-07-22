#!/usr/bin/env python3
"""SWOTの1サイクル分のSSHファイルから日本近海だけを抽出・結合する。

既定では ``/home/kuwabara/SwotData/Japan/Cycle_002`` にあるNetCDFを
ファイル名順に処理し、北緯20～50度・東経110～160度の ``ssh_karin_2`` を
``/home/kuwabara/SwotData/Japan/JapanSSHdata002.nc`` に保存する。

再グリッド化、補間、平滑化、間引きは行わない。対象海域と交差する沿軌道
ラインだけを残し、そのライン内で範囲外にあるピクセルはNaNにする。各入力
ファイルを1つずつ読み書きするため、約600ファイルでも全データを同時に
メモリへ展開しない。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import netCDF4
import numpy as np
import xarray as xr


# Kuwabara環境におけるCycle 002の既定入出力パス。
DEFAULT_INPUT_DIR = Path("/home/kuwabara/SwotData/Japan/Cycle_002")
DEFAULT_OUTPUT_FILE = Path("/home/kuwabara/SwotData/Japan/JapanSSHdata002.nc")

# 日本近海の既定範囲。日本は西経ではなく東経110～160度に位置する。
DEFAULT_LAT_MIN = 20.0
DEFAULT_LAT_MAX = 50.0
DEFAULT_LON_MIN = 110.0
DEFAULT_LON_MAX = 160.0


def _spatial_dims(ssh: xr.DataArray) -> tuple[str, str]:
    """SSHから沿軌道方向と横軌道方向の次元名を決定する。"""
    if "num_lines" in ssh.dims and "num_pixels" in ssh.dims:
        return "num_lines", "num_pixels"
    if ssh.ndim == 2:
        return ssh.dims[0], ssh.dims[1]
    raise ValueError(
        f"SSH変数は2次元である必要があります。入力次元: {ssh.dims}"
    )


def _as_float_array(data: xr.DataArray) -> np.ndarray:
    """DataArrayをNaN表現可能なfloat64配列として取り出す。"""
    values = data.values
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.asarray(values, dtype=np.float64)


def _longitude_mask(
    longitude: np.ndarray, lon_min: float, lon_max: float
) -> np.ndarray:
    """経度表現を[-180, 180)へ統一し、指定範囲のマスクを返す。

    ``lon_min > lon_max`` の場合は、日付変更線を跨ぐ範囲として処理する。
    例えば170度～-170度という指定にも対応できる。
    """
    lon180 = (longitude + 180.0) % 360.0 - 180.0
    west = (lon_min + 180.0) % 360.0 - 180.0
    east = (lon_max + 180.0) % 360.0 - 180.0
    if west <= east:
        return (lon180 >= west) & (lon180 <= east)
    return (lon180 >= west) | (lon180 <= east)


def _copyable_attrs(attrs: dict) -> dict:
    """デコード済み値と重複するpacking属性を除いて属性をコピーする。"""
    excluded = {
        "_FillValue",
        "missing_value",
        "scale_factor",
        "add_offset",
        "_Unsigned",
    }
    return {key: value for key, value in attrs.items() if key not in excluded}


def _create_float_variable(
    output: netCDF4.Dataset,
    name: str,
    dimensions: tuple[str, ...],
    source: xr.DataArray,
    compression_level: int,
) -> netCDF4.Variable:
    """圧縮されたfloat64変数を作り、入力変数の説明属性を引き継ぐ。"""
    variable = output.createVariable(
        name,
        "f8",
        dimensions,
        fill_value=np.nan,
        zlib=True,
        complevel=compression_level,
        shuffle=True,
    )
    variable.setncatts(_copyable_attrs(source.attrs))
    return variable


def _initialize_output(
    output_path: Path,
    template: xr.Dataset,
    ssh_name: str,
    latitude_name: str,
    longitude_name: str,
    time_name: str,
    along_dim: str,
    cross_dim: str,
    compression_level: int,
) -> tuple[netCDF4.Dataset, dict[str, netCDF4.Variable]]:
    """最初の有効入力をテンプレートとして出力NetCDFを初期化する。"""
    output = netCDF4.Dataset(output_path, "w", format="NETCDF4")
    output.createDimension(along_dim, None)  # ファイルごとに追記する無制限次元
    output.createDimension(cross_dim, template.sizes[cross_dim])

    variables: dict[str, netCDF4.Variable] = {}
    variables[along_dim] = output.createVariable(along_dim, "i8", (along_dim,))
    variables[along_dim].long_name = "concatenated along-track line index"
    variables[along_dim].comment = (
        "0-based continuous index after concatenating the selected source lines"
    )

    variables[cross_dim] = output.createVariable(cross_dim, "i4", (cross_dim,))
    variables[cross_dim].long_name = "native cross-track pixel index"
    if cross_dim in template.coords and template[cross_dim].dims == (cross_dim,):
        cross_values = np.asarray(template[cross_dim].values)
        if np.issubdtype(cross_values.dtype, np.number):
            variables[cross_dim][:] = cross_values.astype(np.int32)
        else:
            variables[cross_dim][:] = np.arange(template.sizes[cross_dim])
    else:
        variables[cross_dim][:] = np.arange(template.sizes[cross_dim])

    variables[latitude_name] = _create_float_variable(
        output,
        latitude_name,
        (along_dim, cross_dim),
        template[latitude_name],
        compression_level,
    )
    variables[longitude_name] = _create_float_variable(
        output,
        longitude_name,
        (along_dim, cross_dim),
        template[longitude_name],
        compression_level,
    )
    variables[ssh_name] = _create_float_variable(
        output,
        ssh_name,
        (along_dim, cross_dim),
        template[ssh_name],
        compression_level,
    )

    # 元ファイルと元ライン番号を追跡できるよう、結合後の各ラインへ索引を付ける。
    variables["source_file_index"] = output.createVariable(
        "source_file_index",
        "i4",
        (along_dim,),
        fill_value=np.int32(-1),
        zlib=True,
        complevel=compression_level,
        shuffle=True,
    )
    variables["source_file_index"].long_name = "index of source NetCDF file"
    variables["source_line_index"] = output.createVariable(
        "source_line_index",
        "i4",
        (along_dim,),
        fill_value=np.int32(-1),
        zlib=True,
        complevel=compression_level,
        shuffle=True,
    )
    variables["source_line_index"].long_name = (
        "original along-track line index within the source file"
    )

    if time_name in template:
        time = template[time_name]
        if time.dims == (along_dim,):
            variables[time_name] = _create_float_variable(
                output,
                time_name,
                (along_dim,),
                time,
                compression_level,
            )
        else:
            print(
                f"警告: {time_name!r} の次元 {time.dims} が想定外のため保存しません",
                file=sys.stderr,
            )

    return output, variables


def combine_cycle_region(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_file: str | Path = DEFAULT_OUTPUT_FILE,
    pattern: str = "*.nc",
    ssh_name: str = "ssh_karin_2",
    latitude_name: str = "latitude",
    longitude_name: str = "longitude",
    time_name: str = "time",
    lat_min: float = DEFAULT_LAT_MIN,
    lat_max: float = DEFAULT_LAT_MAX,
    lon_min: float = DEFAULT_LON_MIN,
    lon_max: float = DEFAULT_LON_MAX,
    cycle_number: str = "002",
    compression_level: int = 4,
    recursive: bool = False,
    strict: bool = False,
    overwrite: bool = False,
) -> Path:
    """1サイクル分のSWOTファイルを日本近海に限定して1ファイルへ結合する。

    ファイルごとに緯度・経度マスクを作り、対象海域を1点以上含む沿軌道
    ラインだけを追記する。同じラインの範囲外ピクセルはSSH・緯度・経度を
    NaNにするため、矩形外の観測が出力へ混入しない。

    ``strict=False`` では必須変数を持たないNetCDFを警告付きでスキップする。
    Basic・Expert等が混在する場合は、``--pattern`` で対象製品を限定することを
    推奨する。
    """
    input_dir = Path(input_dir).expanduser()
    output_file = Path(output_file).expanduser()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"入力ディレクトリがありません: {input_dir}")
    if lat_min > lat_max:
        raise ValueError("lat-minはlat-max以下にしてください")
    if not 0 <= compression_level <= 9:
        raise ValueError("compression-levelは0～9で指定してください")

    iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
    output_resolved = output_file.resolve()
    temporary_file = output_file.with_suffix(output_file.suffix + ".part")
    excluded_paths = {output_resolved, temporary_file.resolve()}
    files = sorted(
        path for path in iterator if path.is_file() and path.resolve() not in excluded_paths
    )
    if not files:
        raise FileNotFoundError(
            f"{input_dir} にパターン {pattern!r} と一致するファイルがありません"
        )
    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"出力先が既に存在します: {output_file}。上書きする場合は --overwrite を指定してください"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if temporary_file.exists():
        temporary_file.unlink()

    output: netCDF4.Dataset | None = None
    output_variables: dict[str, netCDF4.Variable] = {}
    output_along_dim: str | None = None
    output_cross_dim: str | None = None
    total_lines = 0
    used_files: list[str] = []
    skipped_missing = 0
    skipped_outside = 0

    try:
        for file_number, path in enumerate(files, start=1):
            if file_number == 1 or file_number % 25 == 0 or file_number == len(files):
                print(f"[{file_number}/{len(files)}] {path.name}")

            writing_started = False
            try:
                # 時刻は数値表現のまま読み、元のunits/calendar属性を保持する。
                with xr.open_dataset(
                    path,
                    decode_cf=True,
                    decode_times=False,
                    mask_and_scale=True,
                    chunks=None,
                ) as dataset:
                    required = (ssh_name, latitude_name, longitude_name)
                    missing = [name for name in required if name not in dataset]
                    if missing:
                        message = f"{path.name}: 必須変数がありません: {missing}"
                        if strict:
                            raise KeyError(message)
                        print(f"警告: {message}（スキップ）", file=sys.stderr)
                        skipped_missing += 1
                        continue

                    ssh = dataset[ssh_name]
                    along_dim, cross_dim = _spatial_dims(ssh)
                    if (
                        dataset[latitude_name].dims != ssh.dims
                        or dataset[longitude_name].dims != ssh.dims
                    ):
                        raise ValueError(
                            f"{path.name}: SSH・緯度・経度の次元が一致しません"
                        )
                    ssh = ssh.transpose(along_dim, cross_dim)
                    latitude = dataset[latitude_name].transpose(along_dim, cross_dim)
                    longitude = dataset[longitude_name].transpose(along_dim, cross_dim)

                    lat_values = _as_float_array(latitude)
                    lon_values = _as_float_array(longitude)
                    region_mask = (
                        np.isfinite(lat_values)
                        & np.isfinite(lon_values)
                        & (lat_values >= lat_min)
                        & (lat_values <= lat_max)
                        & _longitude_mask(lon_values, lon_min, lon_max)
                    )
                    selected_lines = np.flatnonzero(region_mask.any(axis=1))
                    if selected_lines.size == 0:
                        skipped_outside += 1
                        continue

                    if output is None:
                        # ここから先の失敗は出力ファイルの整合性に関わるため、
                        # 不適合入力の単純スキップではなく全処理を停止する。
                        writing_started = True
                        output_along_dim, output_cross_dim = along_dim, cross_dim
                        output, output_variables = _initialize_output(
                            temporary_file,
                            dataset,
                            ssh_name,
                            latitude_name,
                            longitude_name,
                            time_name,
                            along_dim,
                            cross_dim,
                            compression_level,
                        )
                    elif along_dim != output_along_dim or cross_dim != output_cross_dim:
                        raise ValueError(
                            f"{path.name}: 空間次元 {(along_dim, cross_dim)} が"
                            f"先頭ファイル {(output_along_dim, output_cross_dim)} と一致しません"
                        )
                    elif dataset.sizes[cross_dim] != output.dimensions[cross_dim].size:
                        raise ValueError(
                            f"{path.name}: {cross_dim} のサイズが先頭ファイルと一致しません"
                        )

                    block_mask = region_mask[selected_lines, :]
                    ssh_block = _as_float_array(ssh.isel({along_dim: selected_lines}))
                    lat_block = lat_values[selected_lines, :].copy()
                    lon_block = lon_values[selected_lines, :].copy()
                    ssh_block[~block_mask] = np.nan
                    lat_block[~block_mask] = np.nan
                    lon_block[~block_mask] = np.nan

                    writing_started = True
                    start = total_lines
                    stop = start + selected_lines.size
                    output_variables[along_dim][start:stop] = np.arange(start, stop)
                    output_variables[latitude_name][start:stop, :] = lat_block
                    output_variables[longitude_name][start:stop, :] = lon_block
                    output_variables[ssh_name][start:stop, :] = ssh_block
                    output_variables["source_file_index"][start:stop] = len(used_files)
                    output_variables["source_line_index"][start:stop] = selected_lines

                    if time_name in output_variables:
                        if time_name not in dataset or dataset[time_name].dims != (along_dim,):
                            raise ValueError(
                                f"{path.name}: {time_name!r} がないか、次元が想定と異なります"
                            )
                        time_values = _as_float_array(
                            dataset[time_name].isel({along_dim: selected_lines})
                        )
                        output_variables[time_name][start:stop] = time_values

                    total_lines = stop
                    used_files.append(path.name)
            except Exception as error:
                if strict or writing_started:
                    raise
                print(f"警告: {path.name} の処理に失敗しました: {error}（スキップ）", file=sys.stderr)
                skipped_missing += 1

        if output is None or total_lines == 0:
            raise RuntimeError("指定範囲に該当する観測は見つかりませんでした")

        # 複数入力に共通しない元の軌道属性はコピーせず、結合処理の履歴を明示する。
        output.setncatts(
            {
                "title": "SWOT SSH observations around Japan for cycle " + cycle_number,
                "summary": (
                    "Native SWOT swath observations clipped without resampling, "
                    "smoothing, interpolation, or decimation"
                ),
                "Conventions": "CF-1.8, ACDD-1.3",
                "cycle_number": cycle_number,
                "geospatial_lat_min": np.float64(lat_min),
                "geospatial_lat_max": np.float64(lat_max),
                "geospatial_lon_min": np.float64(lon_min),
                "geospatial_lon_max": np.float64(lon_max),
                "geospatial_lat_units": "degrees_north",
                "geospatial_lon_units": "degrees_east",
                "source_directory": str(input_dir),
                "source_file_pattern": pattern,
                "source_file_count": np.int32(len(used_files)),
                "source_files": "\n".join(used_files),
                "input_files_examined": np.int32(len(files)),
                "input_files_outside_region": np.int32(skipped_outside),
                "input_files_skipped": np.int32(skipped_missing),
                "processing_note": (
                    "Only native along-track lines intersecting the bounding box are retained; "
                    "pixels outside the box are set to missing"
                ),
                "history": (
                    f"{datetime.now(timezone.utc).isoformat()}: created by "
                    "combine_japan_ssh_cycle.py"
                ),
            }
        )
        output.close()
        output = None
        temporary_file.replace(output_file)
    except Exception:
        if output is not None:
            output.close()
        if temporary_file.exists():
            temporary_file.unlink()
        raise

    print(f"使用ファイル数: {len(used_files)} / {len(files)}")
    print(f"結合後の沿軌道ライン数: {total_lines}")
    print(f"保存しました: {output_file}")
    return output_file


def _parser() -> argparse.ArgumentParser:
    """コマンドライン引数を定義する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Cycle_002ディレクトリ",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="結合後のNetCDF",
    )
    parser.add_argument("--pattern", default="*.nc", help="入力ファイルのglobパターン")
    parser.add_argument("--ssh-var", default="ssh_karin_2", help="SSH変数名")
    parser.add_argument("--lat-var", default="latitude", help="緯度変数名")
    parser.add_argument("--lon-var", default="longitude", help="経度変数名")
    parser.add_argument("--time-var", default="time", help="時刻変数名")
    parser.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN, help="南端緯度")
    parser.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX, help="北端緯度")
    parser.add_argument("--lon-min", type=float, default=DEFAULT_LON_MIN, help="西端経度（東経を正）")
    parser.add_argument("--lon-max", type=float, default=DEFAULT_LON_MAX, help="東端経度（東経を正）")
    parser.add_argument("--cycle", default="002", help="出力メタデータに記録するサイクル番号")
    parser.add_argument("--compression-level", type=int, default=4, help="NetCDF圧縮レベル（0～9）")
    parser.add_argument("--recursive", action="store_true", help="サブディレクトリも再帰検索する")
    parser.add_argument("--strict", action="store_true", help="不適合ファイルをスキップせず停止する")
    parser.add_argument("--overwrite", action="store_true", help="既存の出力ファイルを上書きする")
    return parser


def main() -> None:
    """指定されたCycleディレクトリを地域抽出・結合する。"""
    args = _parser().parse_args()
    combine_cycle_region(
        input_dir=args.input_dir,
        output_file=args.output,
        pattern=args.pattern,
        ssh_name=args.ssh_var,
        latitude_name=args.lat_var,
        longitude_name=args.lon_var,
        time_name=args.time_var,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        cycle_number=args.cycle,
        compression_level=args.compression_level,
        recursive=args.recursive,
        strict=args.strict,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
