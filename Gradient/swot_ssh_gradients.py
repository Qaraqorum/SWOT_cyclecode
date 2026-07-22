#!/usr/bin/env python3
"""Compute native-grid SWOT L2_LR_SSH slopes and plot the result.

No resampling, smoothing, interpolation, or decimation is performed.  Each
slope is an SSH difference between immediately adjacent native pixels divided
by their WGS84 ellipsoidal distance.
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
    """Vectorized WGS84 geodesic distance; NaN endpoints remain NaN."""
    _, _, distance = WGS84.inv(lon0, lat0, lon1, lat1)
    return np.asarray(distance, dtype=np.float64)


def _geodesic_distance(
    lon0: xr.DataArray,
    lat0: xr.DataArray,
    lon1: xr.DataArray,
    lat1: xr.DataArray,
) -> xr.DataArray:
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
) -> xr.DataArray:
    """Adjacent forward difference, with backward difference only at edges."""
    ssh_next = ssh.shift(shifts)
    lat_next = latitude.shift(shifts)
    lon_next = longitude.shift(shifts)
    d_forward = _geodesic_distance(longitude, latitude, lon_next, lat_next)
    forward = (ssh_next - ssh) / d_forward

    reverse_shifts = {dim: -step for dim, step in shifts.items()}
    ssh_prev = ssh.shift(reverse_shifts)
    lat_prev = latitude.shift(reverse_shifts)
    lon_prev = longitude.shift(reverse_shifts)
    d_backward = _geodesic_distance(lon_prev, lat_prev, longitude, latitude)
    backward = (ssh - ssh_prev) / d_backward

    # xarray.shift uses: output[i] = input[i - shift].  A shift of -1 is the
    # forward neighbour.  Only the terminal edge(s) use the backward segment.
    edge = xr.zeros_like(ssh, dtype=bool)
    for dim, step in shifts.items():
        if step != -1:
            raise ValueError("This implementation expects forward shifts of -1")
        idx = xr.DataArray(
            np.arange(ssh.sizes[dim]), dims=(dim,), coords={dim: ssh[dim]}
        )
        edge = edge | (idx == ssh.sizes[dim] - 1)

    result = xr.where(edge, backward, forward)
    # Protect against zero/invalid geometry and preserve all SSH/geolocation gaps.
    return result.where(np.isfinite(result))


def _pick_spatial_dims(ssh: xr.DataArray) -> tuple[str, str]:
    preferred = ("num_lines", "num_pixels")
    if all(dim in ssh.dims for dim in preferred):
        return preferred
    if ssh.ndim == 2:
        return ssh.dims[0], ssh.dims[1]
    raise ValueError(
        f"SSH variable must be 2-D; got dimensions {ssh.dims}. "
        "Select/squeeze any non-spatial dimension first."
    )


def compute_gradients(
    input_nc: str | Path,
    output_nc: str | Path = "swot_ssh_gradients_2km.nc",
    ssh_name: str = "ssh_karin_2",
    latitude_name: str = "latitude",
    longitude_name: str = "longitude",
    chunks: str | int | None = "auto",
) -> Path:
    """Compute three native-swath slopes and write one self-contained NetCDF."""
    input_nc, output_nc = Path(input_nc), Path(output_nc)
    ds = xr.open_dataset(input_nc, decode_cf=True, mask_and_scale=True, chunks=chunks)
    for name in (ssh_name, latitude_name, longitude_name):
        if name not in ds:
            raise KeyError(f"Required variable {name!r} is absent from {input_nc}")

    ssh = ds[ssh_name].astype(np.float64)
    lat = ds[latitude_name].astype(np.float64)
    lon = ds[longitude_name].astype(np.float64)
    along_dim, cross_dim = _pick_spatial_dims(ssh)
    if lat.dims != ssh.dims or lon.dims != ssh.dims:
        raise ValueError("latitude, longitude, and SSH must share the same 2-D grid")

    along = _directional_slope(ssh, lat, lon, {along_dim: -1})
    cross = _directional_slope(ssh, lat, lon, {cross_dim: -1})
    # +45-degree grid diagonal: (line, pixel) -> (line+1, pixel+1).
    oblique = _directional_slope(
        ssh, lat, lon, {along_dim: -1, cross_dim: -1}
    )

    # Keep dimensional coordinates plus the full 2-D geolocation and line time.
    keep = {ssh_name, latitude_name, longitude_name}
    keep.update(ds.coords)
    if "time" in ds:
        keep.add("time")
    out = ds[[name for name in ds.variables if name in keep]].copy()
    out[ssh_name] = ds[ssh_name]  # retain the source dtype/attributes
    out["slope_along"] = along.astype(np.float32)
    out["slope_cross"] = cross.astype(np.float32)
    out["slope_oblique"] = oblique.astype(np.float32)

    common = {
        "units": "m m-1",
        "coordinates": f"{latitude_name} {longitude_name}",
        "source_ssh": ssh_name,
        "distance_method": "WGS84 ellipsoidal inverse geodesic",
        "difference_method": "adjacent-pixel forward difference; backward at terminal edge",
        "resolution_note": "native grid retained; no resampling, smoothing, or interpolation",
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
    """Read the saved NetCDF and render every native cell in four map panels."""
    input_nc, output_png = Path(input_nc), Path(output_png)
    with xr.open_dataset(input_nc, mask_and_scale=True) as ds:
        lat = ds[latitude_name].values
        lon = ds[longitude_name].values
        ssh = ds[ssh_name].values
        slopes = [ds[n].values for n in ("slope_along", "slope_cross", "slope_oblique")]

    cmap_ssh = plt.colormaps["viridis"].copy()
    cmap_slope = plt.colormaps["RdBu_r"].copy()
    cmap_ssh.set_bad("white")
    cmap_slope.set_bad("white")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

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
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    compute = sub.add_parser("compute", help="compute slopes and save NetCDF")
    both = sub.add_parser("all", help="compute NetCDF and then create PNG")
    plot = sub.add_parser("plot", help="plot an already generated NetCDF")
    for p in (compute, both):
        p.add_argument("input", type=Path, help="input SWOT L2_LR_SSH NetCDF")
        p.add_argument("--output-nc", type=Path, default=Path("swot_ssh_gradients_2km.nc"))
        p.add_argument("--ssh-var", default="ssh_karin_2")
        p.add_argument("--lat-var", default="latitude")
        p.add_argument("--lon-var", default="longitude")
        p.add_argument("--no-dask", action="store_true", help="disable lazy chunked processing")
    for p in (plot, both):
        if p is plot:
            p.add_argument("input", type=Path, help="gradient NetCDF")
            p.add_argument("--ssh-var", default="ssh_karin_2")
            p.add_argument("--lat-var", default="latitude")
            p.add_argument("--lon-var", default="longitude")
        p.add_argument("--output-png", type=Path, default=Path("swot_gradients_map.png"))
        p.add_argument("--percentile", type=float, default=98.0)
        p.add_argument("--dpi", type=int, default=220)
    return parser


def main() -> None:
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
        print(f"Wrote {nc}")
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
        print(f"Wrote {png}")


if __name__ == "__main__":
    main()
