# SWOT native-grid SSH gradients

This tool computes three directional sea-surface slopes directly on a SWOT
`L2_LR_SSH` swath without resampling, interpolation, smoothing, or decimation.

## Install and run

```powershell
python -m pip install -r requirements.txt
python swot_ssh_gradients.py all INPUT_SWOT.nc `
  --output-nc swot_ssh_gradients_2km.nc `
  --output-png swot_gradients_map.png
```

If the SSH variable differs from the default, add `--ssh-var VARIABLE_NAME`.
The default is `ssh_karin_2`. Use the `compute` and `plot` subcommands to run
the two stages separately. Add `--no-dask` when Dask is unavailable or the
whole pass comfortably fits in memory.

## Difference convention

At grid cell `(i,j)` the three slopes are

* along: `[SSH(i+1,j) - SSH(i,j)] / d[(i,j),(i+1,j)]`
* cross: `[SSH(i,j+1) - SSH(i,j)] / d[(i,j),(i,j+1)]`
* oblique: `[SSH(i+1,j+1) - SSH(i,j)] / d[(i,j),(i+1,j+1)]`

Here `d` is the WGS84 inverse-geodesic distance calculated from the native
2-D latitude/longitude arrays. Thus the oblique denominator is close to
sqrt(2) x 2 km but uses the actual geometry. At a terminal array edge, the
corresponding backward adjacent difference is used. The two geometrically
unpaired oblique corners remain missing. Any difference whose SSH or
geolocation endpoint is missing also remains missing, so gaps are not bridged.

The output retains the native SSH dimensions and coordinate variables. Slopes
are stored as dimensionless `m m-1` (`m/m`) variables with compression and
CF-style descriptive metadata.
