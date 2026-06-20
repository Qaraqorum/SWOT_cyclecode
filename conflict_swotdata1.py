import xarray as xr

# 確認したいファイルを1つ指定（どれでもOKです）
file_path = '/home/kuwabara/SwotData/SWOT_L2_LR_SSH_Expert_001_411_20230804T210804_20230804T215833_PGC0_01.nc'

# データを読み込む
ds = xr.open_dataset(file_path)

# 中身（構造）を表示
print(ds)