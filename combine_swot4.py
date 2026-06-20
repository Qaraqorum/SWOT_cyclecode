import xarray as xr
import glob
import os
from tqdm import tqdm  # 🌟追加：ファイルチェック用の進捗バー
from dask.diagnostics import ProgressBar  # 🌟追加：xarray(Dask)保存用の進捗バー

data_dir = '/home/kuwabara/SwotData/' 

file_pattern = os.path.join(data_dir, '*.nc')
file_paths = glob.glob(file_pattern)
file_paths.sort()

print(f"{len(file_paths)} 個のファイルが見つかりました。")

valid_files = []

# 🌟変更：tqdm(...) で囲むだけで、自動的に進捗バーと残り時間が表示されます！
for f in tqdm(file_paths, desc="破損チェック進捗"):
    try:
        with xr.open_dataset(f, engine='netcdf4') as ds:
            pass
        valid_files.append(f)
    except Exception:
        pass

print(f"\n正常なファイル {len(valid_files)} 個の結合を開始します...")

try:
    if len(valid_files) > 0:
        # ここは「仮想的な読み込み」なので一瞬で終わります
        ds_combined = xr.open_mfdataset(
            valid_files, 
            combine='nested', 
            concat_dim='num_lines', 
            engine='netcdf4'
        )
        
        # 必要な変数だけを抽出（軽くする）
        target_vars = ['ssh_karin', 'ssha_karin'] 
        ds_combined = ds_combined[target_vars]
        
        output_filename = 'SWOT_cycle_combined_light.nc'
        print(f"\n大元の結合データを {output_filename} に書き出します。")
        print("（ここからが本番です。メモリとCPUを使って計算します）")

        # 🌟変更：with ProgressBar(): で囲むと、xarrayの重い保存処理のETAが出ます！
        with ProgressBar():
            ds_combined.to_netcdf(output_filename)
        
        print(f"\n🎉 成功: {output_filename} として保存されました！")
    else:
        print("正常なファイルがありませんでした。")

except Exception as e:
    print(f"\n結合中にエラーが発生しました: {e}")