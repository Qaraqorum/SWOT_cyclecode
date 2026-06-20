import xarray as xr
import glob
import os

# パスはご自身の環境に合わせてください
data_dir = '/home/kuwabara/SwotData/' 

file_pattern = os.path.join(data_dir, '*.nc')
file_paths = glob.glob(file_pattern)
file_paths.sort() # 時間順（パス順）に並べ替えてから結合します

print(f"{len(file_paths)} 個のファイルが見つかりました。破損チェックを開始します...")

valid_files = []
for f in file_paths:
    try:
        with xr.open_dataset(f, engine='netcdf4') as ds:
            pass
        valid_files.append(f)
    except Exception as e:
        print(f"⚠️ スキップ: {os.path.basename(f)}")

print(f"\n正常なファイル {len(valid_files)} 個の結合を開始します...")

try:
    if len(valid_files) > 0:
        # 🌟 変更点：concat_dim を 'num_lines' に指定して縦に連結します
        ds_combined = xr.open_mfdataset(
            valid_files, 
            combine='nested', 
            concat_dim='num_lines', 
            engine='netcdf4'
        )
        
        output_filename = 'SWOT_cycle_combined.nc'
        ds_combined.to_netcdf(output_filename)
        
        print(f"🎉 成功: {output_filename} として保存されました！")
    else:
        print("正常なファイルがありませんでした。")

except Exception as e:
    print(f"結合中にエラーが発生しました: {e}")