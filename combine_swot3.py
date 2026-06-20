import xarray as xr
import glob
import os
#データ数削減のために一部データに絞った結合

data_dir = '/home/kuwabara/SwotData/' 

file_pattern = os.path.join(data_dir, '*.nc')
file_paths = glob.glob(file_pattern)
file_paths.sort()

print(f"{len(file_paths)} 個のファイルが見つかりました。破損チェックを開始します...")

valid_files = []
for f in file_paths:
    try:
        with xr.open_dataset(f, engine='netcdf4') as ds:
            pass
        valid_files.append(f)
    except Exception as e:
        pass # エラーの詳細は省略

print(f"\n正常なファイル {len(valid_files)} 個の結合を開始します...")

try:
    if len(valid_files) > 0:
        ds_combined = xr.open_mfdataset(
            valid_files, 
            combine='nested', 
            concat_dim='num_lines', 
            engine='netcdf4'
        )
        
        # 🌟 追加：必要な変数だけをリストで指定して抽出する（緯度・経度・時間は自動でついてきます）
        # Panoplyで見たい変数をここに書きます。
        target_vars = ['ssh_karin', 'ssha_karin'] 
        ds_combined = ds_combined[target_vars]
        
        print("必要な変数の抽出が完了しました。ファイルに書き出します（これには少し時間がかかります）...")

        output_filename = 'SWOT_cycle_combined_light.nc'
        ds_combined.to_netcdf(output_filename)
        
        print(f"🎉 成功: {output_filename} として保存されました！")
    else:
        print("正常なファイルがありませんでした。")

except Exception as e:
    print(f"結合中にエラーが発生しました: {e}")