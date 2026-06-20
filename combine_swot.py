import xarray as xr
import glob
import os

# 1. 結合したい .nc ファイルが入っているフォルダのパスを指定します
# （ご自身の環境に合わせてパスを書き換えてください）
data_dir = '/home/kuwabara/SwotData/' 

# フォルダ内のすべての .nc ファイルのパスをリストとして取得
file_pattern = os.path.join(data_dir, '*.nc')
file_paths = glob.glob(file_pattern)

print(f"{len(file_paths)} 個のファイルが見つかりました。結合を開始します...")

# 2. xarrayの open_mfdataset を使って複数ファイルを一括読み込み
# daskを使って遅延評価（メモリ節約）しながら読み込みます
try:
    # SWOTのパスデータは時間に沿って結合するのが一般的なため、concat_dim='time' としています。
    # データ構造によっては combine='by_coords' のみが適している場合もあります。
    ds_combined = xr.open_mfdataset(file_paths, combine='nested', concat_dim='time')
    
    print("データの結合（仮想的な読み込み）が完了しました。ファイルに書き出します...")

    # 3. 結合したデータを1つの新しい .nc ファイルとして出力
    output_filename = 'SWOT_cycle_combined.nc'
    ds_combined.to_netcdf(output_filename)
    
    print(f"成功: {output_filename} として保存されました！")

except Exception as e:
    print(f"エラーが発生しました: {e}")