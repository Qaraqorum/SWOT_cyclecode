# SWOTネイティブグリッド海面勾配計算

## 概要

`swot_ssh_gradients.py` は、SWOT衛星の `L2_LR_SSH` などの2 kmグリッド
海面高度（SSH）データから、スワス座標系における次の3方向の海面勾配を
計算するPythonスクリプトです。

- Along-track：軌道方向
- Cross-track：軌道直交方向
- Oblique-track：軌道グリッド上の正の45度対角方向

元の `num_lines × num_pixels` グリッドをそのまま使用し、リサンプリング、
平滑化、間引き、欠損値の空間内挿は行いません。計算結果は元のSSHおよび
座標情報とともに1つのNetCDFへ保存し、そのNetCDFから確認用PNGを作成します。

## 主な特徴

- 元の2 kmグリッド形状と解像度を完全に維持
- NumPy・Xarray・Daskによるベクトル化処理
- WGS84楕円体上の実測地距離を各隣接ピクセルについて計算
- 海面中央ギャップ、陸域、無効観測を補間せずNaNのまま保持
- NetCDF-4圧縮、CF形式に沿った変数属性・処理履歴を付与
- SSH、Along、Cross、Obliqueの4パネルPNGを出力

## 必要なPythonパッケージ

```powershell
python -m pip install -r requirements.txt
```

主な依存パッケージは `xarray`、`numpy`、`netCDF4`、`pyproj`、
`matplotlib`、`dask[array]` です。

## 基本的な実行方法

NetCDFの作成とPNGの描画を一括実行する場合：

```powershell
python swot_ssh_gradients.py all INPUT_SWOT.nc `
  --output-nc swot_ssh_gradients_2km.nc `
  --output-png swot_gradients_map.png
```

勾配計算とNetCDF保存だけを実行する場合：

```powershell
python swot_ssh_gradients.py compute INPUT_SWOT.nc `
  --output-nc swot_ssh_gradients_2km.nc
```

保存済みNetCDFからPNGだけを作成する場合：

```powershell
python swot_ssh_gradients.py plot swot_ssh_gradients_2km.nc `
  --output-png swot_gradients_map.png
```

SSH変数名の既定値は `ssh_karin_2` です。製品内の変数名が異なる場合は、
次のように指定します。

```powershell
python swot_ssh_gradients.py all INPUT_SWOT.nc `
  --ssh-var SSH_VARIABLE_NAME
```

緯度・経度変数名が異なる場合は、`--lat-var` と `--lon-var` で指定できます。
Daskを使用しない場合は `--no-dask` を追加してください。

## 勾配の計算方法

グリッド点 `(i, j)` における通常の前方隣接差分は次のとおりです。

### Along-track

```text
slope_along(i,j)
  = [SSH(i+1,j) - SSH(i,j)] / d[(i,j), (i+1,j)]
```

### Cross-track

```text
slope_cross(i,j)
  = [SSH(i,j+1) - SSH(i,j)] / d[(i,j), (i,j+1)]
```

### Oblique-track（正の45度対角方向）

```text
slope_oblique(i,j)
  = [SSH(i+1,j+1) - SSH(i,j)] / d[(i,j), (i+1,j+1)]
```

ここで `d` は、2次元の `latitude`・`longitude` から `pyproj.Geod` を用いて
計算したWGS84楕円体上の逆測地距離です。そのため、斜め方向の距離は概ね
`sqrt(2) × 2 km ≒ 2.828 km` ですが、固定値2828 mではなく各ピクセルの
実際の配置を反映します。

勾配の符号は、正方向の隣接点のSSHから現在点のSSHを引いた値を正とします。
配列の終端では、出力形状を維持するため同一方向の後方隣接差分を使用します。
対角線上に対応する隣接点を持たないOblique方向の2つの角はNaNになります。

## 欠損値の扱い

差分に使う2点のうち、SSH・緯度・経度のいずれかが欠損している場合、対応する
勾配だけをNaNにします。欠損域を跨ぐ補間や遠方ピクセルへの置換は行わないため、
スワス中央ギャップや陸域の影響が不必要に広がりません。

## NetCDF出力変数

| 変数 | 内容 | 単位 |
|---|---|---|
| `ssh_karin_2` | 入力元の海面高度 | m（入力属性を維持） |
| `slope_along` | 軌道方向の海面勾配 | `m m-1` |
| `slope_cross` | 軌道直交方向の海面勾配 | `m m-1` |
| `slope_oblique` | 正の45度対角方向の海面勾配 | `m m-1` |

入力に含まれる次元座標、`latitude`、`longitude`、`time` も可能な限り保持します。
各勾配変数には方向、差分方法、距離計算方法、元SSH変数名、無平滑・無補間で
あることを示す属性を付与します。

## PNG可視化

PNGには元のSSHと3方向の勾配を2 × 2パネルで表示します。欠損値は白抜き、
勾配は発散型カラーマップ `RdBu_r` を使用します。既定の表示範囲は勾配絶対値の
98パーセンタイルで決定され、`--percentile` で変更できます。この範囲設定は
表示だけに適用され、NetCDF内の数値には影響しません。

この出力は、Along・Cross・Obliqueの3方向勾配から東西・南北成分を推定する
LSA3解析の第一段階となる、スワス座標系勾配データセットとして使用できます。
