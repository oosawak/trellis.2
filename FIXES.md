# TRELLIS.2 動作させるまでの修正内容

## 環境
- GPU: NVIDIA GeForce RTX 3090
- CUDA: 12.9 (Driver) / 12.4 (PyTorch)
- Python: 3.12.11 (mise 管理)
- OS: Linux (Ubuntu)

---

## 1. `setup.sh` — CuMesh を独立ビルドに分離

### 問題
- `--cumesh` オプションでビルドが失敗し、setup.sh 全体が止まる
- snap コンテナ内で `nvidia-smi` が見つからず GPU 検出に失敗
- `conda activate` が非インタラクティブシェルで動かない
- `flash-attn` が `No module named 'torch'` で失敗

### 修正
- `--cumesh` オプションを setup.sh から完全削除（独立スクリプトに分離）
- GPU 検出で複数パス (`/usr/bin/nvidia-smi` 等) を試行するよう修正
- `eval "$(conda shell.bash hook)"` を追加
- `flash-attn` の pip install に `--no-build-isolation` を追加

---

## 2. `clean_up.cu` — CuMesh の CUDA 12 対応パッチ（新規作成）

### 問題①: `cuda::std::__4::tuple` エラー
CUDA 12 で内部 ABI が変更され、`cuda::std::__4` 名前空間が使用不可になった。

### 問題②: `cub::DeviceRadixSort::SortPairs` の API 変更
CUB 2.x (CUDA 12) でカスタム decomposer を使う overload の API が変更された。
`begin_bit` / `end_bit` 引数の扱いが変わり、コンパイルエラーになる。

### 解決策
CUB の radix sort を **`thrust::sort_by_key` + `int3_less` 比較器** に完全置き換え。

```cpp
// 変更前: CUB DeviceRadixSort (CUDA 12 で動かない)
cub::DeviceRadixSort::SortPairs(
    nullptr, temp_storage_bytes,
    cu_sorted_faces, cu_sorted_faces_output,
    cu_sorted_face_indices, cu_sorted_indices_output,
    F, int3_decomposer{}
);

// 変更後: thrust::sort_by_key
struct int3_less {
    __device__ bool operator()(const int3& a, const int3& b) const {
        if (a.x != b.x) return a.x < b.x;
        if (a.y != b.y) return a.y < b.y;
        return a.z < b.z;
    }
};

thrust::sort_by_key(
    thrust::device,
    thrust::device_ptr<int3>(cu_sorted_faces),
    thrust::device_ptr<int3>(cu_sorted_faces + F),
    thrust::device_ptr<int>(cu_sorted_face_indices),
    int3_less()
);
```

---

## 3. `install_cumesh.sh` — CuMesh 独立ビルドスクリプト（新規作成）

CuMesh を以下の手順で自動ビルド・インストールする：

1. `/tmp/CuMesh` に clone
2. パッチ済み `clean_up.cu` をコピー
3. `python3 setup.py build_ext --inplace` でビルド
4. `pip install -e .` でインストール
5. `import cumesh` で動作確認

### 使い方
```bash
cd ~/TRELLIS.2
bash install_cumesh.sh
```

---

## 4. `o-voxel/pyproject.toml`

### 問題
o-voxel のインストール時に未パッチの CuMesh が自動でインストールされてしまう。

### 修正
以下の行を削除:
```
"cumesh @ git+https://github.com/JeffreyXiang/CuMesh.git"
```

---

## 5. `trellis2/modules/image_feature_extractor.py`

### 問題
`DINOv3ViTModel` の内部構造と実装が不一致。

```
AttributeError: 'DINOv3ViTModel' object has no attribute 'layer'
```

`DINOv3ViTModel` は `self.model`（encoder）を内部に持ち、`layer` はその中にネストされている。

### 修正
```python
# 変更前
for i, layer_module in enumerate(self.model.layer):

# 変更後
for i, layer_module in enumerate(self.model.model.layer):
```

---

## 6. `app.py` — 外部からのアクセスを有効化

### 問題
デフォルトでは `127.0.0.1` のみにバインドされ、外部からアクセスできない。

### 修正
```python
# 変更前
demo.launch(css=css, head=head)

# 変更後
demo.launch(css=css, head=head, server_name="0.0.0.0")
```

アクセス URL: `http://[マシンのIPアドレス]:7860`

---

## 7. HuggingFace 認証

`facebook/dinov3-vitl16-pretrain-lvd1689m` は Gated Repo のため、以下が必要：

1. https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m でアクセス申請
2. https://huggingface.co/settings/tokens でトークン（Read 権限）を発行
3. ログイン:
   ```python
   python -c "from huggingface_hub import login; login(token='hf_xxxx...')"
   ```

---

## セットアップ手順（まとめ）

```bash
# 1. 基本依存関係のインストール
cd ~/TRELLIS.2
bash setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --o-voxel --flexgemm

# 2. CuMesh のビルド・インストール（別途）
bash install_cumesh.sh

# 3. transformers のアップグレード（DINOv3 対応）
pip install --upgrade "transformers>=5.5.0"

# 4. HuggingFace ログイン
python -c "from huggingface_hub import login; login(token='hf_xxxx...')"

# 5. 起動
python app.py
```
