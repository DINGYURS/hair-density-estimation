# TECH_DESIGN.md

## 文档关系

本文档是技术设计来源。先读根目录 `AGENTS.md` 获取高频工程规则；当需要实现模块、设计接口、确认数据流、训练/评估细节时，再读本文档。产品范围见 `docs/project/PRD.md`，阶段计划见 `docs/project/PROJECT_PLAN.md`。

## 1. 技术目标

构建一个跨 Windows/Linux 的 PyTorch 计算机视觉工程，用 FDU VOC bbox 标注训练毛囊/标注目标密度估计 baseline。第一版模型使用 CSRNet，训练目标是密度图回归和图像内目标计数误差优化。

## 2. 技术栈

- Python 3.10
- PyTorch 2.4.1
- torchvision 0.19.1
- OpenCV
- Pillow
- NumPy
- Matplotlib
- PyYAML
- pandas
- scikit-learn
- tensorboard
- pytest

PyTorch 安装策略：

- Windows 本地：`pytorch==2.4.1` CPU 版。
- Linux 远程：优先保留或安装 `pytorch==2.4.1` CUDA 12.1 构建；如果 CUDA smoke test 失败，再考虑 CUDA 11.8 构建。
- `requirements.txt` 不包含 PyTorch。

## 3. 目录结构

```text
hair-density-estimation/
  FDU_HairFollicleDataset/
  configs/
    csrnet_fdu.yaml
  docs/
    data_report.md
    experiment_log.md
  src/
    datasets/
      __init__.py
      fdu_dataset.py
    models/
      __init__.py
      csrnet.py
    utils/
      __init__.py
      voc_parser.py
      density_map.py
      metrics.py
      visualize.py
      seed.py
  tools/
    analyze_fdu.py
    visualize_annotations.py
    generate_density_preview.py
    check_dataloader.py
  outputs/
    visualizations/
    density_previews/
    checkpoints/
    logs/
    predictions/
  train.py
  eval.py
  infer.py
  requirements.txt
  README.md
```

## 4. 配置设计

默认配置文件：`configs/csrnet_fdu.yaml`

```yaml
project:
  seed: 42

data:
  root: FDU_HairFollicleDataset
  train_split: ImageSets/Main/train.txt
  val_split: ImageSets/Main/val.txt
  test_split: ImageSets/Main/test.txt

train:
  device: cuda
  input_size: 512
  batch_size: 8
  num_workers: 8
  epochs: 50
  learning_rate: 1.0e-6
  sigma: 4
  lambda_count: 0.1

output:
  root: outputs
```

命令行参数应允许覆盖关键字段：

- `--config`
- `--device`
- `--batch-size`
- `--num-workers`
- `--checkpoint`
- `--split`
- `--image`
- `--output-dir`

## 5. 数据流

```text
VOC XML + JPG
  -> voc_parser.py
  -> normalized bbox records
  -> bbox center points
  -> density_map.py
  -> image tensor + density tensor
  -> CSRNet
  -> predicted density map
  -> count_from_density
  -> MAE/RMSE + visualizations
```

## 6. VOC 解析设计

模块：`src/utils/voc_parser.py`

建议数据结构：

```python
{
    "image_id": "230219_A001_1",
    "filename": "230219_A001_1.jpg",
    "width": 1280,
    "height": 1024,
    "objects": [
        {
            "class_name": "premium",
            "bbox": [x1, y1, x2, y2],
            "raw_bbox": [xmin, ymin, xmax, ymax],
        }
    ],
    "invalid_objects": [
        {
            "class_name": "single",
            "raw_bbox": [xmin, ymin, xmax, ymax],
            "reason": "non_positive_area",
        }
    ],
}
```

解析规则：

- XML 解析使用标准库 `xml.etree.ElementTree`。
- 坐标读取后转为 `float`。
- 使用 min/max 修正坐标顺序。
- 对超出图像范围的 bbox，先裁剪到 `[0, width]` 和 `[0, height]`。
- 裁剪后 `x2 <= x1` 或 `y2 <= y1` 则过滤。
- 返回有效对象和无效对象，便于统计。

## 7. 数据分析设计

模块：`tools/analyze_fdu.py`

输入：

- `--data-root FDU_HairFollicleDataset`
- `--output docs/data_report.md`

输出内容：

- 图片数量。
- XML 数量。
- split 数量。
- split 交叉检查。
- 类别分布。
- 每图目标数 min、max、mean。
- 原始异常 bbox 数量。
- 修正后无效 bbox 数量。
- 数据质量结论。

实现要点：

- 使用 `pathlib.Path` 遍历。
- 使用 `voc_parser.py` 作为唯一解析入口。
- 使用 `collections.Counter` 统计类别。
- 输出 Markdown 表格。

## 8. 可视化设计

模块：

- `src/utils/visualize.py`
- `tools/visualize_annotations.py`
- `tools/generate_density_preview.py`

标注可视化：

- OpenCV 读取图片。
- 每个类别使用固定颜色。
- 绘制 bbox、类别名。
- 输出到 `outputs/visualizations/annotations/`。

密度图可视化：

- 使用 `cv2.applyColorMap`。
- 归一化密度图到 0-255。
- overlay alpha 建议 0.35 到 0.5。
- 图上可写 `gt_count`、`density_sum`、`pred_count`。

## 9. 密度图设计

模块：`src/utils/density_map.py`

核心函数：

- `bbox_to_points(objects) -> list[tuple[float, float]]`
- `make_density_map(points, height, width, sigma, downsample=1) -> np.ndarray`
- `resize_density_map_keep_count(density, output_size) -> np.ndarray`

积分原则：

- 原图尺度密度图 `density.sum()` 应接近有效点数量。
- 如果下采样到 `1/8`，必须保持积分不变。
- 推荐先在原图尺度生成密度图，再用面积插值或显式缩放保持 sum。
- 验收误差目标：`abs(density.sum() - count) < 1e-3`，若下采样误差更大，需要在报告中说明原因。

边界处理：

- 高斯核靠近边界时只截取有效区域。
- 截断后对局部高斯重新归一化，使该点贡献总和为 1。
- 空点列表返回全 0 密度图。

## 10. Dataset 设计

模块：`src/datasets/fdu_dataset.py`

类：`FduDensityDataset`

初始化参数：

- `data_root`
- `split_file`
- `input_size`
- `sigma`
- `downsample`
- `training`
- `augment`

返回字段：

```python
{
    "image": Tensor[3, H, W],
    "density": Tensor[1, H/8, W/8],
    "count": Tensor[],
    "image_id": str,
}
```

训练模式：

- 加载原图和有效 bbox。
- 随机裁剪 512 x 512。
- 保留裁剪区域内中心点。
- 平移点坐标到 crop 局部坐标。
- 生成密度图。
- 执行随机翻转和轻微颜色增强。

验证/测试模式：

- 第一版可使用中心裁剪或缩放到固定尺寸。
- 必须明确 count 与图像区域是否一致。
- 后续可支持滑窗或整图推理。

## 11. 模型设计

模块：`src/models/csrnet.py`

CSRNet 第一版：

- frontend：VGG-16 前 10 个卷积层和 pooling，输出下采样 1/8。
- backend：空洞卷积 dilation=2。
- output layer：`Conv2d(64, 1, kernel_size=1)`。
- 输出非负策略：第一版可使用 `ReLU` 或训练后 clamp，推荐模型 forward 末尾使用 `relu`，避免负密度影响 count。

输入输出：

```text
input:  [B, 3, 512, 512]
output: [B, 1, 64, 64]
```

预训练策略：

- 有网络时使用 torchvision ImageNet VGG16 weights。
- 无网络时允许随机初始化跑通。
- 训练日志中必须记录是否使用预训练。

## 12. 训练设计

入口：`train.py`

职责：

- 读取 YAML config。
- 设置随机种子。
- 创建 train/val dataset 和 dataloader。
- 创建 CSRNet。
- 创建 optimizer，第一版使用 Adam。
- 每个 epoch 训练并在 val 上评估。
- 保存 best checkpoint。
- 写 TensorBoard 日志。

Loss 第一版：

```text
loss = MSE(pred_density, gt_density)
```

Loss 第二版：

```text
loss = MSE(pred_density, gt_density) + lambda_count * abs(sum(pred_density) - sum(gt_density))
```

初始训练参数：

- `input_size: 512`
- `batch_size: 8`
- `epochs: 50`
- `learning_rate: 1e-6`
- `sigma: 4`
- `lambda_count: 0.1`
- Windows 本地 `num_workers: 0`
- Linux 远程 `num_workers: 8`

Checkpoint 内容：

```python
{
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "config": config,
    "best_mae": best_mae,
}
```

## 13. 评估设计

入口：`eval.py`

职责：

- 加载 config 和 checkpoint。
- 选择 `val` 或 `test` split。
- 推理每张图。
- 从 density sum 得到预测 count。
- 计算 MAE、RMSE。
- 保存逐图 CSV。

模块：`src/utils/metrics.py`

函数：

- `count_from_density(density)`
- `mae(pred_counts, gt_counts)`
- `rmse(pred_counts, gt_counts)`

CSV 字段：

```text
image_id,gt_count,pred_count,abs_error,squared_error
```

## 14. 推理设计

入口：`infer.py`

参数：

- `--config`
- `--checkpoint`
- `--image`
- `--output-dir`
- `--device`

输出：

- 预测 count。
- 密度热力图。
- 原图叠加热力图。

限制：

- 单图推理第一版可只支持 FDU 分辨率或固定 resize。
- 如果 resize 改变尺度，必须保持 density count 解释一致。

## 15. 测试与 Smoke Test

建议测试：

- `voc_parser` 对坐标颠倒 bbox 的修正。
- `voc_parser` 对无效 bbox 的过滤。
- `density_map` 单点积分为 1。
- `density_map` 多点积分等于点数。
- `metrics` MAE/RMSE。

本地 smoke test：

```powershell
conda activate hair-density
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
python -c "import cv2; print(cv2.__version__)"
python -c "from PIL import Image; print('Pillow OK')"
python -c "import numpy as np; print(np.__version__)"
python tools/check_dataloader.py --config configs/csrnet_fdu.yaml --device cpu --batch-size 1 --num-workers 0
```

远程 CUDA smoke test：

```bash
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda runtime:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("matmul ok:", float(y[0, 0]))
PY
```

## 16. 风险与技术处理

| 风险 | 技术处理 |
|---|---|
| bbox 不是点标注 | 使用 bbox 中心点作为 proxy，并在报告中说明 |
| bbox 坐标异常 | `voc_parser.py` 统一修正和过滤 |
| 数据量较小 | 使用预训练、数据增强、严格 val/test |
| Windows 本地无 GPU | 本地只做 CPU smoke test，训练放远程 |
| 远程 CUDA 异常 | 先 smoke test，再决定是否改装 CUDA 11.8 PyTorch |
| VGG 权重下载失败 | 允许随机初始化跑通闭环 |
| 显存不足 | 使用 512 crop，从 batch size 8 起步 |
| 指标不理想 | 强调当前目标是 baseline 和工程闭环 |
