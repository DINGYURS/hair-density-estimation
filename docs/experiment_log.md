# 实验记录

本文档记录 FDU bbox 中心点密度估计 baseline 的训练和评估结果。当前指标只表示“毛囊/标注目标计数 proxy”的误差，不表示真实业务 `根/cm^2` 精度。

## 记录模板

### 实验：YYYY-MM-DD-csrnet-fdu

| 项目 | 内容 |
|---|---|
| 数据集 | FDU_HairFollicleDataset |
| 训练划分 | train=992 |
| 验证划分 | val=330 |
| 测试划分 | test=330 |
| 标注构造 | VOC bbox 修正与过滤后取中心点，生成高斯密度图 |
| 模型 | CSRNet |
| checkpoint | `outputs/checkpoints/csrnet_best.pth` |
| input_size | 512 |
| downsample | 8 |
| sigma | 4 |
| batch_size | 8 |
| epochs | 50 |
| learning_rate | 1e-6 |
| lambda_count | 0.1 |
| pretrained | false |
| 训练设备 | Linux RTX 4090 |
| 评估命令 | `python eval.py --config configs/csrnet_fdu.yaml --checkpoint outputs/checkpoints/csrnet_best.pth --split test --device cuda` |

| split | MAE | RMSE | 逐图 CSV |
|---|---:|---:|---|
| val | 待填写 | 待填写 | `outputs/predictions/val_predictions.csv` |
| test | 待填写 | 待填写 | `outputs/predictions/test_predictions.csv` |

结论：

- 数据处理：bbox 坐标先修正顺序并裁剪，修正后仍无效的 bbox 过滤；密度图由 bbox 中心点生成。
- 模型训练：CSRNet baseline，density MSE 加 count loss，保存验证集 MAE 最优 checkpoint。
- 指标解释：MAE/RMSE 是每张图预测 count 与裁剪/评估区域 GT count 的误差。

