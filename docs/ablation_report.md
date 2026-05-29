# 阶段八消融实验报告

本报告只记录基于 FDU bbox 中心点 proxy 的 CSRNet baseline 消融结果。指标含义是毛囊/标注目标计数误差，不代表真实业务 `根/cm^2` 精度。

## 实验矩阵

| 实验 ID | 变量 | 对照 | 目的 | 远程训练状态 |
|---|---|---|---|---|
| baseline_sigma4_count_pretrained_false | sigma=4, MSE + Count Loss, 不使用预训练, 512 crop | - | 作为阶段八统一对照组 | 已完成 |
| e1_sigma8 | sigma=8 | baseline sigma=4 | 判断高斯核宽度对计数误差的影响 | 已完成 |
| e2_mse_only | lambda_count=0.0 | baseline lambda_count=0.1 | 判断计数辅助损失是否有效 | 已完成 |
| e3_pretrained_true | model.pretrained=true | baseline pretrained=false | 验证 ImageNet 预训练收益 | 已完成 |
| e4_exclude_abnormal | data.exclude_classes=[abnormal] | baseline 保留 abnormal | 判断异常类别对计数任务的影响 | 已完成 |
| e5_resize_full_image | data.transform_mode=resize | baseline 512 crop | 比较整图缩放与裁剪训练的稳定性和误差 | 已完成 |

## 运行命令生成

在远程主机完成 CUDA smoke test 后，先安装项目依赖并同步 FDU 数据集，然后生成阶段八命令：

```bash
python tools/ablation_matrix.py --plan configs/ablations/ablation_plan.yaml --device cuda --split test --output outputs/ablations/run_stage8.sh
```

随后逐条执行 `outputs/ablations/run_stage8.sh` 中的训练和测试命令。每个实验会写入独立目录：

```text
outputs/ablations/<experiment_id>/checkpoints/csrnet_best.pth
outputs/ablations/<experiment_id>/predictions/test_predictions.csv
```

## 结果记录

| 实验 ID | Val MAE | Val RMSE | Test MAE | Test RMSE | 结论 |
|---|---:|---:|---:|---:|---|
| baseline_sigma4_count_pretrained_false | 0.9537 | 1.2032 | 0.9867 | 1.2550 | 默认基线表现稳定，可作为保守对照 |
| e1_sigma8 | 0.9531 | 1.2073 | 0.9824 | 1.2559 | sigma=8 与 sigma=4 差异很小，Test MAE 略低但 RMSE 略高 |
| e2_mse_only | 1.0661 | 1.4366 | 1.1561 | 1.4781 | 去掉 Count Loss 后明显变差，保留计数辅助损失 |
| e3_pretrained_true | 0.9535 | 1.2042 | 0.9851 | 1.2548 | 与 baseline 基本持平，预训练收益不明显 |
| e4_exclude_abnormal | 0.8970 | 1.1751 | 0.9332 | 1.2214 | 当前最优配置，过滤 abnormal 后 Val/Test 均下降 |
| e5_resize_full_image | 2.7455 | 3.5915 | 2.6122 | 3.3479 | 整图缩放显著变差，不建议替代 512 crop |

## 推荐 baseline 选择规则

优先选择 Test MAE/RMSE 更低且训练过程稳定的配置。如果 Test 指标接近，优先选择解释成本更低、与阶段五到阶段七默认链路一致的配置：`sigma=4`、`MSE + Count Loss`、`512 crop`。

## 消融结论

推荐阶段八后的 baseline 使用 `e4_exclude_abnormal`：`sigma=4`、`MSE + Count Loss`、不使用 ImageNet 预训练、过滤 `abnormal` 类、`512 crop`。该配置在验证集和测试集上均为当前最优，Test MAE 为 `0.9332`，Test RMSE 为 `1.2214`。

从变量影响看，`sigma=4` 与 `sigma=8` 差异很小，当前没有必要为了 `sigma=8` 改默认配置。Count Loss 对计数误差有明确帮助，去掉后 Test MAE 从 `0.9867` 升到 `1.1561`。ImageNet 预训练在本轮实验中没有带来稳定收益，可能与数据规模、训练轮数、预训练权重是否完整加载、以及 FDU 图像域差异有关。过滤 `abnormal` 类有效降低误差，说明当前计数任务更适合先学习主要毛囊目标。整图缩放明显劣于裁剪训练，说明直接把 1280 x 1024 压到 512 x 512 会损失局部细节，不适合作为当前训练策略。

后续阶段如继续优化，优先在 `e4_exclude_abnormal` 基础上做学习率、batch size、训练轮数和 checkpoint 选择策略的微调，不建议优先推进整图缩放路线。
