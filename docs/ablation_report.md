# 阶段八消融实验报告

本报告只记录基于 FDU bbox 中心点 proxy 的 CSRNet baseline 消融结果。指标含义是毛囊/标注目标计数误差，不代表真实业务 `根/cm^2` 精度。

## 实验矩阵

| 实验 ID | 变量 | 对照 | 目的 | 远程训练状态 |
|---|---|---|---|---|
| baseline_sigma4_count_pretrained_false | sigma=4, MSE + Count Loss, 不使用预训练, 512 crop | - | 作为阶段八统一对照组 | 待远程训练 |
| e1_sigma8 | sigma=8 | baseline sigma=4 | 判断高斯核宽度对计数误差的影响 | 待远程训练 |
| e2_mse_only | lambda_count=0.0 | baseline lambda_count=0.1 | 判断计数辅助损失是否有效 | 待远程训练 |
| e3_pretrained_true | model.pretrained=true | baseline pretrained=false | 验证 ImageNet 预训练收益 | 待远程训练 |
| e4_exclude_abnormal | data.exclude_classes=[abnormal] | baseline 保留 abnormal | 判断异常类别对计数任务的影响 | 待远程训练 |
| e5_resize_full_image | data.transform_mode=resize | baseline 512 crop | 比较整图缩放与裁剪训练的稳定性和误差 | 待远程训练 |

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
| baseline_sigma4_count_pretrained_false | 待填 | 待填 | 待填 | 待填 | 待远程训练后填写 |
| e1_sigma8 | 待填 | 待填 | 待填 | 待填 | 待远程训练后填写 |
| e2_mse_only | 待填 | 待填 | 待填 | 待填 | 待远程训练后填写 |
| e3_pretrained_true | 待填 | 待填 | 待填 | 待填 | 待远程训练后填写 |
| e4_exclude_abnormal | 待填 | 待填 | 待填 | 待填 | 待远程训练后填写 |
| e5_resize_full_image | 待填 | 待填 | 待填 | 待填 | 待远程训练后填写 |

## 推荐 baseline 选择规则

优先选择 Test MAE/RMSE 更低且训练过程稳定的配置。如果 Test 指标接近，优先选择解释成本更低、与阶段五到阶段七默认链路一致的配置：`sigma=4`、`MSE + Count Loss`、`512 crop`。
