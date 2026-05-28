# PRD.md

## 文档关系

本文档是产品需求来源。先读根目录 `AGENTS.md` 了解执行规则；当需要判断功能范围、非目标、验收标准或汇报口径时，再读本文档。实现细节见 `docs/project/TECH_DESIGN.md`，阶段计划见 `docs/project/PROJECT_PLAN.md`。

## 1. 背景

原始业务需求希望完成发量密度检测模型，并最终服务于真实设备或应用。但当前现实条件限制明显：

- NIA 数据集暂不可用。
- 自采业务数据短期内没有。
- 当前本地只有 FDU-HairFollicleDataset。
- FDU 标注是 VOC bbox，不是真实发根点标注。
- 本地开发环境为 Windows，正式训练环境为远程 Linux RTX 4090。

因此当前产品范围需要收敛为一个可复现的研究原型，而不是生产系统。

## 2. 当前目标

基于 FDU-HairFollicleDataset 完成一个毛囊/标注目标密度估计 baseline：

1. 解析 FDU 数据集。
2. 修正并过滤异常 bbox。
3. 将 bbox 中心点转换为高斯密度图。
4. 训练 CSRNet baseline。
5. 评估 val/test MAE 和 RMSE。
6. 对单图输出预测 count、密度热力图和叠加可视化图。
7. 形成可向导师说明的数据处理、训练、评估和可视化闭环。

## 3. 非目标

当前阶段不做以下事项：

- 不承诺真实业务 `根/cm^2` 精度。
- 不做生产级模型部署。
- 不做 REST API、小程序或后台服务。
- 不使用 NIA 或自采数据训练。
- 不优先引入 YOLO、Faster R-CNN、Transformer 或复杂模型。
- 不做 Kubernetes、高并发服务或云端部署。

## 4. 用户与使用场景

主要用户：

- 研究/开发者本人。
- 后续汇报对象，例如导师或项目负责人。

核心使用场景：

1. 了解 FDU 数据标注质量和类别分布。
2. 查看 bbox 可视化，确认标注修正是否正确。
3. 生成密度图，确认密度图积分与目标数一致。
4. 在远程 GPU 上训练 CSRNet baseline。
5. 在 val/test 上评估计数误差。
6. 生成单图热力图用于汇报展示。

## 5. 数据需求

数据集必须放在项目根目录：

```text
FDU_HairFollicleDataset/
  Annotations/
  Images/
  ImageSets/Main/
```

已知数据统计：

| 项目 | 数量 |
|---|---:|
| 图片数量 | 1652 |
| XML 标注文件 | 1652 |
| train | 992 |
| val | 330 |
| test | 330 |
| 总标注目标数 | 20695 |
| 平均每图目标数 | 12.53 |
| 每图目标数范围 | 5 - 28 |
| 图片分辨率 | 1280 x 1024 |

类别分布：

| 类别 | 数量 |
|---|---:|
| premium | 13728 |
| single | 4718 |
| undersize | 1492 |
| abnormal | 757 |

数据质量要求：

- bbox 坐标顺序必须修正。
- 修正后仍无效的 bbox 必须过滤。
- train/val/test 必须保持无交叉。
- 数据集不进入 Git 仓库。

## 6. 功能需求

### 6.1 数据分析

系统应提供数据分析脚本：

- 统计图片数量和 XML 数量。
- 统计 split 数量。
- 统计类别分布。
- 统计每图目标数 min、max、mean。
- 统计异常 bbox 数量。
- 检查 train、val、test 是否交叉。
- 输出 `docs/data_report.md`。

### 6.2 标注可视化

系统应提供标注可视化脚本：

- 从 train、val、test 随机抽样。
- 绘制 bbox 和类别名。
- 保存到 `outputs/visualizations/annotations/`。
- 至少能生成 30 张可检查图片。

### 6.3 密度图生成

系统应将 bbox 中心点转换为密度图：

- 支持固定 `sigma=4` 和 `sigma=8`。
- 支持原图尺度密度图。
- 支持 `1/8` 下采样密度图。
- 明确保证或解释密度图积分与目标数的关系。

### 6.4 Dataset/DataLoader

系统应提供 PyTorch Dataset：

- 读取 split txt。
- 加载图片。
- 加载 XML 标注。
- 生成密度图。
- 返回 image tensor、density tensor、count、image_id。
- 支持 512 x 512 随机裁剪。
- 裁剪时同步过滤和平移中心点。
- 支持水平翻转、垂直翻转、轻微亮度/对比度扰动。

### 6.5 模型训练

系统应提供 CSRNet baseline：

- VGG-16 frontend。
- 空洞卷积 backend。
- 输出单通道密度图。
- 输入 `[B, 3, 512, 512]` 时输出 `[B, 1, 64, 64]`。

训练脚本应支持：

- 读取 config。
- 创建 train/val DataLoader。
- 创建模型。
- MSE density loss。
- 可选 count loss。
- 保存 best checkpoint。
- 输出 val MAE/RMSE。

### 6.6 评估

评估脚本应支持：

- 加载 checkpoint。
- 在 val/test 上推理。
- 输出 MAE、RMSE。
- 输出逐图真实数量、预测数量、误差。
- 保存 `outputs/predictions/test_predictions.csv`。

### 6.7 单图推理

推理脚本应支持：

- 输入任意 FDU 图片。
- 加载 checkpoint。
- 输出预测 count。
- 保存密度热力图。
- 保存原图叠加热力图。
- 可选在图上写真实 count、预测 count、误差。

## 7. 非功能需求

- 跨 Windows/Linux 可运行。
- 所有路径使用 `pathlib.Path`。
- 所有实验参数从配置文件读取。
- 输出路径统一位于 `outputs/`。
- 训练结果可复现，至少固定随机种子。
- 核心脚本应能通过 `--config configs/csrnet_fdu.yaml` 运行。
- 本地 CPU smoke test 应能快速验证代码通路。
- 远程 GPU 训练前必须完成 CUDA smoke test。

## 8. 验收标准

阶段性验收：

- Windows 本地 `hair-density` 环境可用。
- PyTorch 2.4.1 CPU 版可导入。
- 远程 Linux PyTorch 可调用 RTX 4090。
- 数据报告可生成。
- 至少 30 张 bbox 可视化图可查看。
- 随机 100 张密度图积分误差可解释。
- DataLoader 能稳定输出 batch。
- CSRNet forward 能跑通且输出尺寸正确。
- Windows 本地可跑 1 个 mini-batch forward/loss smoke test。
- Linux 远程至少完整训练 1 epoch。
- val/test 可输出 MAE、RMSE。
- 单图推理可生成热力图和叠加图。

最终原型验收：

- 能从 FDU 数据到模型训练、评估、单图可视化完整闭环。
- README 说明环境、数据结构、训练、评估和推理方法。
- `docs/experiment_log.md` 能记录实验配置、指标和结论。
- 汇报材料中清楚说明当前是 FDU bbox proxy baseline，而非真实 `根/cm^2` 业务模型。
