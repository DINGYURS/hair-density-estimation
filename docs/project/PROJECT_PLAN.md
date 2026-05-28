# FDU 发量密度估计项目实施计划

## 文档关系

本文档是长期实施计划和背景约束来源。先读根目录 `AGENTS.md` 获取当前执行规则；需要产品范围时读 `docs/project/PRD.md`；需要模块与接口设计时读 `docs/project/TECH_DESIGN.md`；需要阶段拆解、时间安排、学习路线和风险处理时再读本文档。

## 1. 当前现实约束

原始需求文档建议使用 NIA + FDU + 自采数据完成预训练、域适配和生产部署。但当前实际条件是：

- NIA 数据集无法获取。
- 自采数据集短期内没有。
- 当前目录只有 FDU-HairFollicleDataset。
- 本地代码开发环境是 Windows。
- 本地需要使用 Conda 创建新的虚拟环境。
- 正式模型训练环境是远程 Linux 主机，GPU 为 RTX 4090。
- 远程主机信息：CUDA Toolkit 11.8，`nvidia-smi` 显示 Driver Version 525.147.05、CUDA Version 12.0，Python 3.10.9、pip 24.0、conda 23.1.0，当前 PyTorch 为 `2.4.1+cu121`。
- 个人背景是研0计算机学生，本科以 Web 开发为主，需要先把计算机视觉训练闭环跑通。

因此当前阶段不应直接追求生产级 `根/cm²` 指标，也不应先做 REST API 或小程序对接。当前最合理目标是：

> 基于 FDU-HairFollicleDataset，完成一个可复现的发量/毛囊密度估计原型：能解析数据、生成密度图、训练 CSRNet baseline、评估计数误差，并输出可视化热力图。

## 2. 已核对的数据集情况

数据集位置：

```text
FDU_HairFollicleDataset/
  Annotations/
  Images/
  ImageSets/Main/
```

实际统计结果：

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
| 图片分辨率 | 全部 1280 x 1024 |

类别分布：

| 类别 | 数量 |
|---|---:|
| premium | 13728 |
| single | 4718 |
| undersize | 1492 |
| abnormal | 757 |

标注质量注意事项：

- 已发现 3119 个 bbox 存在坐标顺序颠倒，例如 `ymin > ymax` 或 `xmin > xmax`。
- 已发现 93 个 bbox 在 min/max 归一化后仍然宽度或高度无效。
- train/val/test 三个划分没有交叉，合计覆盖 1652 张图。

处理原则：

- 解析 bbox 时必须使用 `min(xmin, xmax)`、`max(xmin, xmax)`、`min(ymin, ymax)`、`max(ymin, ymax)` 修正坐标顺序。
- 对修正后宽度或高度仍然无效的框，应丢弃并记录日志。
- 当前 FDU 标注是毛囊/目标框，不是真实业务里的发根点标注。因此当前阶段的计数目标应描述为“毛囊/标注目标计数 proxy”，不能直接声称已经得到真实 `根/cm²` 业务精度。

## 3. 双环境开发与训练方案

本项目采用“双环境”工作流：

- Windows 本地：负责代码开发、数据解析、可视化、单元测试、小 batch CPU smoke test。
- Linux RTX 4090 远程主机：负责正式训练、长时间实验、checkpoint 生成、测试集评估。

核心原则：

- 代码必须跨 Windows/Linux 可运行，所有路径使用 `pathlib.Path`。
- 配置文件中只写相对路径或可配置路径，不在代码里硬编码 `W:\...` 或 `/home/...`。
- Windows 本地不需要安装 CUDA Toolkit；本地 PyTorch 优先安装 CPU 版，保证 API 与远程训练环境一致即可。
- 远程训练环境已有 GPU PyTorch，应先做 CUDA smoke test，再决定是否重装 PyTorch。

### 3.1 Windows 本地 Conda 环境

建议新建独立 Conda 环境：

```powershell
conda create -n hair-density python=3.10 -y
conda activate hair-density
```

说明：

- 建议 Windows 本地使用 Python 3.10，与远程 Python 3.10.9 保持同一大版本。
- 不建议直接使用 base 环境，后续依赖容易混乱。
- 本地环境目标是开发和调试，不承担正式训练。

### 3.2 Windows 本地 PyTorch 安装策略

为了尽量贴近远程主机，Windows 本地建议安装 PyTorch 2.4.1 CPU 版：

```powershell
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 cpuonly -c pytorch -y
```

验证：

```powershell
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```

预期：

- `torch.__version__` 应为 `2.4.1` 或接近版本。
- `torch.cuda.is_available()` 在 Windows 本地可以是 `False`，这是正常的。

如果 Windows 本地也有 NVIDIA 显卡，仍建议第一阶段先用 CPU 版，避免本地 CUDA 环境和远程 CUDA 环境不一致带来额外问题。真正需要训练时交给远程 RTX 4090。

### 3.3 Windows 本地安装项目依赖

创建 `requirements.txt` 后，安装其余依赖：

```powershell
pip install -r requirements.txt
```

建议 `requirements.txt` 第一版包含：

```text
numpy
opencv-python
Pillow
matplotlib
tqdm
PyYAML
scikit-learn
pandas
tensorboard
pytest
```

注意：

- PyTorch 不写入 `requirements.txt`，因为 Windows 本地和 Linux 远程使用的 CUDA/CPU 构建不同。
- Windows 下路径统一使用 `pathlib.Path`。
- 输出目录统一写到 `outputs/`，不要写到系统临时目录或绝对路径。

### 3.4 Windows 本地环境验收

环境配置完成后，至少确认下面命令都能运行：

```powershell
conda activate hair-density
python --version
where python
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
python -c "import cv2; print(cv2.__version__)"
python -c "from PIL import Image; print('Pillow OK')"
python -c "import numpy as np; print(np.__version__)"
```

### 3.5 远程 Linux 训练环境

远程主机当前信息：

```text
GPU: RTX 4090
OS: Linux
CUDA Toolkit: 11.8
nvidia-smi Driver Version: 525.147.05
nvidia-smi CUDA Version: 12.0
Python: 3.10.9
pip: 24.0
conda: 23.1.0
torch: 2.4.1+cu121
```

需要理解两点：

- `nvidia-smi` 里显示的 CUDA Version 表示驱动最高支持的 CUDA 运行时能力，不等于本机安装的 CUDA Toolkit 版本。
- PyTorch 的 `+cu121` wheel/conda 包通常自带 CUDA runtime；只要不编译自定义 CUDA 扩展，项目训练一般不依赖系统 CUDA Toolkit 11.8。

远程环境当前不建议立刻重装 PyTorch。先执行 GPU smoke test：

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

验收标准：

- `cuda available` 为 `True`。
- 能显示 RTX 4090。
- CUDA tensor 创建和矩阵乘法不报错。

如果 smoke test 通过：

- 保持当前 `torch==2.4.1+cu121`。
- 训练代码按当前环境适配。

如果 smoke test 失败：

- 优先不要改代码，先修远程环境。
- 方案 A：让管理员升级 NVIDIA Driver 到更高版本。
- 方案 B：改装与驱动/环境更保守匹配的 PyTorch CUDA 11.8 版本：

```bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

版本依据：

- NVIDIA CUDA 12.1 release notes 中，CUDA 12.1.x 的 Linux 最低驱动版本兼容要求为 `>=525.60.13`，当前远程驱动 `525.147.05` 高于该值，因此 `torch 2.4.1+cu121` 有机会正常工作，但仍必须以实际 CUDA smoke test 为准。
- PyTorch 2.4.1 官方安装命令同时支持 `pytorch-cuda=11.8` 和 `pytorch-cuda=12.1`，所以如果远程 `cu121` 运行失败，可以退回 CUDA 11.8 构建。
- 参考链接：NVIDIA CUDA 12.1 Release Notes: https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/；PyTorch Previous Versions: https://docs.pytorch.org/get-started/previous-versions/

### 3.6 远程训练 Conda 环境建议

如果远程当前 PyTorch 是装在 base 环境里，不建议继续污染 base。建议创建独立环境：

```bash
conda create -n hair-density-train python=3.10.9 -y
conda activate hair-density-train
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

如果创建新环境后 CUDA smoke test 失败，改用 CUDA 11.8 版：

```bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### 3.7 Windows 与 Linux 代码同步

推荐同步方式：

1. Windows 本地使用 Git 管理代码。
2. 远程主机 clone 同一个仓库。
3. 数据集 `FDU_HairFollicleDataset/` 可以单独用 `scp`/`rsync` 上传到远程，不建议提交到 Git。
4. 每次训练前在远程执行 `git pull` 或同步代码。
5. 训练输出的 `outputs/checkpoints/` 和 `outputs/logs/` 保留在远程；重要结果再下载到 Windows 本地。

项目代码需要遵守：

- 所有路径从 config 读取。
- 所有脚本都支持 `--config configs/csrnet_fdu.yaml`。
- 不在代码里写 Windows 盘符。
- 不在代码里写远程用户目录。

建议配置文件写法：

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

output:
  root: outputs
```

Windows 本地 smoke test 时可覆盖为：

```powershell
python tools/check_dataloader.py --config configs/csrnet_fdu.yaml --device cpu --batch-size 1 --num-workers 0
```

远程训练时使用：

```bash
python train.py --config configs/csrnet_fdu.yaml
```

## 4. 项目最终原型目标

在只有 FDU 数据集的前提下，本阶段交付物应包括：

1. 数据统计报告。
2. VOC XML 标注解析器。
3. bbox 可视化脚本。
4. bbox 中心点到高斯密度图的转换脚本。
5. PyTorch Dataset 和 DataLoader。
6. CSRNet baseline 模型。
7. 训练脚本。
8. 评估脚本，输出 MAE、RMSE。
9. 单图推理脚本，输出预测数量、密度热力图、叠加可视化图。
10. README，说明环境、数据结构、训练、评估和推理方法。

阶段性验收标准：

- 任意一张图的密度图积分应接近有效标注目标数，误差小于 `1e-3` 或由下采样/裁剪策略解释清楚。
- 训练流程能完整跑通至少 1 个 epoch。
- 验证集和测试集能输出 MAE/RMSE。
- 单图推理能生成可查看的热力图。
- 所有实验参数、数据划分和输出路径可复现。

## 5. 推荐项目结构

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
  PROJECT_PLAN.md
```

## 6. 分阶段实施计划

### 阶段 0：环境与仓库整理

目标：让项目变成一个可维护、可在 Windows 开发并在 Linux 远程训练的 Python/CV 工程。

任务：

1. Windows 本地创建 Conda 环境 `hair-density`。
2. Windows 本地安装 PyTorch 2.4.1 CPU 版和基础依赖。
3. 远程 Linux 创建或确认训练环境 `hair-density-train`。
4. 远程执行 CUDA smoke test，确认 RTX 4090 可被 PyTorch 正常调用。
3. 创建项目目录结构。
4. 编写 `.gitignore`，排除 checkpoint、日志、缓存、可视化输出等大文件。
5. 创建 `requirements.txt`。
6. 在 README 中写清楚 Windows 本地环境、Linux 训练环境、代码同步和数据集目录要求。

验收：

- Windows 本地 `conda activate hair-density` 可正常进入环境。
- Windows 本地 `python -c "import torch; print(torch.__version__)"` 输出 2.4.1 或接近版本。
- Windows 本地 DataLoader smoke test 可用 CPU 跑通。
- 远程 Linux `torch.cuda.is_available()` 为 `True`。
- 远程 Linux 能完成 CUDA tensor 创建和矩阵乘法 smoke test。
- 项目目录结构创建完成。

### 阶段 1：数据理解与标注可视化

目标：先确认数据是否可信，避免带着错误标注训练模型。

任务：

1. 实现 `src/utils/voc_parser.py`：
   - 读取 VOC XML。
   - 提取图片宽高。
   - 提取每个 object 的类别和 bbox。
   - 修正坐标顺序。
   - 过滤无效 bbox。
   - 返回统一结构。

2. 实现 `tools/analyze_fdu.py`：
   - 统计图片数量、标注数量、类别分布。
   - 统计每图目标数 min/max/mean。
   - 统计异常 bbox 数量。
   - 检查 train/val/test 是否有交叉。
   - 输出 `docs/data_report.md`。

3. 实现 `tools/visualize_annotations.py`：
   - 随机抽取 train/val/test 图片。
   - 把 bbox 和类别画到原图上。
   - 保存到 `outputs/visualizations/annotations/`。

验收：

- 能生成数据统计报告。
- 能看到至少 30 张带标注框的可视化图。
- 能确认坐标修正后框位置基本正确。

### 阶段 2：密度图生成

目标：把检测框标注转换为密度估计模型可训练的 Ground Truth。

任务：

1. 实现 `src/utils/density_map.py`：
   - bbox 转中心点。
   - 根据中心点生成高斯密度图。
   - 支持固定 `sigma=4`、`sigma=8`。
   - 后续支持自适应 sigma。
   - 支持输出原图大小密度图或 `1/8` 下采样密度图。

2. 实现密度图积分校验：
   - 原图尺度密度图的 sum 应接近目标数。
   - 如果生成 `1/8` 密度图，需要明确是否已保持积分不变。

3. 实现 `tools/generate_density_preview.py`：
   - 随机选图。
   - 保存原图、bbox 图、密度热力图、叠加图。
   - 标注真实目标数和密度图积分。

验收：

- 随机 100 张图的密度图积分误差可解释。
- 能生成清晰热力图。
- 密度图峰值位置和 bbox 中心位置一致。

### 阶段 3：PyTorch Dataset 与数据增强

目标：把 FDU 数据封装为可训练数据管线。

任务：

1. 实现 `src/datasets/fdu_dataset.py`：
   - 读取 split txt。
   - 加载图片。
   - 加载 XML 标注。
   - 生成密度图。
   - 返回 image tensor、density map tensor、count、image_id。

2. 第一版先固定输入尺寸：
   - 原图是 1280 x 1024，直接训练会占显存。
   - 建议先使用 `512 x 512` 随机裁剪。
   - 裁剪时必须同步过滤/平移中心点。

3. 数据增强第一版：
   - 随机水平翻转。
   - 随机垂直翻转。
   - 轻微亮度/对比度扰动。

验收：

- DataLoader 能稳定输出 batch。
- 裁剪后密度图积分等于裁剪区域内目标数量。
- 图像增强不会破坏密度图位置对应关系。

### 阶段 4：CSRNet baseline

目标：先实现最经典、最容易解释的密度估计 baseline。

任务：

1. 实现 `src/models/csrnet.py`：
   - VGG-16 前端。
   - 空洞卷积后端。
   - 输出单通道密度图。
   - 输出层使用 ReLU 或保证非负的处理。

2. 预训练权重策略：
   - 有网络时使用 torchvision 的 ImageNet VGG16 权重。
   - 无网络时先允许随机初始化跑通训练流程。
   - 训练脚本中要清楚记录是否使用了预训练。

3. 模型 smoke test：
   - 输入 `[B, 3, 512, 512]`。
   - 输出应为 `[B, 1, 64, 64]`，对应 1/8 下采样。

验收：

- 模型 forward 能跑通。
- 输出尺寸符合预期。
- 参数量和结构清晰可打印。

### 阶段 5：训练脚本

目标：完成训练闭环。

任务：

1. 实现 `train.py`：
   - 加载 config。
   - 创建 train/val DataLoader。
   - 创建 CSRNet。
   - 定义 loss。
   - 训练多个 epoch。
   - 每个 epoch 在 val 上评估。
   - 保存 best checkpoint。

2. 第一版 loss：

```text
loss = MSE(pred_density, gt_density)
```

3. 第二版 loss：

```text
loss = MSE(pred_density, gt_density) + lambda_count * abs(sum(pred) - sum(gt))
```

4. 建议初始训练参数：

```text
input_size: 512
batch_size: 8 起步，RTX 4090 显存允许时逐步增加到 16
epochs: 50
optimizer: Adam
learning_rate: 1e-6
sigma: 4
lambda_count: 0.1
num_workers: Windows 本地 0，Linux 远程 8 起步
device: Windows 本地 cpu，Linux 远程 cuda
```

验收：

- Windows 本地能用 CPU 完成 1 个 mini-batch 的 forward/loss smoke test。
- Linux 远程至少能完整训练 1 个 epoch。
- loss 不为 NaN。
- val MAE/RMSE 能正常输出。
- best checkpoint 能保存。

### 阶段 6：评估脚本与实验记录

目标：让结果可以向导师汇报，而不是只说“训练过”。

任务：

1. 实现 `eval.py`：
   - 加载 checkpoint。
   - 在 val/test 上推理。
   - 输出 MAE、RMSE。
   - 输出每张图的真实数量、预测数量、误差。

2. 实现 `src/utils/metrics.py`：
   - `mae`
   - `rmse`
   - `count_from_density`

3. 保存预测结果：

```text
outputs/predictions/test_predictions.csv
```

4. 编写 `docs/experiment_log.md`，记录每次实验的配置、指标和结论。

验收：

- test set 有完整评估结果。
- 能比较不同实验配置。
- 能回答导师常问的三个问题：
  - 数据怎么处理的？
  - 模型怎么训练的？
  - 指标是多少？

### 阶段 7：单图推理与热力图可视化

目标：做出最直观的 demo。

任务：

1. 实现 `infer.py`：
   - 输入图片路径。
   - 加载 checkpoint。
   - 输出预测 count。
   - 输出密度热力图。
   - 输出原图叠加热力图。

2. 热力图可视化：
   - 使用 OpenCV colormap。
   - 叠加透明度建议 0.35 - 0.5。
   - 图上写真实 count、预测 count、误差。

验收：

- 能对任意 FDU 图片生成预测结果。
- 可视化图可以直接放进汇报 PPT。

### 阶段 8：消融实验

目标：用实验支撑技术选择。

建议至少做以下实验：

| 实验 | 变量 | 目的 |
|---|---|---|
| E1 | fixed sigma=4 vs sigma=8 | 确定高斯核宽度 |
| E2 | MSE vs MSE + Count Loss | 判断计数辅助损失是否有效 |
| E3 | 是否使用 ImageNet 预训练 | 验证预训练收益 |
| E4 | 是否过滤 abnormal | 判断异常类别对计数任务的影响 |
| E5 | 512 裁剪 vs 缩放整图 | 比较训练稳定性和计数误差 |

验收：

- 每组实验都有配置、指标和结论。
- 最终能选出一个推荐 baseline。

### 阶段 9：可选工程化服务

目标：利用 Web 开发背景，把模型做成可演示服务。

前提：

- 训练和推理脚本已经稳定。
- 有一个可用 checkpoint。

任务：

1. 用 FastAPI 封装推理接口。
2. 接口支持上传图片。
3. 返回 JSON：

```json
{
  "image_id": "xxx",
  "pred_count": 12.3,
  "heatmap_path": "outputs/...",
  "overlay_path": "outputs/..."
}
```

注意：

- 当前不要承诺 `根/cm²`，因为缺少设备物理标定 PPCE。
- 可以先返回 `pred_count` 和“相对密度热力图”。

## 7. 推荐时间安排

如果每周能投入 15 - 25 小时，建议按以下节奏：

| 周次 | 目标 | 交付 |
|---|---|---|
| 第 1 周 | Windows/远程 Conda 环境、数据解析与可视化 | 双环境可用、远程 CUDA smoke test、data_report、bbox 可视化 |
| 第 2 周 | 密度图生成 | density_map、热力图预览、积分校验 |
| 第 3 周 | Dataset/DataLoader | 可训练数据管线 |
| 第 4 周 | CSRNet forward + 远程训练 1 epoch | 本地 smoke test、远程训练脚本跑通 |
| 第 5 周 | 完成 baseline 训练 | val/test MAE/RMSE |
| 第 6 周 | 推理脚本和热力图 demo | 单图预测可视化 |
| 第 7 - 8 周 | 消融实验 | ablation_report |
| 第 9 周以后 | FastAPI demo 或改进模型 | 可演示系统 |

## 8. 阶段性汇报口径

第一次向导师汇报时，可以说：

> 由于 NIA 数据集和自采数据暂时不可用，我先将任务收敛为基于 FDU 数据集的密度估计 baseline。当前 FDU 是 VOC 检测框标注，因此我会先将 bbox 转换为中心点，再生成高斯密度图，用 CSRNet 做毛囊目标计数和热力图预测。该阶段目标是跑通数据处理、训练、评估和可视化闭环，为后续接入自采数据和设备标定做准备。

需要避免的说法：

- 不要说“已经能准确计算真实发量密度 `根/cm²`”。
- 不要说“能满足生产环境指标”。
- 不要把 FDU 的 bbox 计数直接等同于丝馥生终端真实业务里的发根密度。

更准确的说法：

- 当前做的是“毛囊/标注目标密度估计 baseline”。
- 当前输出的是“图像内目标数”和“相对密度热力图”。
- `根/cm²` 需要自有设备 PPCE 标定和自采数据验证后才能可靠输出。

## 9. 技术学习路线

优先学习：

1. PyTorch 基础：
   - Tensor shape。
   - Dataset/DataLoader。
   - forward/backward。
   - optimizer。
   - checkpoint 保存和加载。

2. 图像处理基础：
   - RGB/BGR。
   - resize/crop/flip。
   - 坐标变换。
   - heatmap 叠加。

3. 密度图估计：
   - 点标注转高斯密度图。
   - 密度图积分等于计数。
   - MAE/RMSE 评估。

4. CSRNet：
   - VGG frontend。
   - dilated convolution。
   - 输出下采样密度图。

5. 工程化：
   - 配置文件。
   - 日志。
   - 可复现实验。
   - FastAPI 推理服务。

暂时不优先学习：

- 复杂检测框架如 YOLO、Faster R-CNN。
- Transformer 大模型。
- 复杂部署如 Kubernetes。
- 生产级高并发服务。

## 10. 关键风险与处理方式

| 风险 | 影响 | 处理 |
|---|---|---|
| FDU 与业务场景差异大 | 模型不能直接用于真实设备 | 当前只做 baseline，明确等待自采数据微调 |
| bbox 不是点标注 | 密度图中心点是近似 | 用 bbox 中心点作为 proxy，并在报告中说明 |
| bbox 坐标异常 | 训练标签错误 | 解析阶段修正坐标并过滤无效框 |
| 数据量较小 | 过拟合 | 使用数据增强、预训练、严格 val/test |
| Windows 本地没有 GPU 或 GPU 环境不一致 | 本地无法正式训练 | 本地只做 CPU smoke test，正式训练放到 Linux RTX 4090 |
| 远程 PyTorch/CUDA 组合异常 | 训练无法使用 GPU | 先跑 CUDA smoke test；失败时优先修驱动或改装 PyTorch CUDA 11.8 |
| 显存不足 | 无法整图训练 | 使用 512 crop；RTX 4090 上从 batch_size 8 起步逐步增大 |
| 网络不可用导致 VGG 权重下载失败 | 无法用预训练 | 先随机初始化跑通，后续手动下载权重 |
| 指标不理想 | 难以汇报 | 强调当前目标是工程闭环和 baseline，而不是最终业务精度 |

## 11. 下一步具体行动清单

优先完成以下 10 件事：

1. Windows 本地创建 Conda 环境 `hair-density`。
2. Windows 本地安装 PyTorch 2.4.1 CPU 版和基础依赖。
3. 远程 Linux 创建或确认训练环境，并执行 CUDA smoke test。
4. 确定代码同步方式：Git 优先，数据集单独同步。
5. 建立项目目录结构。
6. 补全 `.gitignore`。
7. 创建 `requirements.txt`。
8. 编写 `src/utils/voc_parser.py`。
9. 编写 `tools/analyze_fdu.py` 并生成 `docs/data_report.md`。
10. 编写 `tools/visualize_annotations.py` 并保存 30 张 bbox 可视化。

最建议从双环境确认和 `voc_parser.py` 开始，因为它们决定后面所有代码能否稳定运行、训练能否真正使用 RTX 4090。
