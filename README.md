# hair-density-estimation

基于 FDU-HairFollicleDataset 的毛囊/标注目标密度估计原型。

当前目标是先完成一个可复现的 baseline：解析 FDU VOC 标注，将 bbox 中心点转换为密度图，训练 CSRNet，评估计数误差，并输出热力图可视化。

当前输出应描述为“毛囊/标注目标计数 proxy”和“相对密度热力图”。它还不是生产级 `根/cm^2` 指标。

## 项目结构

```text
hair-density-estimation/
  FDU_HairFollicleDataset/
  configs/
    csrnet_fdu.yaml
  docs/
    project/
      PRD.md
      TECH_DESIGN.md
      PROJECT_PLAN.md
    data_report.md
    experiment_log.md
  src/
    datasets/
    models/
    utils/
  tools/
  outputs/
    visualizations/
    density_previews/
    checkpoints/
    logs/
    predictions/
  AGENTS.md
  requirements.txt
```

`FDU_HairFollicleDataset/` 已被 Git 忽略，应单独复制或同步，不提交到仓库。

## 文档入口

- [AGENTS.md](AGENTS.md)：agent 和开发协作入口，包含高频规则和文档导航。
- [docs/project/PRD.md](docs/project/PRD.md)：产品范围、非目标、功能需求和验收标准。
- [docs/project/TECH_DESIGN.md](docs/project/TECH_DESIGN.md)：技术设计、模块接口、数据流、训练评估方案。
- [docs/project/PROJECT_PLAN.md](docs/project/PROJECT_PLAN.md)：阶段计划、时间安排、学习路线和风险处理。

## Windows 本地开发环境

创建本地 CPU 开发环境：

```powershell
conda create -n hair-density python=3.10 -y
conda activate hair-density
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 cpuonly -c pytorch -y
pip install -r requirements.txt
```

验证本地环境：

```powershell
python --version
where python
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
python -c "import cv2; print(cv2.__version__)"
python -c "from PIL import Image; print('Pillow OK')"
python -c "import numpy as np; print(np.__version__)"
```

Windows 本地 `torch.cuda.is_available()` 可以是 `False`。正式训练预期在远程 Linux RTX 4090 主机上执行。

## Linux 远程训练环境

推荐远程环境：

```bash
conda create -n hair-density-train python=3.10.9 -y
conda activate hair-density-train
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

训练前先执行 CUDA smoke test：

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

## 数据集要求

将数据集放在项目根目录：

```text
FDU_HairFollicleDataset/
  Annotations/
  Images/
  ImageSets/Main/
```

代码和配置中的路径应使用相对路径，或通过 `configs/csrnet_fdu.yaml` 配置，不要硬编码本机绝对路径。

## 代码同步建议

推荐采用 Git 管理代码、单独同步数据集的工作流：

1. Windows 本地负责代码开发、数据解析、可视化、单元测试和 CPU smoke test。
2. 远程 Linux RTX 4090 主机 clone 同一个 Git 仓库，用于正式训练和长时间实验。
3. `FDU_HairFollicleDataset/` 不提交 Git，使用 `scp`、`rsync` 或其他文件同步方式单独上传到远程主机。
4. 远程训练前执行 `git pull` 或重新同步代码。
5. 远程训练产物保留在 `outputs/checkpoints/`、`outputs/logs/`、`outputs/predictions/`，重要结果再下载回 Windows 本地。
