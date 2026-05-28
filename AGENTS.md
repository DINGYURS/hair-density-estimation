# AGENTS.md

这是 agent 进入本仓库后应优先阅读的入口文件。它只保留高频执行规则和文档导航；更详细的产品、技术和阶段计划内容放在 `docs/project/` 下，按需读取。

## 渐进式披露

推荐阅读顺序：

1. `AGENTS.md`：始终生效的执行规则、项目边界和当前约束。
2. `docs/project/PRD.md`：需要判断功能范围、非目标、用户价值或验收标准时阅读。
3. `docs/project/TECH_DESIGN.md`：需要实现模块、设计接口、确认数据流或训练评估细节时阅读。
4. `docs/project/PROJECT_PLAN.md`：需要阶段拆解、时间安排、学习路线和风险处理时阅读。

`docs/project/` 中的文档是项目规划和设计来源。不要把其中的大段内容复制到代码注释或 README 中，除非用户明确要求整理面向读者的文档。

## 当前项目边界

本项目当前是基于 `FDU_HairFollicleDataset` 的研究原型，不是生产级发量密度检测系统。

当前目标：

- 解析 FDU VOC XML 标注。
- 修正并过滤 bbox 标注问题。
- 将 bbox 中心点转换为高斯密度图。
- 训练 CSRNet baseline。
- 用 MAE/RMSE 评估计数误差。
- 输出密度热力图和原图叠加可视化。

不要声称当前输出具备真实业务 `根/cm^2` 精度。应使用以下表述：

- “毛囊/标注目标计数 proxy”
- “相对密度热力图”
- “基于 FDU bbox 中心点的密度估计 baseline”

## 已知数据事实

- 数据集路径：`FDU_HairFollicleDataset/`
- 数据结构：`Annotations/`、`Images/`、`ImageSets/Main/`
- 图片数量：1652
- XML 标注数量：1652
- 数据划分：train 992，val 330，test 330
- 图片分辨率：全部 1280 x 1024
- 总标注目标数：20695
- 平均每图目标数：12.53
- 每图目标数范围：5 到 28
- 类别分布：`premium` 13728，`single` 4718，`undersize` 1492，`abnormal` 757
- 已知 bbox 坐标顺序异常：3119 个
- 已知 min/max 归一化后仍无效 bbox：93 个
- train、val、test 无交叉，合计覆盖全部 1652 张图

## 环境事实

Windows 本地环境已初始化：

- Conda 环境：`hair-density`
- Python：3.10.20
- PyTorch：2.4.1 CPU
- OpenCV：4.13.0
- NumPy：2.0.1
- Pillow：12.2.0
- 本地 `torch.cuda.is_available()` 为 `False` 属于预期

远程 Linux RTX 4090 训练环境尚未在当前会话中验证。已知远程信息：

- CUDA Toolkit：11.8
- NVIDIA Driver：525.147.05
- `nvidia-smi` CUDA Version：12.0
- Python：3.10.9
- 现有 PyTorch：`2.4.1+cu121`

远程训练前必须先执行 CUDA smoke test。不要在 smoke test 通过前盲目重装远程 PyTorch。

## 工程规则

- 所有路径使用 `pathlib.Path`。
- 不在代码里硬编码 Windows 盘符、远程用户目录或绝对数据路径。
- 数据集和生成产物不提交 Git。
- PyTorch 不写入 `requirements.txt`，因为本地 CPU 和远程 CUDA 构建不同。
- 生成文件统一写入 `outputs/`。
- 脚本应支持 `--config configs/csrnet_fdu.yaml`。
- Windows 本地只负责开发、解析、可视化、测试和 CPU smoke test。
- Linux RTX 4090 负责正式训练和长实验。

## 标注处理规则

解析 bbox 时必须先修正坐标顺序：

```text
x1 = min(xmin, xmax)
y1 = min(ymin, ymax)
x2 = max(xmin, xmax)
y2 = max(ymin, ymax)
```

修正和图像边界裁剪后仍无效的 bbox 必须过滤并记录。

密度图阶段使用 bbox 中心点作为点标注 proxy。

## 推荐实施顺序

1. `src/utils/voc_parser.py`
2. `tools/analyze_fdu.py`
3. `tools/visualize_annotations.py`
4. `src/utils/density_map.py`
5. `tools/generate_density_preview.py`
6. `src/datasets/fdu_dataset.py`
7. `src/models/csrnet.py`
8. `train.py`
9. `eval.py`
10. `infer.py`

在 baseline 闭环跑通前，不优先做 REST API、小程序对接、生产部署、复杂检测框架或 Transformer 模型。

## 验证要求

优先用小范围验证降低风险：

- Parser：验证 bbox 修正、过滤和数据统计。
- 可视化：至少生成 30 张标注预览图。
- 密度图：随机 100 张检查积分与目标数关系。
- Dataset：本地 CPU DataLoader smoke test。
- 模型：验证 `[B, 3, 512, 512] -> [B, 1, 64, 64]`。
- 训练：本地 CPU 跑 1 个 mini-batch，再到远程 GPU 跑 1 个 epoch。
- 评估：输出 MAE、RMSE 和逐图预测 CSV。
