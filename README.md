# Dueling DQN Emergency Replanning for Multi-UAV Search

基于 Dueling DQN 的多无人机协同搜索临机决策项目。系统面向搜索区域位置变化、新增搜索区域、突发禁飞区、无人机故障等四类突发场景，在已有任务分配基础上进行快速重分配，并与 GA 重优化、规则策略、距离贪心和权重贪心等基线方法进行对比评估。

English summary: A Dueling DQN based emergency replanning framework for cooperative multi-UAV search under dynamic mission events.

## Features

- Dueling DQN 临机决策：支持轮询式多步 MDP 分配剩余搜索任务。
- 四类突发事件：`S1` 区域位移、`S2` 新增区域、`S3` 禁飞区、`S4` 无人机故障。
- 统一物理评价：训练、演示和评估共用航程、禁飞区、覆盖率和到达时间口径。
- 多方法对比评估：`DQN`、`GA_Replan`、`Rigid_Rule`、`Distance_Greedy`、`Weight_Greedy`。
- 论文级可视化：支持临机场景示意图、连续轨迹 demo、雷达图、帕累托气泡图和提琴散点图。

## Project Structure

```text
core/        路径规划、螺旋搜索、遗传算法和底层导航工具
emergency/   DQN 智能体、突发事件仿真器和统一工具函数
scripts/     训练、评估、演示和绘图入口
tests/       最小单元测试与工程健康检查
docs/        精选设计文档、架构说明和结果分析
outputs/     本地运行生成的模型、日志、图表和评估结果，默认不进入 Git
```

## Installation

建议使用 Python 3.10 或更高版本。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

在 Windows 上也可以使用 `py` 启动器：

```bash
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
```

## Training

从头训练 80000 回合：

```bash
python scripts/train.py --episodes 80000
```

从最近 checkpoint 自动续训：

```bash
python scripts/train.py --episodes 80000 --resume auto
```

常用训练参数：

```bash
python scripts/train.py ^
  --episodes 80000 ^
  --batch-size 128 ^
  --buffer-capacity 200000 ^
  --model-path outputs/models/emergency_dqn_model.pt
```

训练输出默认写入 `outputs/models/` 和 `outputs/checkpoints/`。这些文件默认不提交到 Git。

## Evaluation

在平衡的 `S1/S2/S3/S4` 场景集上评估五种方法：

```bash
python scripts/eval.py --scenarios-per-type 50 --seed 42
```

快速冒烟测试：

```bash
python scripts/eval.py --scenarios-per-type 1 --seed 42 --no-plots
```

默认输出目录为 `outputs/eval/`，包括：

- `eval_detail.csv`
- `eval_summary.csv`
- `eval_summary_by_type.csv`
- `eval_summary.json`
- 功能成功率、雷达图、帕累托图、提琴散点图和核心指标表格图

## Demo and Figures

运行单类临机决策 demo：

```bash
python scripts/demo.py --type S1 --no-show
```

一次性生成四类连续轨迹 demo：

```bash
python scripts/demo.py --type all --no-show
```

只生成突发场景示意图，不绘制实际轨迹：

```bash
python scripts/demo_emergency_only.py --type all
```

训练曲线可视化：

```bash
python scripts/plot_training.py --csv outputs/models/emergency_dqn_model_log.csv
```

## Model Weights and Results

本仓库不直接跟踪模型权重、训练日志、评估 CSV 或生成图像。推荐做法是：

- 使用上述命令在本地复现实验结果。
- 如需共享已训练权重，将 `outputs/models/emergency_dqn_model.pt` 和 `outputs/models/emergency_dqn_model_best.pt` 上传为 GitHub Release 附件。
- 如需共享论文插图，将精选图片放入 Release 或单独的论文材料仓库。

## Tests

安装测试依赖后运行：

```bash
python -m pytest
```

没有安装 `pytest` 时，可先运行基础语法检查：

```bash
python -m py_compile core/*.py emergency/*.py scripts/*.py
```

## Notes

- 当前项目以研究复现和论文实验为目标，未承诺实时飞控部署能力。
- 仿真中的航程、搜索覆盖和禁飞区绕飞由统一代价函数核算。
- 公开仓库不包含个人论文、PPT、训练中间 checkpoint 或本地 IDE/agent 配置。
