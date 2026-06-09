# Documentation Index

本目录保留项目公开发布所需的精选设计、架构和结果分析文档。过程性调试记录、个人论文/PPT、训练日志和生成式 HTML 不纳入版本控制。

## Recommended Reading Order

1. `architecture/DQN_HANDOFF.md`：Dueling DQN 训练与决策逻辑交接说明。
2. `architecture/emergency.md`：突发事件建模与仿真流程。
3. `architecture/simulator_and_utils.md`：仿真器与通用工具函数说明。
4. `design/scenario_design.md`：S1/S2/S3/S4 场景设计。
5. `design/GA_constrained.md`：GA 约束分配和代价函数设计。
6. `result_analysis.md`：论文结果分析文本草稿。

## Common Commands

```bash
python scripts/train.py --episodes 80000
python scripts/eval.py --scenarios-per-type 50 --seed 42
python scripts/demo.py --type all --no-show
python scripts/demo_emergency_only.py --type all
```

运行产物默认写入 `outputs/`，模型权重和评估结果不直接提交到 Git。
