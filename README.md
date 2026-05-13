# 多无人机协同搜索 — DQN突发事件临机决策

基于 Dueling DQN 的多无人机协同搜索系统，支持 V2 轮询式 MDP 决策。

## 安装

```bash
pip install -r requirements.txt
```

## 常用命令

```bash
python scripts/train.py --episodes 100000                 # 从头训练
python scripts/train.py --episodes 100000 --resume auto    # 续训
python scripts/demo.py --type S1                           # 单场景演示 (S1/S2/S3/S4/random)
python scripts/eval.py --scenarios 100                     # 5方法对比评估
python scripts/plot_training.py                            # 训练曲线
```
