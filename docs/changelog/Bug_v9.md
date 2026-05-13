### 项目结构调整

我对项目结构进行了调整，调整后结果如下：

```
v3/
├── pyproject.toml              # [新增]
├── README.md                   # [新增]
├── requirements.txt
├── CLAUDE.md                   # [更新路径描述]
├── .gitignore                  # [更新]
├── styles.qss
│
├── core/                       # 核心算法 (不变)
│   └── ... (6 py)
├── emergency/                  # DQN模块 (仅保留库文件)
│   ├── agent.py
│   ├── simulator.py
│   └── utils.py
│
├── scripts/                    # [新增] 入口脚本
│   ├── main.py                 # [移入] GUI
│   ├── train.py                # [移入]
│   ├── demo.py                 # [移入]
│   ├── eval.py                 # [移入]
│   └── plot_training.py        # [移入]
│
├── docs/                       # 精简后
│   ├── readme.md
│   ├── architecture/           # [新增]
│   ├── design/                 # [新增]
│   └── changelog/              # [新增] 归档 Bug/supplement 系列
│
├── outputs/                    # 统一输出 (.gitignore)
│   ├── models/                 # [移入] 原 models/
│   ├── checkpoints/            # [移入] 原 checkpoints/
│   ├── figures/                # [移入] 原 figure/
│   └── eval/                   # [移入] 原 outputs/
│
├── tests/                      # [新增]
│   └── __init__.py
│
└── .claude/ + .bkit/ + .vscode/
```



### 问题描述

之前的训练结果其实已经差不多了，但是不知道调错了什么东西。

建议你先关注demo.py中有无明显错误，然后再关注其他部分代码有无错误，因为之前结果明明差不多了。

导致现在的情况：从头训练了50000回合，得到模型结果进行demo测试，发现bug：

- 仅在 S1 中出现航程超限！S2/3/4均未出现航程超限；且有一架飞机在突发情况发生后停止飞行了，明明航程余量还有很多！

  ![emergency_demo_S1_ep50000](C:\Users\dell\Desktop\FQ\simulation\v3\outputs\eval\emergency_demo_S1_ep50000.png)

- loss未收敛

  ![training_metrics](C:\Users\dell\Desktop\FQ\simulation\v3\outputs\training_metrics.png)



