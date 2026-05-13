### 运行demo的效果展示与问题_v6

##### 概述

以下是使用训练了约380000回合的模型测试结果。

##### 要求

结合以下对demo效果指出的问题，分析代码的bug，给出解决方案。

注：建议主要关注训练过程逻辑、奖励设计

- 修改 compute_route_distance_km 函数中对 actual_spiral 的计算，不要用现有根据距离与目标半径的关系、线性打折螺旋距离的逻辑，因为螺旋路径的特点，此折扣太粗糙；要求进程修改为真实计算，即用准确已经搜索该区域的螺旋长度除以该区域搜索需要的总路径长度。
- 关注_cheapest_insertion函数逻辑、关注apply_decision函数逻辑，以及关注train.py中逐目标决策循环中的MDP过程合理性
- S3情景中对get_affected_targets的定义合理吗，同时关注奖励设计中S3情况禁飞区的影响
- 上述建议中有的互相关联，修改时注意

##### S1描述

![emergency_demo_S1_ep378049](C:\Users\dell\Desktop\FQ\simulation\v3\outputs\ep378049\emergency_demo_S1_ep378049.png)

- UAV 3在突发情况后分配了新9号区域，此分配是不合理的，因为9号区域很明显离UAV 2近，且UAV 2剩余航程多
- 由于不合理的分配，UAV 3严重超限，而UAV 2航程剩余很多没利用上

##### S2描述

![emergency_demo_S2_ep378049](C:\Users\dell\Desktop\FQ\simulation\v3\outputs\ep378049\emergency_demo_S2_ep378049.png)

- 结果应该正确，UAV 2剩余航程不足以支持它搜索新出现的区域

##### S3描述

![emergency_demo_S3_ep378049](C:\Users\dell\Desktop\FQ\simulation\v3\outputs\ep378049\emergency_demo_S3_ep378049.png)

- 首先，UAV 3和 UAV 4均穿越了禁飞区，这是绝对不允许的，实际穿越禁飞区就视作坠毁的；
- 其次，UAV 4 突发后仅分配10号区域，放弃了原有的7号和14号区域（出现了在禁飞区两端），顺路的航程没利用充分；
- 如此，导致了UAV 3通过长距离转场区搜素7号和14号区域，造成了航程严重超限，而且没学会放弃。

##### S4

![emergency_demo_S4_ep378049](C:\Users\dell\Desktop\FQ\simulation\v3\outputs\ep378049\emergency_demo_S4_ep378049.png)

- UAV 4突发后额外分配了区域12，但是明明有更好的选择，7号区域明显更近且权重大，航程也完全足够，8号区域的权重更大。

  但是UAV 4却没有选择7或8，而是选择12，糟糕的选择。



### 回合收敛曲线

![training_metrics](C:\Users\dell\Desktop\FQ\simulation\v3\outputs\ep378049\training_metrics.png)

- 模型确实已经收敛了，在100000步左右绝对收敛了

![training_cost](C:\Users\dell\Desktop\FQ\simulation\v3\outputs\ep378049\training_cost.png)

- 这两个代价这样变化合理吗