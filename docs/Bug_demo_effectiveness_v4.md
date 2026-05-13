### 运行demo的效果展示与问题

##### 要求

结合以下对demo效果指出的问题，分析是由于现有代码（即最近给你发的那些代码，现有指出错误均完成修改）的bug造成的问题还是训练不足，或者是其他情况。

如果分析是代码有bug，则思考如何更正。

注：我感觉可能和这几点有关（仅是个人感觉），可以关注一下

- 首先包括get_affected_targets函数，我认为它可能造成下方出现折线来回的航程浪费，因为其按权重降序排列受影响目标，然后依次决策，需要这样按权重依次决策吗，随意顺序可以吗，或者其他因素造成的；
- 另外，再关注一下各处距离的计算是否合理，对于突发事件发生在螺旋搜索中途的情况。

##### S1描述

![emergency_demo_S1_ep10459](C:\Users\dell\Desktop\FQ\simulation\v3\outputs\emergency_demo_S1_ep10459.png)

- 图中5号区域位置变的更利于航程缩短了，但UAV 1怎么最后航程比左边原来规划的还长呢？
- UAV3为什么走了一个来回的折线，很浪费航程啊？

##### S2描述

![emergency_demo_S2_ep10459](C:\Users\dell\Desktop\FQ\simulation\v3\outputs\emergency_demo_S2_ep10459.png)

- 情况符合预期，但刚好卡了一个临界350km的值，但是为什么没学会放弃？是因为可能预估时可以满足最大航程吗

##### S3描述

![emergency_demo_S3_ep12249](C:\Users\dell\Desktop\FQ\simulation\v3\outputs\emergency_demo_S3_ep12249.png)

- UAV 1 穿越了禁飞区！严重错误，是回合数太少没学习到吗？
- UAV 1出现了来回的折线，航程浪费，且超限。

##### S4描述

![emergency_demo_S4_ep10459](C:\Users\dell\Desktop\FQ\simulation\v3\outputs\emergency_demo_S4_ep10459.png)

- 为什么UAV1和UAV4超限这么多，还没学会放弃？
- UAV4选了一个很不聪明的路径，几乎全地图折返跑，航程严重浪费于转场



### eval展示

展示结果如下，请你对展示结果分析合理性，进而思考代码部分有无bug

![emergency_eval_heatmap](C:\Users\dell\Desktop\FQ\simulation\v3\outputs\emergency_eval_heatmap.png)

![emergency_eval_comparison](C:\Users\dell\Desktop\FQ\simulation\v3\outputs\emergency_eval_comparison.png)

![emergency_eval_range](C:\Users\dell\Desktop\FQ\simulation\v3\outputs\emergency_eval_range.png)

### 训练回合曲线

展示结果如下，请你对展示结果分析合理性，进而思考代码部分有无bug

![training_constraints](C:\Users\dell\Desktop\FQ\simulation\v3\outputs\training_constraints.png)

![training_cost](C:\Users\dell\Desktop\FQ\simulation\v3\outputs\training_cost.png)

![training_metrics](C:\Users\dell\Desktop\FQ\simulation\v3\outputs\training_metrics.png)