### 运行demo的效果展示与问题_v5

##### 要求

结合以下对demo效果指出的问题，分析是由于现有代码（即最近给你发的那些代码，现有指出错误均完成修改）的bug造成的问题还是训练不足（现在demo是用训练6000回合的模型测试的），或者是其他情况。

如果分析是代码有bug，则思考如何更正。

##### S1描述

![emergency_demo_S1_ep5973](C:\Users\dell\Desktop\FQ\simulation\v3\outputs\emergency_demo_S1_ep5973.png)

- UAV 3超限没放弃

##### S2描述

![emergency_demo_S2_ep5973](C:\Users\dell\Desktop\FQ\simulation\v3\outputs\emergency_demo_S2_ep5973.png)

- UAV 3 为什么超限了啊，明明它的搜索目标及序列没任何改变，原GA不超限而现在超限了

##### S3描述

![emergency_demo_S3_ep5973](C:\Users\dell\Desktop\FQ\simulation\v3\outputs\emergency_demo_S3_ep5973.png)

- UAV 2剩余很多航程没使用，理想的情况是它规划绕行曲线去搜索其他区域啊，但是它搜索完8号区域就不行动了
- 如此也导致UAV 4负责了很多区域，最终航程超限，为啥也不放弃

##### S4描述

![emergency_demo_S4_ep5973](C:\Users\dell\Desktop\FQ\simulation\v3\outputs\emergency_demo_S4_ep5973.png)

- 6号区域和13号区域放弃的是否太轻易了，最终结束UAV 4仅196km，应该还能完成至少1个区域的搜索吧