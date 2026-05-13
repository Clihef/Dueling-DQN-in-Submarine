### epsilon衰减策略修改

`epsilon` 的衰减策略存在几个明显的逻辑漏洞和不合理之处。特别是在断点续训（Resume）**和**改变总训练回合数（num_episodes）时，会导致严重的训练不稳定。

以下是具体的缺陷分析以及修改建议：

#### 1. 存在的不合理之处

**缺陷一：断点续训时动态调整 `num_episodes` 会导致 Epsilon 发生“跳跃”（突变）**

- **现象**：代码中定义了 `explore_episodes = int(num_episodes * 0.9)`，并在循环内使用 `epsilon = 1.0 - (1.0 - epsilon_min) * (current_episode / explore_episodes)` 进行计算。
- **问题**：假设你第一次训练了 50,000 回合，训练结束时 `epsilon` 已经降到了 `0.05`。如果此时模型还没完全收敛，你想加载 Checkpoint 继续训练到 100,000 回合。在新的运行中，`explore_episodes` 变成了 90,000。当第 50,001 回合开始时，`epsilon` 会被重置为 `1.0 - 0.95 * (50001 / 90000) ≈ 0.47`。
- **后果**：原本已经收敛（或接近收敛）的模型，突然被迫进行 47% 的随机探索，这会瞬间破坏网络已经学到的策略（Policy Collapse）。

**缺陷二：Checkpoint 中加载的 `epsilon` 形同虚设**

- **现象**：在 `load_checkpoint` 函数中，代码正确地提取了保存的 `ckpt['epsilon']`。
- **问题**：但在主循环 `for ep in pbar:` 中，每一回合的末尾都用绝对公式强制覆盖了 `epsilon` 的值。加载上来的 `epsilon` 在第一步之后就被完全抛弃了。

**缺陷三：探索期占比过长（90%）**

- **现象**：前 90% 的回合都在进行线性衰减，直到最后 10% 才保持最小值。
- **问题**：对于 50,000 步的训练，前 45,000 步都在进行高频随机探索。通常 DQN 需要足够的纯“利用（Exploitation）”阶段来微调和稳定 Q 值。90% 的探索期过于保守，可能导致模型在训练结束时仍未完全收敛。

*(注：优先级回放的 `beta` 参数也存在与缺陷一完全相同的按比例计算导致的回退问题。)*

------

#### 2. 如何修改

推荐使用**指数衰减**，因为它与断点续训天然兼容，不需要记录繁琐的全局基准。

这种方法只依赖于上一步的 `epsilon`，完美继承 Checkpoint 中加载的值。

**修改步骤：**

1. 删掉循环外面的 `explore_episodes = int(num_episodes * 0.9)`。
2. 设定一个衰减率（如 `0.9998`），在循环末尾按乘法衰减。

Python

```
    # 修改前 (约在 255 行左右)
    # explore_episodes = int(num_episodes * 0.9)
    
    # 修改后：设定衰减率
    epsilon_min = 0.05
    # 假设你希望在约 15000 回合时衰减到 0.05：0.9998^15000 ≈ 0.049
    epsilon_decay = 0.9998 

    # ----------------------------------------
    # 在 for ep in pbar: 循环内部 (约在 402 行左右)
    
    # 修改前：
    # if current_episode <= explore_episodes:
    #     epsilon = 1.0 - (1.0 - epsilon_min) * (current_episode / explore_episodes)
    # else:
    #     epsilon = epsilon_min

    # 修改后：
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
```



### 优先级回放（PER）的 Beta 参数修改

代码第 362 行计算 `beta` 时也存在动态比例导致跳跃的问题：

Python

```
# 原代码
beta = beta_start + (beta_end - beta_start) * (current_episode / max(num_episodes, 1))
```

**建议修改为基于固定总步数：**

Python

```
# ========== 训练一步 ==========
if replay_buffer.size >= batch_size:
    # 设定一个绝对的退火周期，比如我们希望在前 40,000 回合让 beta 涨到 1.0
    beta_anneal_episodes = 40000.0 
    
    # 计算当前的退火进度，最高不超过 1.0
    progress = min(1.0, current_episode / beta_anneal_episodes)
    
    # 线性计算 beta
    beta = beta_start + (beta_end - beta_start) * progress
    
    samples = replay_buffer.sample(batch_size, beta)
```

关于优先经验回放（PER）中的 `beta` 参数，确实需要特别注意。在 DQN 的训练中，`beta` 的作用可能不如 `epsilon` 那么直观，但它一旦设置不合理，会直接导致网络在训练后期崩溃（Loss 爆炸）。

`beta_anneal_episodes` 设置为多少比较合适？

- **原则**：通常建议在训练总进度的 **70% 到 80%** 左右让 `beta` 达到 `1.0`。

  