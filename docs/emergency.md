# 突发事件临机决策系统 — 代码说明文档

## 概述

在原有静态协同搜索仿真（4架无人机 × 15个热点目标）基础上，新增基于 **Dueling DQN** 的突发事件临机决策模块。当飞行中发生热点位移、新目标涌现、禁飞区阻断、无人机故障时，DQN智能体在毫秒级内做出任务重分配决策。

---

## 文件结构

```
v3/
├── main.py                       # [修改] 新增 --emergency-mode CLI
├── emergency_dqn_agent.py        # [新建] DQN网络 + 优先经验回放
├── emergency_simulator.py        # [新建] 突发事件生成 + 状态收集 + 代价计算
├── emergency_train.py            # [新建] 训练入口
├── emergency_eval.py             # [新建] 评估对比（DQN vs GA vs 启发式）
└── emergency_demo.py             # [新建] 可视化演示
```

---

## 1. emergency_dqn_agent.py — DQN智能体

### 1.1 EmergencyDQN (Dueling DQN网络)

**架构设计**（基于 `Reference/rl/dqn/model.py`，将卷积层替换为全连接层）：

```
输入: 115维状态向量
  │
  ├─ Linear(115→256) + ReLU
  ├─ Linear(256→256) + ReLU
  ├─ Linear(256→256) + ReLU   ← 共享特征提取
  │
  ├─ 状态价值头 V(s):
  │    Linear(256→128) + ReLU → Linear(128→1)
  │
  └─ 动作优势头 A(s,a):
       Linear(256→128) + ReLU → Linear(128→60)

输出: Q(s,a) = V(s) + [A(s,a) - mean(A(s,a))]
       形状 (batch, 60)，reshape为 (4 UAVs × 15 targets)
```

**设计要点**：
- 60个输出对应每对 (UAV, target) 的Q值，即"无人机k搜索目标j的期望代价"
- Dueling架构将状态价值与动作优势分离：V(s)评估当前局势有多糟糕，A(s,a)评估每个分配决策的相对优劣
- 去均值操作 (`A - mean(A)`) 确保优势函数的数学可辨识性

**关键方法**：
| 方法 | 功能 |
|------|------|
| `forward(state)` | 返回60维Q值向量 |
| `get_q_matrix(state)` | 返回 (N, 4, 15) Q矩阵，便于贪心解码 |

### 1.2 PrioritizedReplayBuffer (优先经验回放)

**数据结构**：二叉线段树（1-indexed），同时维护 `priority_sum` 和 `priority_min`。

**优先采样原理**：
```
采样概率 P(i) = p_i^α / Σ_j p_j^α
重要性权重 w_i = (1/N × 1/P(i))^β / max_j(w_j)

其中:
  p_i = |TD_error| + ε   (优先级)
  α = 0.6               (优先级指数，0=均匀)
  β: 0.4 → 1.0          (偏差修正指数，训练后期逐渐增大)
```

**存储格式**（适配向量状态而非图像）：
```python
self.data = {
    'state':      (capacity, 115),   # float32
    'action':     (capacity,),       # int32 (0~59)
    'reward':     (capacity,),       # float32
    'next_state': (capacity, 115),   # float32
    'done':       (capacity,),       # bool
}
```

### 1.3 q_matrix_to_assignment (贪心解码)

将Q值矩阵解码为具体的分配方案。算法流程：

```
1. 屏蔽不可用的 (UAV, target) 配对：
   - 故障UAV → 对应行置 -∞
   - 已完成目标 → 对应列置 -∞

2. 贪心迭代（直到所有目标被分配或无可选配对）：
   a. 找到Q值最大的 (uav_idx, tgt_idx) 配对
   b. 若最大Q值 ≤ -1e9，终止（全部屏蔽）
   c. 将 tgt_idx 加入 routes[uav_idx]
   d. 将该目标列置 -∞（已分配，不可再选）
   e. UAV行不屏蔽（一架UAV可接收多个目标）

3. 返回 routes: [[UAV0目标列表], [UAV1目标列表], ...]
```

### 1.4 random_assignment (探索策略)

ε-greedy训练时，以概率ε随机生成有效分配方案：将剩余目标随机洗牌后均匀分发给活跃UAV。

---

## 2. emergency_simulator.py — 突发事件仿真器

### 2.1 整体架构

```
EmergencySimulator
├── 预计算: _precompute_spiral_costs()     ← 缓存螺旋搜索代价
├── 场景生成:
│   ├── _generate_s1()   热点位置突变
│   ├── _generate_s2()   突发新热点
│   ├── _generate_s3()   禁飞区阻断
│   └── _generate_s4()   无人机故障
├── 突发事件应用: apply_emergency()
├── 飞行仿真: simulate_until_emergency()
├── 状态构建: build_state_vector()
└── 代价计算:
    ├── compute_cost_for_assignment()    ← 快速代价估算
    ├── compute_oracle_cost()            ← GA重优化（上界）
    └── compute_reward()                 ← 奖励信号
```

### 2.2 状态向量设计（115维）

总共115维，所有值归一化到 [0, 1]：

| 模块 | 维度 | 编码内容 | 归一化方式 |
|------|------|---------|-----------|
| UAV位置 | 4×2=8 | 每架无人机的 (x, y) km | ÷100 |
| UAV状态 | 4×1=4 | 1=活跃，0=故障 | 二值 |
| UAV进度 | 4×2=8 | 全局已完成/剩余目标比例 | 除以总数 |
| 目标剩余 | 15×1=15 | 1=仍需搜索，0=已处理 | 二值 |
| 目标位置 | 15×2=30 | 每个目标的 (x, y) km | ÷100 |
| 目标权重 | 15×1=15 | 权重值 | 已在[0,1] |
| 目标半径 | 15×1=15 | 搜索半径 km | ÷5 (最大半径) |
| 事件类型 | 1×4=4 | One-hot [S1,S2,S3,S4] | 二值 |
| 事件严重度 | 1×1=1 | 位移距离/新目标数等 | 按类型归一化 |
| 受影响目标 | 1×15=15 | 哪些目标被突发事件影响 | 二值 |

### 2.3 四种突发事件生成逻辑

**S1 — 热点位置突变**：
```python
1. 随机选1~3个目标
2. 对每个选中目标，中心偏移 (dx, dy)，dx,dy ∈ [-10, 10] km
3. 严重度 = max偏移距离 / 20
4. 清除被移动目标的螺旋代价缓存（位置变了需重算）
```

**S2 — 突发新热点**：
```python
1. 随机生成1~3个新目标
2. 新目标：随机位置 (10~90 km)、权重 (0.3~1.0)、半径 (2~4 km)
3. 临时ID从100开始（后续在代价计算中重映射为连续索引）
4. 严重度 = 新目标数 / 3
```

**S3 — 禁飞区阻断航路**：
```python
1. 找有≥2个目标的UAV路由
2. 随机选一个航段 (target_a → target_b)
3. 在航段中点附近放置禁飞区，半径3~10 km
4. 严重度固定1.0
5. 代价惩罚：若直线路径穿越禁飞区 → +5000s
```

**S4 — 无人机故障**：
```python
1. 随机选一架有剩余任务的UAV
2. 标记为故障 (active_uavs[failed] = False)
3. 其未完成的目标成为"孤儿任务"
4. 严重度固定1.0
```

### 2.4 simulate_until_emergency（飞行仿真到触发点）

**不运行完整动力学模型**，而是沿参考路径线性插值：

```python
1. 对每架UAV:
   a. 计算其参考路径总长度
   b. trigger_distance = 触发比例 × 总长度
   c. 沿路径累积距离，到达 trigger_distance 时插值位置
2. 简化假设：触发前没有目标被完全完成
```

### 2.5 compute_cost_for_assignment（快速代价计算）

这是训练和评估的**核心优化点**。每回合需计算DQN分配的代价，如果用完整飞行仿真太慢，所以：

```python
1. 将DQN分配方案的原始ID映射为连续索引（0,1,2,...）
2. 复用预计算的螺旋搜索时间（避免重复调用spiral_search_arc_exact）
3. 仅当有位置变化的目标时才重新计算螺旋代价
4. 调用 ga.calculate_raw_costs()：
   - 转场距离用直线估算（非Dubins路径）
   - 搜索时间用预计算值
5. S3禁飞区：检查直线路径是否与禁飞区圆相交 → 惩罚5000s
6. 返回: {J_max, J_sum, weighted_arrival_sum, routes, finish_times}
```

**ID重映射机制**：S2场景中新目标原始ID为100、101等，在GA染色体中会被误判为分隔符（分隔符 = 值 ≥ 目标总数）。解决方案：
```python
id_to_idx = {原始ID: 连续索引}   # 如 {0:0, 1:1, ..., 100:15, 101:16}
# 将DQN分配中的原始ID → 连续索引 → 生成染色体 → 调用GA函数
```

### 2.6 奖励函数

```python
weighted_cost = J_max + 0.1 × J_sum
reward = -weighted_cost / 5000.0         # 缩放到[-10, 10]
            ↑ 代价越小，奖励越大
```

---

## 3. emergency_train.py — 训练流程

### 3.1 训练回合结构（单步Bandit）

```
初始化:
  1. 生成热力图 prob_grid
  2. 运行GA获取基线分配 routes
  3. 生成4架无人机的参考路径 ref_paths

每个训练回合:
  1. 随机生成突发事件 (S1/S2/S3/S4 各25%概率)
  2. 应用突发事件到热点列表 → modified_hotspots
  3. 沿参考路径插值推算UAV触发位置 → uav_positions
  4. 构建115维状态向量 → state
  5. ε-greedy动作选择:
     - ε概率: 随机有效分配
     - 1-ε概率: DQN前向 → Q矩阵 → 贪心解码 → 分配方案
  6. 计算代价 → 计算奖励
  7. 存入优先回放缓存
  8. 若缓存够一批(batch=64)，采样训练:
     - 计算当前Q值 q_online(s, a)
     - 计算目标Q值 (单步bandit: target = reward，因为无下一状态)
     - Huber损失 + 重要性采样权重
     - 反向传播 + 梯度裁剪
     - 更新样本优先级
  9. 每1000步硬更新target网络
 10. epsilon衰减: 1.0 → 0.01
```

### 3.2 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 回合数 | 50000 | 推荐值，当前冒烟测试30回合 |
| Batch size | 64 | 每批次采样数 |
| 学习率 | 1e-4 | Adam优化器 |
| 折扣因子 γ | 0.99 | 单步bandit中实际不使用 |
| 回放缓存容量 | 100000 | 2的幂，简化线段树 |
| 目标网络更新 | 每1000步 | 硬更新 |
| ε初始/终止 | 1.0 → 0.01 | 指数衰减 |
| PER α / β | 0.6 / 0.4→1.0 | 优先级指数 / 偏差修正 |
| 梯度裁剪 | max_norm=1.0 | 防梯度爆炸 |

### 3.3 模型保存策略

- 每5000回合保存 `emergency_dqn_model.pt`
- 若avg_reward创新高，保存 `emergency_dqn_model_best.pt`
- 训练历史保存为 `training_history.npz`（含rewards和losses序列）

---

## 4. emergency_eval.py — 评估对比

### 4.1 四种对比方法

| 方法 | 原理 | 计算量 |
|------|------|--------|
| **DQN** | 训练好的网络前向 + 贪心解码 | 低 (ms级) |
| **GA重优化** | 在剩余问题上重新运行GA (pop=80, gen=100) | 高 (秒级) |
| **最近邻** | 每个目标分配给距离最近的活跃UAV，按权重降序处理 | 极低 |
| **按权重贪心** | 最高权重目标分配给累积距离最短的UAV | 极低 |

### 4.2 评估指标

| 指标 | 含义 | 越低越好 |
|------|------|---------|
| J_max | 任务完成时间（木桶短板） | ✓ |
| J_sum | 四架UAV总飞行时间 | ✓ |
| weighted_arrival_sum | 按权重加权的到达时间（高危目标惩罚重） | ✓ |
| vsGA | J_max / GA_J_max | ✓ (<1表示优于GA) |
| 决策时间 | 从状态到分配方案的耗时 | ✓ |

### 4.3 输出产物

- 控制台对比表格
- `emergency_eval_comparison.png` — 箱线图（4方法 × 3指标）
- `emergency_eval_time.png` — 决策时间柱状图

---

## 5. emergency_demo.py — 可视化演示

### 5.1 演示流程

```
1. 运行GA获取原始计划 → baseline_routes
2. 生成4架UAV的参考路径
3. 生成/选择突发事件
4. 推算突发事件触发时各UAV位置
5. 应用突发事件 → modified_hotspots
6. DQN前向推理 → Q矩阵 → 贪心解码 → 新分配方案
7. 生成双面板对比图:
   ┌─────────────────┬─────────────────┐
   │  左: 原始GA计划   │  右: DQN重分配   │
   │  (虚线箭头)       │  (实线箭头)       │
   │  热点原始位置     │  突发事件标记     │
   │                  │  UAV触发位置(■)  │
   └─────────────────┴─────────────────┘
```

### 5.2 可视化元素

- **背景**：概率热力图（plasma色板，透明度35%）
- **热点**：红色圆（仍有待搜索）/ 灰色圆（已完成）
- **基地**：黑色三角
- **UAV触发位置**：彩色方块（■），标注UAV编号
- **禁飞区**（仅S3）：橙色半透明圆
- **原计划路径**：彩色虚线箭头
- **DQN重分配路径**：彩色实线箭头

---

## 6. main.py — CLI集成

### 6.1 新增参数

```bash
--emergency-mode {none,train,eval,demo}
    none  : 原有静态仿真（默认，不影响原有功能）
    train : 训练DQN智能体
    eval  : 评估对比四种方法
    demo  : 单个突发事件可视化演示

--emergency-type {S1,S2,S3,S4,random}   # demo模式的事件类型
--emergency-episodes N                   # train模式的训练回合数
--emergency-eval-scenarios N             # eval模式的测试场景数
--model-path PATH                        # DQN模型文件路径
```

### 6.2 调用示例

```bash
# 训练（推荐5000+回合）
python main.py --emergency-mode train --emergency-episodes 5000

# 评估
python main.py --emergency-mode eval --emergency-eval-scenarios 50

# 可视化无人机故障场景
python main.py --emergency-mode demo --emergency-type S4

# 原有静态仿真（不受影响）
python main.py --dynamic-plot
```

### 6.3 代码分发逻辑（main() 函数）

```python
if emergency_mode == 'train':
    导入 emergency_train → 训练 → return (跳过原有流程)
if emergency_mode == 'eval':
    导入 emergency_eval → 评估 → return
if emergency_mode == 'demo':
    导入 emergency_demo → 演示 → return
# 否则执行原有仿真流程
```

---

## 7. 关键设计决策与权衡

### 7.1 为什么用单步Bandit而非多步MDP？

突发事件需要**一次性**重分配所有剩余任务，而非逐目标决定。单步建模：
- 简化信用分配（无需处理时序依赖）
- 训练更稳定（无bootstrapping误差累积）
- 推理速度快（一次前向即可得完整分配方案）

代价：无法学习"先搜A再搜B"的**顺序**决策，但对重分配场景影响小（顺序可由螺旋路径的自然连接关系隐式决定）。

### 7.2 为什么用成对Q值而非输出完整排列？

排列空间：18个基因（15目标+3分隔符）→ 18! ≈ 6.4×10^15 种可能  
Q值方案：4×15 = 60个标量 + 贪心解码 → O(N²) 解码

前者的动作空间对DQN来说太大，后者将组合优化问题转化为Q值估计 + 后处理。

### 7.3 为什么优先经验回放？

突发事件场景中，不同紧急类型的样本难易程度差异大。PER让高TD误差的样本（即DQN判断很差的场景）被更频繁地重放，加速收敛。

### 7.4 代价计算的速度优化链

```
完整飞行仿真 (>1s/次)
  → calculate_raw_costs (直线+预计算螺旋 ≈ 0.001s/次)
     → 缓存螺旋时间避免重复计算
        → 仅S1/S2有位置变化时才重算螺旋
```

最终训练每回合耗时：GA初始化 ≈ 10-30s → 训练阶段 ≈ 0.1s/回合（S3/S4类型）。

---

## 8. 数据流总览

```
                      ┌──────────────────────┐
                      │     main.py (CLI)     │
                      └──────┬───────┬───────┘
                             │       │
              ┌──────────────┘       └──────────────┐
              ▼                                      ▼
   ┌─────────────────────┐              ┌─────────────────────┐
   │ emergency_train.py  │              │ emergency_demo.py   │
   │                     │              │                     │
   │  for episode loop:  │              │  单次突发事件 →      │
   │   1.生成突发事件     │              │  DQN推理 →          │
   │   2.收集状态         │              │  可视化对比          │
   │   3.DQN前向 → 分配   │              └─────────────────────┘
   │   4.计算代价/奖励    │
   │   5.训练 → 更新权重  │
   └──┬──────────────────┘
      │
      ├── emergency_dqn_agent.py
      │   ├── EmergencyDQN (网络)
      │   ├── PrioritizedReplayBuffer (缓存)
      │   ├── q_matrix_to_assignment (解码)
      │   └── random_assignment (探索)
      │
      └── emergency_simulator.py
          ├── build_state_vector (115维)
          ├── EmergencySimulator (场景+代价)
          │   ├── _generate_s1/s2/s3/s4
          │   ├── apply_emergency
          │   ├── simulate_until_emergency
          │   ├── compute_cost_for_assignment
          │   └── compute_oracle_cost
          └── routes_to_chromosome (编码转换)

   复用现有模块:
   ├── heatmap.py              (generate_controlled_prob_field)
   ├── multi_uav_allocation_ga.py  (init_ga_env, run_ga, calculate_raw_costs)
   ├── spiral_search_generator.py  (spiral_search_arc_exact)
   ├── dubins_path_planner.py      (dubins_curve)
   └── uav_navi.py                 (uav_navi_traverse)
```
