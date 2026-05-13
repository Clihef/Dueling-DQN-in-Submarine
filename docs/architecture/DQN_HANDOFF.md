# DQN 突发事件临机决策 — 实验交接文档

> 最后更新：2026-05-10 | 对应代码版本：STATE_DIM=38, MAX_RANGE=350km

---

## 1. MDP 基础设定

### 1.1 场景背景

4 架 UAV 从基地 (0,0) 出发，协作搜索 100×100km 区域内的 15 个热点目标。突发事件在任务执行到 30%-70% 时触发，破坏原 GA 基线计划。DQN 需在突发事件后逐目标做出临机决策。

四种突发事件：
| 类型 | 描述 | 影响 |
|------|------|------|
| S1 | 1-3 个已有热点位置突变 (位移 ≤10km) | 原计划航路点偏移 |
| S2 | 1-3 个新热点出现 | 新增搜索目标 |
| S3 | 一处禁飞区阻断航路 (半径 3-8km) | 需绕飞 |
| S4 | 一架 UAV 故障 | 其任务需重分配 |

### 1.2 状态空间 (38 维)

```
UAV Block — 24 dims
  0-7    UAV 位置 (4 UAVs × x,y)，归一化 /100
  8-11   UAV 活跃状态 (0/1)，S4 故障 UAV→0
  12-15  UAV 到当前决策目标的距离，归一化 /141
  16-19  UAV 已分配路由长度 (目标数)，归一化 /15
  20-23  UAV 航程利用率 consumed_ranges/MAX_RANGE_KM ← v2 新增

Target Block — 5 dims
  24-25  当前目标 (x,y)，归一化 /100
  26     目标权重 weight [0,1]
  27     目标半径 radius_km，归一化 /5
  28     是否新目标 (S2: ID≥100→1)

Emergency Block — 7 dims
  29-32  事件类型 one-hot [S1,S2,S3,S4]
  33-35  禁飞区 (cx,cy,radius) — 仅 S3 非零，归一化 /100

Sequence Block — 2 dims
  36     剩余决策比例 remain/total
  37     剩余目标平均权重
```

### 1.3 动作空间 (5 个离散动作)

| 动作 | 含义 | 屏蔽条件 |
|------|------|----------|
| 0 | 放弃当前目标 | 无（v2：所有场景可用） |
| 1-4 | 指派给 UAV1-UAV4 | S4：故障 UAV 对应动作屏蔽 |

### 1.4 MDP 结构

**多步序列决策**：每个突发事件产生 n 个受影响目标，DQN 逐目标决策。前 n-1 步为中间步 (done=False)，第 n 步为终端步 (done=True)。中间步和终端步均获得即时奖励。

---

## 2. 奖励设计演进史（核心）

### 2.1 V0 — 纯终端奖励 (已废弃)

**设计**：
```python
dqn_val = J_max + 0.1 * J_sum
abandon_penalty = sum(target.weight) * 5000
reward = (baseline_val - dqn_val - abandon_penalty) / baseline_val
# 中间步: r = 0, done = False
# 终端步: r = reward, done = True
```

**失败原因**：
- **稀疏奖励**：n 步决策只有终端一步收到非零信号。对 1-3 步的场景尚可，但对 5 步以上的场景，信用分配极差（哪一步做对了？哪一步做错了？DQN 不知道）。
- **S2/S4 强制接受**：动作 0（放弃）被屏蔽，DQN 被迫接受所有目标。在 S4 场景（3-4 个目标重新分配），导致某 UAV 严重超载。
- **S3 惩罚失真**：禁飞区穿越使用 `2.5*r/0.05` 的工程估算，不是真实绕飞距离。
- **放弃代价过低**：`weight*5000` 在 baseline_val ~25000 的分母下仅有 ~0.06 的影响，DQN 学会"放弃保平安"。

### 2.2 V1 — 两段式奖励 + 航程约束初版 (已废弃)

**设计**：
```python
# 中间步
if action == 0:
    r_step = -target.weight * 0.5        # ← 关键问题：太小
else:
    extra_km = incremental_dist + spiral_km
    range_after = consumed_ranges[k] + extra_km
    range_pressure = clamp((range_after - 120) / 30, 0, 1)
    r_step = -(extra_km / 0.05 / 5000 + range_pressure * 0.5)

# 终端步
fatal_penalty = range_violations * 0.2 * max(dqn_val, 1.0)
abandon_penalty = sum(weight) * 5000
reward = (baseline_val - dqn_val - fatal_penalty - abandon_penalty) / baseline_val
```

MAX_RANGE=150km, SOFT_RANGE=120km.

**失败原因**：
- **航程重复计算 Bug**：`compute_route_distance_km` 从 base(0,0) 起算剩余路程，又加上 `consumed_ranges`（base→触发点），base→触发这段被算了两次。导致 `total_ranges` 膨胀 ~50%，**几乎所有 episode 都有 3-4 架 UAV 超限**（range_violations=4）。这个 bug 掩盖了真实航程数据，让 150km 看起来严重不足，实际上 GA 基线单架 ~300km。
- **放弃惩罚仍然过低**：中间步 `weight*0.5` = -0.15（对 w=0.3），而接受代价约 -0.13 到 -0.63。放弃几乎总是更便宜，导致 DQN 学习到"放弃是最优策略"——尤其在 S2/S4 场景下放弃几乎所有目标。
- **150km 约束过紧**：即使修复 bug，GA 基线已用 ~300km，150km 完全不可能满足。

### 2.3 V2 — 当前运行版本

**设计**：
```python
# ===== 中间步 (done=False) =====
def compute_step_reward(action, target, uav_routes, uav_positions,
                         consumed_ranges, id_to_hotspot):
    NORM = 5000.0  # 归一化因子 (~150km 飞行时间)

    if action == 0:
        return -target.get('weight', 0.3) * 2.0   # ← 提升至 weight×2.0

    uav_idx = action - 1
    # 增量飞行距离：UAV最后一个目标 → 当前目标 (或 UAV位置 → 目标)
    incremental_dist = distance(last_waypoint, target)
    spiral_km = target.spiral_dist / 1000.0
    extra_km = incremental_dist + spiral_km
    extra_time = extra_km / 0.05  # UAV_VELOCITY_KM_S

    # 航程压力：0 when <80%MAX, 线性增长到 1.0 at 100%MAX
    range_after = consumed_ranges[uav_idx] + extra_km
    range_pressure = clamp((range_after - SOFT_RANGE) / (MAX - SOFT_RANGE), 0, 1)

    return -(extra_time / NORM + range_pressure * 0.5)

# ===== 终端步 (done=True) =====
def compute_reward(dqn_cost, baseline_val, abandoned_targets):
    dqn_val = dqn_cost['J_max'] + 0.1 * dqn_cost['J_sum']

    # 致命越限惩罚：每架超 MAX_RANGE 的 UAV 扣 20% dqn_val
    fatal_penalty = range_violations * 0.2 * max(dqn_val, 1.0)

    # 放弃惩罚：权重越高代价越大
    abandon_penalty = sum(target.weight) * 15000.0  # ← 提升至 15000

    reward = (baseline_val - dqn_val - fatal_penalty - abandon_penalty) / baseline_val
    return clip(reward, -10.0, 10.0)
```

**当前参数**：`MAX_RANGE_KM=350, SOFT_RANGE_KM=280` (用户调整为 GA 基线 ~300km 留 50km 余量)

**设计意图**：

| 场景 | 接收代价 (w=0.3) | 放弃代价 (w=0.3) | DQN 倾向 |
|------|-------------------|-------------------|----------|
| 航程充裕 (range_pressure=0) | ~ -0.13 | **-0.60** | 接收 ✓ |
| 航程紧张 (range_pressure=0.5) | ~ -0.38 | **-0.60** | 接收 ✓ |
| 航程极限 (range_pressure=1.0) | ~ -0.63 | **-0.60** | 接近持平，由终端 reward 决定 |

高权重目标 (w=1.0)：放弃 = -2.0 vs 接收 ≤ -0.63 → **绝不放弃**。低权重目标 (w=0.2)：放弃 = -0.4 vs 接收 ≤ -0.63 → 仅在高压时考虑。

### 2.4 奖励函数设计原则总结

1. **中间步即时信号**：每个决策立刻得到增量代价反馈，解决 V0 的稀疏奖励问题。同时保持 done=False 维持多步 MDP 的长期规划能力（避免退化为 Contextual Bandit）。
2. **航程压力梯度**：从 80%MAX 开始线性增长，让 DQN 在"无可挽回之前"就感知压力，而非到了 100% 才被 fatal_penalty 惩罚。
3. **放弃代价校准**：放弃应比"无压力接受"贵，但比"高压接受"便宜。weight×2.0 在当前归一化框架下实现了这个梯度。
4. **终端全局评价**：最后一步仍保留 fatal_penalty，确保 DQN 不会通过贪心放弃来优化中间步奖励。

---

## 3. 超参数与网络现状

### 3.1 网络结构

```
Dueling DQN (PyTorch)
├── Shared Feature: Linear(38,128) → ReLU × 3
├── Value Head V(s):  Linear(128,64) → ReLU → Linear(64,1)
└── Advantage Head A(s,a): Linear(128,64) → ReLU → Linear(64,5)
Q(s,a) = V(s) + [A(s,a) − mean(A)]
```

总参数量：~38×128 + 3×128² + 128×64×2 + 64×1 + 64×5 ≈ 70K

### 3.2 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| learning_rate | 1×10⁻⁴ | Adam optimizer |
| batch_size | 64 | |
| gamma | 0.95 | 有限视野 MDP，不打折过远 |
| epsilon | 1.0 → 0.01 | exponential decay over 90% episodes |
| target_update_freq | 1000 steps | Target Network 硬更新 |
| replay_buffer_capacity | 100,000 | PER 优先经验回放 |
| PER alpha | 0.6 | 优先级指数 |
| PER beta | 0.4 → 1.0 | 重要性采样修正，线性增长 |
| save_interval | 10 episodes | checkpoint + model |
| oracle_interval | 10 episodes | GA oracle 计算频率 |
| grad_clip | 1.0 | 梯度裁剪 |

### 3.3 仿真环境参数

| 参数 | 值 |
|------|-----|
| 区域 | 100×100 km, dx=0.1km |
| UAV 速度 | 50 m/s (0.05 km/s) |
| UAV 数量 | 4 |
| 热点目标数 | 15 (固定) |
| MAX_RANGE_KM | 350 km |
| SOFT_RANGE_KM | 280 km (80%) |
| 触发时间 | 任务进度的 30%-70% 随机 |
| GA 基线权重 | (0.5, 0.3, 0.4) |

### 3.4 关键常量

```python
# emergency/utils.py
STATE_DIM = 38
NUM_UAVS = 4
NUM_ACTIONS = 5
MAX_RANGE_KM = 350.0
SOFT_RANGE_KM = 280.0
UAV_VELOCITY_KM_S = 0.05
NORM = 5000.0  # step reward 归一化因子
```

---

## 4. 当前痛点与下一步实验方向

### 4.1 已知问题

1. **S2/S4 放弃行为待观察**：V2 将放弃惩罚大幅提升后，尚未进行充分训练验证。旧模型 (16726 eps) 是在 V0 框架下训练的，学到了"放弃是最优"的错误策略。需要用 V2 从头训练足够回合后评估。
2. **航程约束的实际效果未知**：MAX_RANGE=350km 下，GA 基线 ~300km (82-87%)。该航程约束是否合理，是否仍然需要结合运行结果调整？突发事件后是否仍有大量 UAV 超限，取决于 DQN 能否学会在 ~50km 余量内做战略调整。
3. **range_pressure 可能启动过晚**：SOFT=280km (80%) 意味着基线飞行时 range_pressure≈0。只有 S2/S4 追加大量航程后才会进入压力区。如果模型总是等到压力区才反应，可能来不及。可以考虑降到 70% (245km) 让压力更早出现。
4. **S3 绕飞代价是否足够被感知**：绕飞额外距离通常 5-15km，在 NORM=5000 归一化下信号微弱 (~0.001-0.003)。DQN 可能学不会"主动绕飞 vs 硬闯"的区别，因为绕飞的中间步代价太低。
5. **负载均衡可能冲突**：`balance_routes()` 在终端步才执行，但 DQN 中间步基于未均衡的路由做决策。中间步的状态（UAV route lengths）可能在均衡后大幅变化，导致 Q 值不准确。

### 4.2 建议下一步实验

| 优先级 | 方向 | 具体操作 |
|--------|------|----------|
| **P0** | 充分训练 V2 | 从头训练 ≥20000 回合，观察 reward/J_max/range_violations 曲线是否收敛 |
| **P1** | 调整 SOFT_RANGE | 若训练后 range_violations 仍高，降低 SOFT 到 245km (70%)，让压力提前 |
| **P2** | S3 绕飞信号增强 | 将 S3 绕飞代价在 step_reward 中单独加权 (×5-10)，或加到 terminal_reward |
| **P3** | 中间步状态一致性 | 每步执行轻量 nearest_neighbor_reorder 后再构建 next_state，减小均衡前后的分布偏移 |
| **P4** | eval 分类型统计 | 完善 eval.py 中按 S1/S2/S3/S4 分组的 J_max/range_violations 统计表 |

### 4.3 验证方法

```bash
# 从头训练
venv/Scripts/python -m emergency.train --episodes 20000

# 训练中检查日志
tail -20 models/emergency_dqn_model_log.csv

# 评估（5 方法对比 + 新指标）
venv/Scripts/python -m emergency.eval --scenarios 100

# 单场景 Demo（可视化航程）
venv/Scripts/python -m emergency.demo --type S2
venv/Scripts/python -m emergency.demo --type S4
```

### 4.4 成果判断标准

- **成功**：range_violations 均值 < 0.5（少于一半场景超限），J_max 接近或优于 GA Oracle
- **部分成功**：range_violations 下降趋势但未稳定，需调整 SOFT_RANGE 或放弃惩罚
- **失败**：range_violations 不下降，放弃率 >80% 或 =0%，reward 不收敛 → 需重新设计奖励权重

21.26
