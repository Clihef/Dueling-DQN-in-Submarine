# 突发事件仿真器与工具函数 — 详细说明

## 文件架构

```
emergency/
├── utils.py        # 全局常量、工具函数、状态向量、FSM（被 train/eval/demo 共享）
└── simulator.py    # EmergencySimulator 类：场景生成、飞行仿真、代价计算
```

`utils.py` 是无状态的工具层，`simulator.py` 是有状态的仿真环境。后者依赖前者。

## 目录

1. [utils.py — 全局常量](#1-utilspy--全局常量)
2. [utils.py — 路由/染色体转换](#2-utilspy--路由染色体转换)
3. [utils.py — 最近邻重排序](#3-utilspy--最近邻重排序)
4. [utils.py — DQN决策应用](#4-utilspy--dqn决策应用)
5. [utils.py — S3禁飞区绕行](#5-utilspy--s3禁飞区绕行)
6. [utils.py — 路由负载均衡](#6-utilspy--路由负载均衡)
7. [utils.py — 状态向量构建](#7-utilspy--状态向量构建)
8. [utils.py — 受影响目标提取](#8-utilspy--受影响目标提取)
9. [utils.py — FSM状态机](#9-utilspy--fsm状态机)
10. [simulator.py — EmergencySimulator 类](#10-simulatorpy--emergencysimulator-类)
11. [simulator.py — 场景生成](#11-simulatorpy--场景生成)
12. [simulator.py — 应用突发事件](#12-simulatorpy--应用突发事件)
13. [simulator.py — 飞行仿真](#13-simulatorpy--飞行仿真)
14. [simulator.py — 代价计算](#14-simulatorpy--代价计算)

---

## 1. utils.py — 全局常量

```python
NUM_UAVS = 4           # 无人机数量（固定）
NUM_TARGETS = 15        # 静态目标数量（GA染色体用）
MAX_NEW_TARGETS = 3     # S2场景最多新增目标数
STATE_DIM = 34          # DQN状态向量维度
SPAN_KM = 100.0         # 仿真区域边长 (km)
UAV_V_M_S = 50.0        # 无人机巡航速度 (m/s)
UAV_VELOCITY_KM_S = 0.05  # 无人机速度 (km/s)，用于时间/距离换算
MAX_DIAG_KM = 141.0     # 区域对角线 (sqrt(100^2 + 100^2))，用于距离归一化
MAX_RADIUS_KM = 5.0     # 最大目标半径，用于半径归一化
```

**设计要点**：
- `STATE_DIM = 34` 是 DQN 网络的输入维度。其构成见[第7节](#7-utilspy--状态向量构建)。（曾为 31 维，后续新增 3 维禁飞区信息）
- `MAX_DIAG_KM` 使用正方形区域的对角线长度，确保任意两点距离 ≤ 该值。
- `UAV_VELOCITY_KM_S`（0.05 km/s）从 `UAV_V_M_S` 换算而来，用于 GA 代价估算和 S3 绕飞代价中的时间换算。

---

## 2. utils.py — 路由/染色体转换

GA 使用"染色体"表示分配方案：目标 ID 拼接成分隔符标记的序列。

### `routes_to_chromosome(routes, num_targets, num_uavs)` (L22-31)

**作用**：将"每架UAV的目标列表"编码为单条染色体。

**编码规则**：分隔符值 = `num_targets, num_targets+1, ..., num_targets+num_uavs-2`。基因值 < `num_targets` 表示目标ID，≥ `num_targets` 表示切换到下一架UAV。

**示例**（5个目标，4架UAV）：
```python
routes = [[0,3], [1,2], [4], []]
# → chromosome = [0, 3, 5, 1, 2, 6, 4, 7]
#                  |——|  | |——|  | |  |
#                  UAV1  ^ UAV2  ^ UAV3 ^ (UAV4空)
#                        分隔符=5     分隔符=6  分隔符=7
```

- `L24`: 初始化空染色体
- `L25`: 分隔符起始值 = 目标总数
- `L26-30`: 遍历每架UAV。非首架UAV前插入分隔符，然后拼接目标列表

### `chromosome_to_routes(chrom, num_targets, num_uavs)` (L34-44)

**作用**：解码染色体回路由列表。是 `routes_to_chromosome` 的逆操作。

- `L36`: 初始化空路由列表（每架UAV一个空列表）
- `L38-43`: 遍历染色体基因。遇到分隔符（值 ≥ `num_targets`）则切换到下一架UAV，否则将目标ID加入当前UAV的路由

---

## 3. utils.py — 最近邻重排序

### `nearest_neighbor_reorder_with_hotspots(routes, uav_positions, id_to_hotspot)` (L49-91)

**作用**：DQN逐目标分配后，每架UAV的路由顺序是"分配顺序"，不一定是好的访问顺序。此函数用贪心最近邻对每架UAV的路由重排序。

**算法**（每架UAV独立执行）：
1. 从UAV当前位置出发
2. 在所有剩余目标中选择距离最近的一个
3. 将该目标从待访问列表移除，加入新路由
4. 更新"当前位置"为该目标位置
5. 重复直到所有目标排序完成

**逐行解释**：
- `L60-64`: 单目标路由无需排序，直接返回副本
- `L66-67`: `remaining` 为可变列表，`pos` 追踪虚拟位置
- `L70-89`: 主循环。`L73-81` 遍历剩余目标找最近者，`L83` 取出，`L84-87` 更新位置
- 时间复杂度：O(N²)，N ≤ 5

---

## 4. utils.py — DQN决策应用

### `apply_decision(action, target_id, routes, abandoned_list)` (L96-116)

**作用**：将DQN的单步决策（0=放弃 / 1-4=指派）原地应用到大UAV路由和放弃列表中。

**关键逻辑（L110-116）** — 分配时必须先从其他UAV移除：
```python
uav_idx = action - 1
for k in range(len(routes)):
    if target_id in routes[k]:
        routes[k].remove(target_id)  # 防止目标出现在多架UAV路由中
routes[uav_idx].append(target_id)
```

**为什么需要先移除**：S3场景中，受禁飞区影响的目标原本在某架UAV的路由中。如果DQN将其"重分配给UAV3"但未从原UAV移除，目标会同时存在于两架UAV的路由中 — 原UAV仍会飞越禁飞区去访问已被"转移"的目标。S1/S4同理。

- `L107-108`: 动作为0则追加到放弃列表
- `L110-116`: 动作为1-4则：映射 `action→uav_idx`，从所有UAV路由中删除该目标，追加到目标UAV路由

---

## 5. utils.py — S3禁飞区绕行

### `replan_around_no_fly(routes, uav_positions, no_fly_center, no_fly_radius, id_to_hotspot)` (L121-161)

**作用**：对每条UAV路由逐段检测禁飞区穿越，在穿越段插入绕行航路点。返回带绕行点的路径点序列，仅用于可视化。不修改原路由。

**算法**：
1. 对每架UAV，构建"路径点序列"：起始点为UAV当前位置，然后依次为目标位置
2. 检测每段 `points[-1] → tgt_pos` 是否穿越禁飞区（调用 `_segment_intersects_circle`）
3. 若穿越，调用 `_compute_detour_waypoint` 计算绕行点并插入
4. 追加目标位置

**演示中的可视化**：绕行点标注为黄色菱形（`marker='D'`），路径为实线绘制。

### `_segment_intersects_circle(p1, p2, cx, cy, r)` (L164-183)

线段-圆相交检测。与 `simulator._line_intersects_circle` 逻辑完全相同。解参数方程 `P(t) = p1 + t*(p2-p1)` 代入圆方程，判断 `t` 在 `[0,1]` 内是否有解。

### `_compute_detour_waypoint(p1, p2, cx, cy, r)` (L186-205)

**绕行点计算方法**：
1. 取线段中点 `(mx, my)`
2. 计算圆心到中点的方向向量 `(dx_c, dy_c)`
3. 若中点恰好在圆心上（`dist_c < 1e-9`），改用线段垂线方向
4. 沿此方向偏移到圆外距离 `r * 1.2` 处

绕行点使得可视化路径绕过禁飞区圆外围，直观展示"应如何绕飞"。

---

## 6. utils.py — 路由负载均衡

### `balance_routes(routes, uav_positions, active_uavs, id_to_hotspot)` (L210-247)

**作用**：防止某架UAV空闲而其他UAV负载过重。在最近邻重排序后调用。

**规则**：若某活跃UAV路由为空，且存在路由长度 > 1 的其他UAV，从超载UAV中选择离空载UAV最近的一个目标转移过去。最多两轮。

**场景**：S3中，DQN可能将所有受影响目标全分配给一架UAV（其余UAV路由策略不变），导致某UAV路由为空。`balance_routes` 在重排序后重新平衡负载。

**逐行解释**：
- `L222`: 最多两轮（避免无限循环）
- `L223-228`: 找空载和超载UAV
- `L231-244`: 遍历所有超载UAV的所有目标，找离空载UAV最近者
- `L245-247`: 执行转移

---

## 7. utils.py — 状态向量构建

### `build_state_vector(current_target, uav_positions, active_uavs, uav_routes, affected_targets, processed_count, emergency_info)` (L252-334)

**作用**：将突发事件瞬间的场景信息编码为34维浮点数向量，供DQN决策。

**设计原则**：聚焦"当前正在决策的这一个目标"。DQN只需回答"对于这个目标，放弃还是分配给UAV X？"。所有特征都已归一化到 [0,1] 或 [-1,1]。

**状态向量布局（34维）**：

| 区间 | 维度 | 特征 | 归一化 |
|------|------|------|--------|
| 0-7 | 8 | UAV x, y 位置 (每架2维) | /100 |
| 8-11 | 4 | UAV 活跃状态 | 0/1 |
| 12-15 | 4 | UAV 到当前目标的距离 | /141 |
| 16-19 | 4 | UAV 路由负载 | /15 |
| 20 | 1 | 当前目标 x | /100 |
| 21 | 1 | 当前目标 y | /100 |
| 22 | 1 | 当前目标权重 | 原值 [0,1] |
| 23 | 1 | 当前目标半径 | /5 |
| 24 | 1 | 是否新目标 (S2) | 0/1 (ID≥100→1) |
| 25-28 | 4 | 事件类型 one-hot | [S1,S2,S3,S4] |
| 29-31 | 3 | 禁飞区信息 (仅S3非零) | /100, /100, /100 |
| 32 | 1 | 剩余待处理比例 | 剩余/总数 |
| 33 | 1 | 剩余目标平均权重 | 原值 [0,1] |

**逐段解释**：

**UAV模块 (L272-291, 20维)**
- `L273-276`: 写入每架UAV的 x、y 坐标（除以100归一化）
- `L278-280`: 写入活跃标志。S4场景中故障UAV为0
- `L282-287`: 计算每架UAV到当前决策目标的欧氏距离。**这是DQN判断"哪个UAV最适合接这个目标"的核心特征**
- `L289-291`: 写入路由负载（已分配目标数/15）。防止DQN把太多目标堆给同一架UAV

**当前目标模块 (L293-303, 5维)**
- `L294-296`: 目标位置
- `L298`: 目标权重（直接使用，已在 [0,1] 范围）
- `L300`: 目标半径（/5归一化，最大半径5km）
- `L302`: `target_is_new` 标志。S2新目标的ID ≥ 100（由 `_generate_s2` 分配），据此判断

**事件类型 one-hot (L305-311, 4维)**
- `L306-308`: 用字典做 one-hot 编码。未知类型默认全0

**禁飞区信息 (L313-319, 3维) — 仅S3非零**
- `L314`: 仅S3场景写入真实值
- `L315-318`: 禁飞区中心 x、y（/100归一化）和半径（/100归一化）
- `L319`: `idx += 3` 无条件执行，保证后续维度索引正确。非S3场景这3维保持初始值0
- **设计理由**：早期版本缺失此信息，DQN只知道"S3发生了"但不知禁飞区在哪，无法做出"分配给哪个UAV能避开"的决策

**序列上下文 (L321-332, 2维)**
- `L322-325`: `remaining_frac` = 剩余待处理目标数/总数。让DQN知道自己处于决策序列的哪个位置
- `L327-332`: 剩余目标的平均权重。**让DQN知道"后面的目标平均有多重要"，用于容量预留决策** — 如果后面有高权重目标，当前低权重可能需要放弃以保留UAV容量

---

## 8. utils.py — 受影响目标提取

### `get_affected_targets(emergency, routes, hotspots)` (L339-387)

**作用**：从突发事件字典中提取需要DQN决策的目标列表，按权重降序排列。

**为什么按权重降序**：高权重目标先决策 → 先抢占UAV容量。如果低权重目标先分配到满载的UAV上，高权重目标就没得选了。

**各场景提取逻辑**：

- **S1 (L354-362)**：遍历 `shifts` 列表，找到对应热点，深拷贝后修改 `center_km` 为位移后的位置
- **S2 (L364-366)**：直接深拷贝 `new_targets` 列表中的所有新目标
- **S3 (L368-378)**：优先用 `affected_segment`（被阻断航段的两端目标）。如果为空（兼容旧事件格式），回退到 `affected_targets`
- **S4 (L380-384)**：提取故障UAV的 `lost_targets`（它未完成的目标）
- `L386`: 所有类型统一按权重降序排列

**注意**：调用方（train/demo/eval）会在 `simulate_until_emergency` 返回 `completed` 后过滤已完成目标：
```python
affected = [t for t in affected if t['id'] not in completed]
```
以及S1的 `shifts` 和 `affected_targets` 也会过滤：
```python
if emergency['type'] == 'S1':
    emergency['shifts'] = [s for s in emergency.get('shifts', [])
                           if s['id'] not in completed]
```

---

## 9. utils.py — FSM状态机

### `EmergencyFSM` (L392-431)

**作用**：两状态有限状态机，管理突发事件生命周期。

**状态**：`NORMAL(0)` — GA方案执行中 → `EMERGENCY(1)` — DQN逐目标决策中

**转换**：`trigger_emergency()` → NORMAL→EMERGENCY；`resolve_emergency()` → EMERGENCY→NORMAL

**方法**：
- `trigger_emergency(emergency)` (L411-415): 仅NORMAL态可触发，重复触发返回False（不支持嵌套事件）
- `resolve_emergency()` (L417-421): 仅EMERGENCY态可调用
- `is_normal()` / `is_emergency()` (L423-427): 状态查询
- `current_state` (L429-431): `@property`，返回可读状态名

**注意**：当前训练流程中 FSM 主要用于结构清晰。实际训练循环（`train.py`）的决策循环由 for 循环驱动。

---

## 10. simulator.py — EmergencySimulator 类

### `__init__(self, hotspots, num_uavs, uav_base_km, spiral_cfg, prob_grid, X_km, Y_km)` (L25-35)

**作用**：初始化仿真器，深拷贝热点为 `self.original_hotspots`，预计算所有热点的螺旋搜索代价。

**导入依赖**：
```python
from emergency.utils import (
    NUM_UAVS, SPAN_KM, UAV_V_M_S, UAV_VELOCITY_KM_S,
    routes_to_chromosome,
)
```
`UAV_VELOCITY_KM_S` 用于 S3 绕飞代价的时间换算。

### `_precompute_spiral_costs(self)` (L37-50)

对每个原始热点运行一次 `spiral_search_arc_exact`（动态规划螺旋路径搜索），存入 `spiral_time`/`spiral_dist`/`pdet`。`spiral_time` 存在则跳过（防重复计算）。`yaw_at_center=0.0` 合理，因为螺旋总长度在圆形区域内旋转不变。

---

## 11. simulator.py — 场景生成

所有场景的 `trigger_time_frac` 统一为 `random.uniform(0.3, 0.7)`，保证事件在飞行进程的中段触发。

### `generate_random_emergency(self, routes, uav_path_lengths=None)` (L54-63)

等概率随机选择 S1/S2/S3/S4，分发到对应生成器。

### `_generate_s1(self)` (L65-86) — 热点位移

随机选 1~3 个热点，每个偏移 ±10km。严重度 = 最大位移 / 20km。

**注意**：`_generate_s1` 从全部热点中选。调用方在 `simulate_until_emergency` 返回 `completed` 后会过滤已完成目标，防止"已搜索目标再位移"。

### `_generate_s2(self, max_retries=50)` (L88-133) — 新热点

**新增重叠检测**：
- 每个新目标与所有已有目标（及已放置的新目标）的圆心距 ≥ `r_existing + r_new`
- 最多重试 50 次。重试耗尽则跳过该新目标
- 新目标 ID 从 100 开始
- `all_existing.append(new_target)` 保证后续新目标也避开前面已放置的

### `_generate_s3(self, routes, max_retries=30)` (L135-183) — 禁飞区

**新增热点不重叠检测**：
- 禁飞区圆不得与任何热点区域圆相交（`d ≥ r_hotspot + r_nofly`）
- 搜索范围从 ±3km 扩到 ±5km，半径上限从 10km 降到 8km
- 最多重试 30 次
- **兜底**：全部重试失败后使用 2km 小半径，确保不卡死
- 退化处理：无 ≥2 个目标的UAV时回退生成 S1

### `_generate_s4(self, routes)` (L185-199) — 无人机故障

选择有 ≥1 个目标的UAV标记为故障，其未完成目标变为 `lost_targets`。退化处理：无候选UAV时回退 S1。

---

## 12. simulator.py — 应用突发事件

### `apply_emergency(self, emergency)` (L203-234)

**返回**：`(modified_hotspots, active_uavs, completed_targets)`

- **S1 (L215-225)**：修改目标 `center_km`，**清除 `spiral_time`/`spiral_dist`/`pdet`**（位置变了，旧螺旋路径无效）。后续 `compute_cost_for_assignment` 检测缺失并重算
- **S2 (L226-228)**：追加新目标到热点列表
- **S3 (L229-230)**：`pass` — 禁飞区不修改热点，仅代价计算时加绕飞时间
- **S4 (L231-232)**：故障UAV `active_uavs[failed] = False`

---

## 13. simulator.py — 飞行仿真

### `simulate_until_emergency(self, ref_paths, emergency, baseline_routes=None, route_boundaries=None)` (L238-309)

**作用**：沿参考路径推算突发事件触发时每架UAV的位置，并判断哪些目标已完成。

**参数**：
- `ref_paths`: 各UAV参考路径（米坐标）
- `emergency`: 含 `trigger_time_frac`（0.3~0.7）
- `baseline_routes`: 用于推算已完成目标。**若传入则启用已完成判定**
- `route_boundaries`: 各UAV每个目标的累积路径距离（米）。**若传入则精确判定，否则用比例估算**

**已完成目标判定（L286-305）**：
- **精确模式**（L293-298）：对比 `trigger_dist` 与预计算的累积距离边界。触发位置已越过目标的螺旋终点 → 该目标已完成
- **近似模式**（L299-305）：按路径比例均匀分配。`trigger_frac ≥ (i+1)/n` → 目标 i 已完成。demo/eval 曾用此模式，现已改为精确模式

**位置推算**（L260-284）：沿路径每段累加距离，到达 `trigger_dist` 时线性插值得到精确 (x, y)。

**返回值**：`uav_positions`(km), `uav_headings`(rad), `completed`(set), `remaining`(set)

---

## 14. simulator.py — 代价计算

### `compute_cost_for_assignment(self, assignment_routes, hotspots, active_uavs, emergency=None)` (L313-376)

**完整流程**：

**Step 1: ID重映射 (L317)**
```python
id_to_idx = {h['id']: i for i, h in enumerate(hotspots)}
```
原始ID（含S2的 ID≥100）→ 连续索引 0,1,2,...。非连续ID会被GA染色体误判为分隔符。

**Step 2: 补算缺失螺旋代价 (L319-336)**
对每个目标检查 `spiral_time` 是否存在。缺失的（S1位移后清除、S2新增）调用 `spiral_search_arc_exact` 重算。**只算缺失的，不重算已有缓存**。

**Step 3: 设置GA全局变量 (L338-340)**
`ga.init_ga_env(..., prob_grid=None, weights=(0.5, 0.3, 0.4))` — `prob_grid=None` 不触发重算（已在Step 2完成）。

**Step 4: 路由重映射 + 原始代价 (L342-346)**
```python
remapped_routes = [[id_to_idx[tid] for tid in route if tid in id_to_idx]
                    for route in assignment_routes]
chrom = routes_to_chromosome(remapped_routes, num_targets, NUM_UAVS)
J_max, J_sum, w_arrival, routes, finish_times = ga.calculate_raw_costs(chrom)
```

**Step 5: S3绕飞代价估算 (L348-367)** — 与旧版的关键区别
```python
detour_time_s = 2.5 * nf_radius / UAV_VELOCITY_KM_S  # 绕飞额外时间
```
- **旧版**：穿越禁飞区 → 加 5000s 固定惩罚。过于严厉，DQN 学会"放弃"来完全避开惩罚
- **新版**：穿越禁飞区 → 加绕飞额外时间。公式 `2.5 × r / 0.05`，r=5km → 250s。代价合理，DQN 能学到"绕飞比放弃好"
- 对每条UAV路由（`calculate_raw_costs` 解码后的），逐段检测线段-圆相交
- `no_fly_penalty` 累加到 `J_sum`，按活跃UAV数分摊到 `J_max`

**返回值**：`{'J_max', 'J_sum', 'weighted_arrival_sum', 'routes', 'finish_times'}`

### `_line_intersects_circle(p1, p2, center, radius)` (L378-403) — 静态方法

解参数方程 `P(t) = p1 + t*(p2-p1)` 代入圆方程，得二次方程 `at² + bt + c = 0`。判断解是否在 `[0,1]` 内。

`t1 ≤ 0 and t2 ≥ 1` 处理线段完全包含圆的情况。

### `compute_oracle_cost(self, remaining_targets, active_uavs, modified_hotspots, weights=...)` (L405-454)

在剩余目标子集上运行GA重优化，获取理论最优代价。作为DQN训练的参考上界。

- 筛掉已完成目标
- 补算缺失螺旋代价
- 调用 `ga.run_ga(pop_size=100, generations=150, patience=50)`（比训练用GA规模小）
- 异常返回极大值 `1e9`

### `compute_reward(self, dqn_cost, oracle_cost=None, baseline_val=None, abandoned_targets=None)` (L456-477)

**作用**：将物理代价转换为标量奖励信号。

**奖励公式（优先级从高到低）**：
1. **零方案基线**（优先）：`reward = (baseline_val - dqn_val - abandon_penalty) / baseline_val`
   - DQN比"什么也不做"好 → 正reward；更差 → 负reward
2. **GA oracle 对比**：`reward = (oracle_val - dqn_val - abandon_penalty) / oracle_val`
3. **绝对尺度**（无基线时）：`reward = -(dqn_val + abandon_penalty) / 10000`

**放弃惩罚**（L467-469）：
```python
abandon_penalty = sum(t.get('weight', 0.3) for t in abandoned_targets) * 5000.0
```
权重越高放弃代价越大。防止DQN随意放弃高权重目标。

**注意**：`reward` 在调用方（train.py）被 clamp 到 `[-10, 10]`。

---

## 数据流总览

```
EmergencySimulator.__init__()
  └─ _precompute_spiral_costs()    ← 一次性预计算15个热点的螺旋代价

每回合训练:
  generate_random_emergency()      → emergency 字典
  simulate_until_emergency()       → UAV位置 + completed/remaining
  过滤 S1 shifts 中的已完成目标
  apply_emergency()                → modified_hotspots, active_uavs
  get_affected_targets()           → 受影响目标列表 (按权重降序)
  过滤 affected 中的已完成目标

  for each affected target:
    build_state_vector()           → 34维状态向量 (含禁飞区位置)
    get_valid_actions(..., etype)  → 有效动作 (S2/S4屏蔽action=0)
    DQN.forward(state)             → 5个Q值
    select_action(q, valid, eps)   → 动作 0-4
    apply_decision(action, ...)    → 更新路由 (从其他UAV先移除)

  nearest_neighbor_reorder_with_hotspots()  → 重排序
  balance_routes()                          → 负载均衡 (空载UAV抢目标)
  移除放弃目标
  compute_cost_for_assignment()             → DQN代价 (含S3绕飞时间)
  (compute_oracle_cost())                   → GA最优代价 (可选)
  compute_reward(abandoned_targets)         → 奖励信号 (含放弃惩罚)
```
