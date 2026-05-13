但是，在仔细审查了你的 `simulator.py` 和 `train.py` 源码后，我发现这份计划**遗漏了两个隐蔽但致命的 Bug**，如果不一并修复，模型在 S1、S2 和 S4 场景下依然会崩溃或学到错误的策略。

以下是评估结论及补充修改建议：

### 🚨 隐患一：“死机复活” (S4 场景失效)

在现有的 `simulator.py` 中，`reset_v2` 方法是这样硬编码初始化的：

Python

```
self.uav_active = [True] * 4
self.current_uav_idx = random.randint(0, 3)
```

**问题：** 如果发生 **S4（无人机故障）**，在物理环境中该飞机已经坠毁（`active_uavs[failed_uav] = False`），但在调用 `reset_v2` 时，它被无条件重置为 `True`！这会导致 DQN 在虚拟推演时继续把目标分配给这架已经死掉的飞机，导致实际执行时代价爆炸。

**👉 补充修改方案：**

`reset_v2` 必须接收物理环境的 `active_uavs` 掩码：

Python

```
# v3/emergency/simulator.py
def reset_v2(self, uav_positions_km, consumed_ranges_km, hotspots, active_uavs=None, completed=None):
    # ... 前面的代码保持不变 ...
    
    # 修复 S4 死机复活问题
    self.uav_active = active_uavs.copy() if active_uavs else [True] * 4
    
    # 确保起始决策的 UAV 是存活的
    valid_uavs = [i for i, active in enumerate(self.uav_active) if active]
    self.current_uav_idx = random.choice(valid_uavs) if valid_uavs else 0
```

*(同时记得在 `train.py` 和 `demo.py` 中调用 `reset_v2` 时，传入 `active_uavs=active_uavs`)*

### 🚨 隐患二：0 距离的“白嫖”搜索 (S1/S2 场景失效)

在 `simulator.py` 的 `apply_emergency` 中，如果是 S1（位置突变），代码会执行：

Python

```
tgt.pop('spiral_time', None)
tgt.pop('spiral_dist', None)
```

如果是 S2（新增热点），新目标字典里天然没有 `spiral_dist` 键。

而在你的修改计划中：

Python

```
'spiral_dist': h.get('spiral_dist', 0.0),
```

**问题：** 这意味着在 S1 和 S2 场景中，突变和新增目标的 `spiral_dist` 会默认变为 `0.0`！DQN 会认为搜索这些目标不消耗任何额外航程，从而疯狂地将它们纳入囊中。这就导致 DQN 规划出的航线在 `compute_cost_for_assignment` 进行真实物理校验时，发生严重的航程超限（Violations）。

**👉 补充修改方案：**

在 `reset_v2` 中提取 `spiral_dist` 时，必须加入与物理校验层对齐的兜底估算逻辑：

Python

```
# v3/emergency/simulator.py -> reset_v2
completed_set = set(completed) if completed else set()
self.targets = []
for h in hotspots:
    # 修复 S1/S2 缺失螺旋距离导致 DQN 白嫖的问题
    s_dist = h.get('spiral_dist')
    if s_dist is None:
        # 降级估算：时间 * 速度 或 面积估算。需与 compute_cost_for_assignment 保持一致
        s_dist = h.get('spiral_time', 500) * UAV_V_M_S 

    self.targets.append({
        'id': h['id'], 
        'center_km': h['center_km'], 
        'weight': h['weight'],
        'spiral_dist': s_dist,
        'mask': h['id'] in completed_set
    })
```
