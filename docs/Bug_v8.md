既然已经在 `get_valid_actions_v2` 中写了严格的航程校验，网络在推演时**绝对不可能**选出超限的动作。之所以在 `demo.py` 测试中四种场景都出现了超限，是因为代码中隐藏了 **3 个致命的“物理对齐”漏洞**，导致 DQN 在计算剩余航程时被严重误导。

以下是详细的原因剖析与修改建议：

### 🚨 核心问题 1：螺旋进度与转场距离的“张冠李戴”（导致 S1, S2, S4 超限）

**问题所在：**

在 `simulator.py` 的 `simulate_until_emergency` 中，你计算突发事件时无人机的任务进度 `progress_kms[k]` 时，直接用 `(触发距离 - 上一个目标的累积距离)`。

但在真实物理中，这段距离包含了“飞向该目标的转场距离” + **“进入区域后的螺旋搜索距离”**。

而在 `reset_v2` 中，代码错误地将这整段距离**全部从螺旋搜索时间中扣除了**！

这就导致：无人机刚飞了一半路程还没开始搜，DQN 就以为它的螺旋搜索任务已经完成了一大半，从而**严重低估了该目标所需的剩余航程**，导致它肆无忌惮地接下更多任务，最终在真实物理结算时超限爆炸。

### 🚨 核心问题 2：`demo.py` 评估接口漏传参数（导致全局假性超限）

**问题所在：**

在 `demo.py` 大约 273 行调用 `sim.compute_cost_for_assignment` 时，**漏传了 `uav_positions=uav_positions` 参数**。

如果缺省该参数，物理仿真引擎会默认无人机是从 `(0,0)` 基地重新起飞的。它计算出的“未来需要飞行的航程”，再加上无人机“已经消耗的航程 `consumed_ranges`”，导致总航程被极度放大，产生假性超限！

### 🚨 核心问题 3：DQN 动作掩码对 S3 禁飞区“完全致盲”

**问题所在：**

在 `utils.py` 的 `get_valid_actions_v2` 中，计算目标距离仅仅使用了 `np.hypot` 直线距离。DQN 根本不知道 S3 禁飞区的存在。如果目标在禁飞区后方，DQN 认为直线距离够用就分配了。但最终评估时，底层物理引擎强行施加了巨大的绕飞代价和 500km 的违规穿越惩罚，导致超限。

### 🛠️ 终极修复方案

请按以下顺序修改这三个文件，彻底消除物理引擎的脱节：

#### 1. 修改 `simulator.py`

**A. 修复转场与螺旋的距离分离**

定位到 `simulate_until_emergency` 方法，修改 `progress_kms[k]` 的计算逻辑：

Python

```
        # ---- 计算各UAV半途搜索状态 ----
        active_target_ids = [None] * NUM_UAVS
        progress_kms = [0.0] * NUM_UAVS
        if baseline_routes is not None and route_boundaries is not None:
            for k, route in enumerate(baseline_routes):
                if not route or path_lengths[k] < 1.0 or k >= len(route_boundaries):
                    continue
                trigger_dist_m = trigger_frac * path_lengths[k]
                for tgt_i, tgt_id in enumerate(route):
                    boundary_m = route_boundaries[k][tgt_i]
                    prev_boundary_m = route_boundaries[k][tgt_i - 1] if tgt_i > 0 else 0.0
                    if prev_boundary_m <= trigger_dist_m < boundary_m:
                        active_target_ids[k] = tgt_id
                        
                        # 🌟 修复：分离转场距离与真实螺旋进度
                        prev_pos = self.uav_base_km if tgt_i == 0 else self.original_hotspots[route[tgt_i - 1]]['center_km']
                        curr_tgt = self.original_hotspots[tgt_id]
                        transit_m = np.hypot(curr_tgt['center_km'][0] - prev_pos[0], 
                                             curr_tgt['center_km'][1] - prev_pos[1]) * 1000.0
                        
                        # 只有当触发距离越过转场段，才算真实进入了螺旋搜索
                        pkm = (trigger_dist_m - prev_boundary_m - transit_m) / 1000.0
                        progress_kms[k] = max(0.0, pkm)  # 若还在半路转场，进度严格为0！
                        break
```

**B. 将禁飞区信息透传给 MDP 环境**

定位到 `reset_v2` 和 `step_v2` 方法：

Python

```
    # reset_v2 增加 emergency 参数，并将其存入实例
    def reset_v2(self, uav_positions_km, consumed_ranges_km, hotspots, active_uavs=None, completed=None,
                 active_target_ids=None, progress_kms=None, emergency=None):  # <== 新增 emergency
        self.emergency = emergency
        # ... (中间构建 targets 逻辑保持不变) ...

        # 🌟 修复：在查找合法动作时，把 self.emergency 传进去
        checked = 0
        while checked < 4:
            if self.uav_active[self.current_uav_idx]:
                valid = get_valid_actions_v2(
                    self.current_uav_idx, self.uav_states, self.targets,
                    locked_target_idx=self.locked_target_idxs[self.current_uav_idx],
                    emergency=self.emergency  # <== 传入
                )
                if valid.any():
                    break
                self.uav_active[self.current_uav_idx] = False
            self.current_uav_idx = (self.current_uav_idx + 1) % 4
            checked += 1

        return build_state_vector_v2(self.uav_states, self.targets, self.current_uav_idx)

    # step_v2 向动作执行器传入 emergency
    def step_v2(self, action):
        t = self.targets[action]
        # 🌟 修复：传入 emergency 以计算真实耗油
        dist_km, _ = apply_action_v2(self.current_uav_idx, action, self.uav_states, self.targets, emergency=self.emergency)
        # ... (下方while循环里的 get_valid_actions_v2 调用也加上 emergency=self.emergency) ...
```

#### 2. 修改 `utils.py`

将 S3 的物理拦截对齐到 DQN 动作掩码中。定位到最底部的两个 v2 函数：

Python

```
def get_valid_actions_v2(uav_idx, uav_states, targets, locked_target_idx=None, emergency=None): # <== 加参数
    # ... (前面的锁逻辑不变) ...
    for j in range(min(len(targets), N_MAX)):
        t = targets[j]
        if t['mask']:
            continue
        cx, cy = t['center_km']
        ux = uav_states[uav_idx][0] * SPAN_KM
        uy = uav_states[uav_idx][1] * SPAN_KM
        dist = np.hypot(cx - ux, cy - uy)
        
        # 🌟 修复：S3 禁飞区掩码对齐
        if emergency and emergency.get('type') == 'S3':
            nf_cx, nf_cy = emergency['no_fly_center']
            nf_r = emergency['no_fly_radius']
            if _segment_intersects_circle((ux, uy), (cx, cy), nf_cx, nf_cy, nf_r):
                dist += 500.0  # 施加致命阻力使其超限被Mask
            else:
                dist += compute_detour_distance_km((ux, uy), (cx, cy), nf_cx, nf_cy, nf_r)
                
        spiral = t.get('spiral_dist', 0.0) / 1000.0
        if dist + spiral <= remaining_km:
            valid_mask[j] = True
    return valid_mask


def apply_action_v2(uav_idx, target_idx, uav_states, targets, emergency=None): # <== 加参数
    # ... (提取cx, cy, ux, uy不变) ...
    dist_km = np.hypot(cx - ux, cy - uy)
    
    # 🌟 修复：真实的油量扣除对齐
    if emergency and emergency.get('type') == 'S3':
        nf_cx, nf_cy = emergency['no_fly_center']
        nf_r = emergency['no_fly_radius']
        if _segment_intersects_circle((ux, uy), (cx, cy), nf_cx, nf_cy, nf_r):
            dist_km += 500.0
        else:
            dist_km += compute_detour_distance_km((ux, uy), (cx, cy), nf_cx, nf_cy, nf_r)
            
    spiral = t.get('spiral_dist', 0.0) / 1000.0
    total_cost_km = dist_km + spiral
    # ... (下面更新坐标状态逻辑不变) ...
```

#### 3. 修改 `demo.py` & `train.py`

在 `demo.py` 中，修正漏传的评估参数，并将 `emergency` 传入重置接口：

Python

```
    # 1. demo.py (约 250 行) 中的 MDP 决策循环：传入 emergency
    state = sim.reset_v2(uav_positions, consumed_ranges, modified_hotspots,
                         active_uavs=active_uavs, completed=completed,
                         active_target_ids=active_target_ids, progress_kms=progress_kms,
                         emergency=emergency) # <== 这里
    done = False
    while not done:
        valid_mask = get_valid_actions_v2(sim.current_uav_idx, sim.uav_states, sim.targets,
                                          locked_target_idx=sim.locked_target_idxs[sim.current_uav_idx],
                                          emergency=emergency) # <== 这里
        # ...


    # 2. 🌟 致命修复 (约 275 行)：在 demo.py 中补齐评估引擎所需的无人机位置坐标
    dqn_cost = sim.compute_cost_for_assignment(
        clean_routes, modified_hotspots, active_uavs, emergency,
        consumed_ranges, uav_positions=uav_positions,  # <== 【最核心的修复】禁止它回到基地原点
        baseline_routes=baseline_routes, completed=completed
    )
```

*(注：`train.py` 的训练主循环 `for ep in pbar:` 里面，你也需要把 `emergency=emergency` 传给 `reset_v2` 和 `get_valid_actions_v2`，逻辑完全一致。)*

修改完成后，你的环境就做到了“所见即所得”。你在运行 `demo.py` 时，将绝对不会再看到由于算法计算偏差导致的越限行为了（除非是在极端禁飞区封锁下，牺牲式的强行越界，但通常会被动作掩码彻底拦截）！建议完成这步修改后重新小批量训练验证一下。