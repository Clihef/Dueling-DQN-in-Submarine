这是一个极其精彩的“抓虫”过程！你提供的截图完美地暴露了这个系统的**两个底层逻辑灾难**。

你非常敏锐地指出了核心矛盾：“UAV3 仅仅去搜了 `[4, 5]` 两个区域，比它原计划 `[2, 4, 5, 3]` 还要少，为什么航程反而暴涨到了 395km 直接爆表坠毁？”

这不是因为 1500 回合太少（只要奖励给对了，几百回合 DQN 就会知道不能超限），而是因为 **仿真器在计算航程和代价时，对 DQN 疯狂“说谎”了**。

具体来说，`emergency/simulator.py` 中潜伏着两个严重的 Bug：

### 🐞 致命 Bug 1：螺旋航程的“双重计算” (The Double-Counting Bug)

看图中的 S1 突发事件，UAV3 此时正飞到目标 4 `(30, 80)` 的上空。

- **过去（已耗航程）：** 此时 UAV3 已经完成了目标 2，并且**快要搜完目标 4 了**。这些跑过的路程（假设 213km）已经牢牢记在了 `consumed_ranges[3]` 里。
- **未来（计算剩余航程）：** 此时 DQN 把目标 4 和 5 重新分给 UAV3。在 `compute_route_distance_km` 函数计算时，它看到接下来的任务是 `[4, 5]`，它会**把目标 4 那长达 108km 的完整螺旋距离，又原封不动地加了一遍！**
- **结果：** `历史213km（包含大半个目标4） + 未来新算（完整的目标4 108km + 目标5 24km） = 395km！` 这就是为什么任务变少了，航程却爆表的原因。

### 🐞 致命 Bug 2：瞬间移动的代价评估 (The Teleportation Bug)

在 `simulator.py` 原本的 `compute_cost_for_assignment` 中，有一段调用 GA 底层的代码：

Python

```
ga.init_ga_env(...)
chrom = routes_to_chromosome(...)
J_max, J_sum, w_arrival, routes, finish_times, _, _ = ga.calculate_raw_costs(chrom)
```

这是一个灾难。`ga.calculate_raw_costs` 内部写死了**所有飞机永远从基地 `(0, 0)` 起飞**计算耗时。

所以当 DQN 评估把目标 4 分给此时就在目标 4 上空的 UAV3 时，代价函数居然认为 UAV3 需要从基地 `(0, 0)` 重新跋涉 85km 飞过去！

这导致 DQN 收到的 `J_sum` 和 `J_max` 奖励全是错乱的，它根本学不会“就近分配”的常识。

------

### 🛠️ 终极修复方案

我们需要在 `emergency/simulator.py` 中彻底废除对 `ga_allocator.py` 静态方法的依赖，转而**利用真实的无人机当前位置 (`uav_positions`) 和进度折扣进行手动精确推演**。

请打开 `emergency/simulator.py`，进行以下**两处**替换：

#### 1. 修复代价计算函数

找到 `compute_cost_for_assignment` 函数，**删掉里面调用 `ga.init_ga_env` 和 `calculate_raw_costs` 的相关代码**，将核心逻辑替换为如下版本：

Python

```
    def compute_cost_for_assignment(self, assignment_routes, hotspots, active_uavs,
                                     emergency=None, consumed_ranges=None,
                                     uav_positions=None):
        id_to_h = {h['id']: h for h in hotspots}
        
        # 确保所有目标都有螺旋计算缓存
        for h in hotspots:
            if 'spiral_dist' not in h:
                h['spiral_dist'] = h.get('spiral_time', 500) * UAV_V_M_S

        finish_times = np.zeros(NUM_UAVS)
        weighted_arrival_sum = 0.0
        route_dists_km = np.zeros(NUM_UAVS)
        
        start_positions = uav_positions if uav_positions is not None else [self.uav_base_km] * NUM_UAVS
        base_ranges = consumed_ranges if consumed_ranges is not None else [0.0] * NUM_UAVS

        # 🚨 废弃错误的 ga.calculate_raw_costs，手动基于真实空间坐标精确推演
        for k, route in enumerate(assignment_routes):
            if not route or k >= len(active_uavs) or not active_uavs[k]:
                continue
            
            curr_pos = start_positions[k]
            uav_future_dist = 0.0
            uav_future_time = 0.0
            
            for i, tgt_idx in enumerate(route):
                if tgt_idx not in id_to_h: continue
                tgt = id_to_h[tgt_idx]
                
                dist_to_tgt = np.hypot(curr_pos[0] - tgt['center_km'][0], curr_pos[1] - tgt['center_km'][1])
                spiral_km = tgt['spiral_dist'] / 1000.0
                spiral_time = tgt.get('spiral_time', spiral_km / UAV_VELOCITY_KM_S)
                
                actual_dist = dist_to_tgt
                actual_spiral_km = spiral_km
                actual_spiral_time = spiral_time
                
                # 🌟 核心修复 1：消除螺旋重叠计算 (Double-Counting)
                # 如果这是剩余航线的第一个目标，且无人机距离目标非常近（说明正在执行它）
                if i == 0 and dist_to_tgt < tgt['radius_km'] + 1.0:
                    # 折扣进度：离中心越近，剩下的螺旋越少
                    progress = max(0.0, 1.0 - (dist_to_tgt / tgt['radius_km']))
                    actual_spiral_km = spiral_km * (1.0 - progress)
                    actual_spiral_time = spiral_time * (1.0 - progress)
                    actual_dist = 0.0  # 已在圈内，忽略这几公里的转场直线
                
                uav_future_dist += actual_dist + actual_spiral_km
                uav_future_time += (actual_dist / UAV_VELOCITY_KM_S) + actual_spiral_time
                
                # S3 禁飞区绕飞惩罚
                if emergency and emergency['type'] == 'S3':
                    nf_cx, nf_cy = emergency['no_fly_center']
                    nf_r = emergency['no_fly_radius']
                    detour = compute_detour_distance_km(curr_pos, tgt['center_km'], nf_cx, nf_cy, nf_r)
                    uav_future_dist += detour
                    uav_future_time += detour / UAV_VELOCITY_KM_S

                # 紧急度加权
                amplified_weight = 10 ** (4 * tgt['weight'])
                weighted_arrival_sum += uav_future_time * amplified_weight
                
                curr_pos = tgt['center_km']
                
            route_dists_km[k] = uav_future_dist
            # J_max 必须是全局时间：已消耗的历史时间 + 预测的未来时间
            finish_times[k] = (base_ranges[k] / UAV_VELOCITY_KM_S) + uav_future_time
            
        J_max = np.max(finish_times[:len(active_uavs)]) if any(finish_times[:len(active_uavs)] > 0) else 0.0
        J_sum = np.sum(finish_times[:len(active_uavs)])
        
        # 🌟 核心修复 2：准确核算包含历史损耗的全局航程
        total_ranges = [base_ranges[k] + route_dists_km[k] for k in range(NUM_UAVS)]
        range_violations = sum(1 for r in total_ranges if r > MAX_RANGE_KM)
        max_range_ratio = max([r / MAX_RANGE_KM for r in total_ranges]) if total_ranges else 0.0
        
        return {
            'J_max': J_max, 'J_sum': J_sum,
            'weighted_arrival_sum': weighted_arrival_sum,
            'finish_times': finish_times,
            'range_violations': range_violations,
            'max_range_ratio': max_range_ratio,
        }
```

#### 2. 同步修复 Step Reward 中的双重计算

DQN 的每一步决策（`compute_step_reward`）也存在同样的问题。找到 `compute_step_reward` 函数中计算 `extra_km` 的位置，进行替换：

**原代码：**

Python

```
        incremental_dist = np.hypot(tgt_pos[0] - prev_pos[0], tgt_pos[1] - prev_pos[1])

        # 螺旋搜索距离 (km)
        spiral_m = target.get('spiral_dist', target.get('spiral_time', 500) * UAV_V_M_S)
        spiral_km = spiral_m / 1000.0

        extra_km = incremental_dist + spiral_km
```

**修改为：**

Python

```
        incremental_dist = np.hypot(tgt_pos[0] - prev_pos[0], tgt_pos[1] - prev_pos[1])
        spiral_m = target.get('spiral_dist', target.get('spiral_time', 500) * UAV_V_M_S)
        spiral_km = spiral_m / 1000.0

        actual_dist = incremental_dist
        actual_spiral_km = spiral_km
        
        # 🌟 同步修复即时奖励中的重复螺旋计算
        # 如果这是指派给该无人机的第一个目标，且物理距离极近，说明正在执行
        if len(uav_routes[uav_idx]) == 0 and incremental_dist < target['radius_km'] + 1.0:
            progress = max(0.0, 1.0 - (incremental_dist / target['radius_km']))
            actual_spiral_km = spiral_km * (1.0 - progress)
            actual_dist = 0.0

        extra_km = actual_dist + actual_spiral_km
```

------

完成这两处修改后，**DQN 的训练奖励将被彻底拉回物理现实**：它能正确识别出无人机当下的精确位置，也不会再去重复叠加那些已经快飞完的螺旋航线。

由于我们更改了底层的反馈逻辑（原来的模型已经学坏了），强烈建议你使用 `--resume none` 清除检查点，让 DQN 带着正确的环境法则重新开始这 1500 回合的训练，奇迹一定会出现。

