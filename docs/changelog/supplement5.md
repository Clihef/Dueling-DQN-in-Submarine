这份计划中存在 **两个容易导致运行时错误和评估失效的疏漏**。

### 🚨 疏漏一：`simulator.py` 中 `reset_v2` 的内部调用被遗漏

**问题分析**：

在你的计划 **Step 2** 中，提到要在 `reset_v2` 中增加 `emergency` 参数并保存为 `self.emergency`，以及在 `step_v2` 中传入 `emergency`。

**但是，你忽略了在 `reset_v2` 函数内部，其实也调用了一次 `get_valid_actions_v2`。** 在 `reset_v2` 的末尾，有一个 `while checked < 4:` 的循环，用来确保随机起手的 UAV 拥有合法动作。如果你不在这里传入 `emergency`，那么第一架无人机在进行初始化评估时，**依然会对 S3 禁飞区致盲**，导致环境起手状态出错。

**👉 补充修改**：

在 `simulator.py` 的 `reset_v2` 方法的循环中追加传参：

Python

```
        # 🌟 必须在 reset_v2 内部的校验中也传入 self.emergency
        checked = 0
        while checked < 4:
            if self.uav_active[self.current_uav_idx]:
                valid = get_valid_actions_v2(
                    self.current_uav_idx, self.uav_states, self.targets,
                    locked_target_idx=self.locked_target_idxs[self.current_uav_idx],
                    emergency=self.emergency  # <== 计划中遗漏了这里
                )
                if valid.any():
                    break
                self.uav_active[self.current_uav_idx] = False
            self.current_uav_idx = (self.current_uav_idx + 1) % 4
            checked += 1
```

### 🚨 疏漏二：`eval.py` 评估脚本完全被遗漏

**问题分析**：

你的计划 **Step 3** 中只包含了 `demo.py` 和 `train.py`。

**但是，`eval.py` 拥有完全相同的 DQN 推演循环 (`while not done:`)。**

如果你不更新 `eval.py`，当你运行批量场景评估（输出学术对比图表）时，`eval.py` 里的 DQN 在面对 S3 场景时，调用的是未传入 `emergency` 的 `reset_v2` 和 `get_valid_actions_v2`。这会导致评估程序中 DQN **依然会选择穿越禁飞区**，最终在 `compute_cost_for_assignment` 物理结算时吃下巨额惩罚，使得 Fig4 (热力图) 中 DQN 的 S3 性能看起来极其糟糕。

**👉 补充修改**：

打开 `eval.py`，定位到 `---- 方法1: DQN (轮询式单步决策 V2) ----`（约在第 187 行附近），修改两处方法调用：

Python

```
        # 1.  补充传入 emergency
        state = sim.reset_v2(uav_positions, consumed_ranges, modified_hotspots,
                             active_uavs=active_uavs, completed=completed,
                             active_target_ids=active_target_ids, progress_kms=progress_kms,
                             emergency=emergency)  # <== 新增
        done = False
        decisions = []
        
        while not done:
            # 2. get_valid_actions_v2 补充传入 emergency
            valid_mask = get_valid_actions_v2(
                sim.current_uav_idx, sim.uav_states, sim.targets,
                locked_target_idx=sim.locked_target_idxs[sim.current_uav_idx],
                emergency=emergency  # <== 新增
            )
            # ... 下方代码保持不变
```

*(注：`eval.py` 中 `sim.compute_cost_for_assignment` 已经按位置正确传入了 `uav_positions`，所以那一部分无需额外修改。)*