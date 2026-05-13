"""
突发事件仿真器：场景生成、飞行仿真、代价计算
支持四种突发事件类型 (S1:热点位移, S2:新热点, S3:禁飞区, S4:无人机故障)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import math
import random
import copy

from core import ga_allocator as ga
from core.spiral_search import spiral_search_arc_exact

from emergency.utils import (
    NUM_UAVS, SPAN_KM, UAV_V_M_S, UAV_VELOCITY_KM_S,
    MAX_RANGE_KM, SOFT_RANGE_KM,
    compute_detour_distance_km,
    GLOBAL_C_DROP,
    N_MAX, ALPHA_REWARD, BETA_REWARD, R_SUCCESS, GAMMA_UNVISITED, MAX_DIAG_KM,
    build_state_vector_v2, get_valid_actions_v2, apply_action_v2,
    _segment_intersects_circle,
)


class EmergencySimulator:
    """突发事件仿真环境：场景生成、飞行仿真、代价计算"""

    def __init__(self, hotspots, num_uavs, uav_base_km, spiral_cfg,
                 prob_grid, X_km, Y_km):
        self.original_hotspots = copy.deepcopy(hotspots)
        self.num_uavs = num_uavs
        self.uav_base_km = uav_base_km
        self.spiral_cfg = spiral_cfg
        self.prob_grid = prob_grid
        self.X_km = X_km
        self.Y_km = Y_km

        self._precompute_spiral_costs()

    def _precompute_spiral_costs(self):
        for tgt in self.original_hotspots:
            if 'spiral_time' not in tgt:
                sol = spiral_search_arc_exact(
                    prob_grid=self.prob_grid,
                    center_km=tgt['center_km'],
                    radius_km=tgt['radius_km'],
                    X_km=self.X_km, Y_km=self.Y_km,
                    yaw_at_center=0.0,
                    cfg=self.spiral_cfg,
                )
                tgt['spiral_dist'] = sol['totalLen']
                tgt['spiral_time'] = sol['totalTime']
                tgt['pdet'] = sol['Pdet']

    # ==================== 场景生成 ====================

    def generate_random_emergency(self, routes, uav_path_lengths=None):
        etype_probs = getattr(self, 'emergency_probs', [0.4, 0.3, 0.2, 0.1])
        etype = random.choices(['S1', 'S2', 'S3', 'S4'], weights=etype_probs, k=1)[0]
        if etype == 'S1':
            return self._generate_s1()
        elif etype == 'S2':
            return self._generate_s2()
        elif etype == 'S3':
            return self._generate_s3(routes)
        else:
            return self._generate_s4(routes)

    def _generate_s1(self):
        """S1: 原有热点位置突变 (1~3个目标)"""
        n_shift = random.randint(1, 3)
        candidates = [h for h in self.original_hotspots]
        selected = random.sample(candidates, min(n_shift, len(candidates)))

        shifts = []
        affected_ids = []
        for tgt in selected:
            dx = random.uniform(-10, 10)
            dy = random.uniform(-10, 10)
            shifts.append({'id': tgt['id'], 'dx': dx, 'dy': dy})
            affected_ids.append(tgt['id'])

        max_shift = max(np.hypot(s['dx'], s['dy']) for s in shifts)
        severity = min(max_shift / 20.0, 1.0)

        return {
            'type': 'S1', 'shifts': shifts,
            'affected_targets': affected_ids, 'severity': severity,
            'trigger_time_frac': random.uniform(0.3, 0.7),
        }

    def _generate_s2(self, max_retries=50):
        """S2: 突发新热点 (1~3个新目标)，不与已有目标及彼此重叠"""
        n_new = random.randint(1, 3)
        new_targets = []
        all_existing = list(self.original_hotspots)  # 已有目标

        for i in range(n_new):
            radius = random.uniform(2.0, 4.0)
            placed = False
            for _ in range(max_retries):
                cx = random.uniform(10, SPAN_KM - 10)
                cy = random.uniform(10, SPAN_KM - 10)
                # 检查与已有目标的重叠
                overlap = False
                for h in all_existing:
                    d = np.hypot(cx - h['center_km'][0], cy - h['center_km'][1])
                    if d < h['radius_km'] + radius:
                        overlap = True
                        break
                # 检查与已放置新目标的重叠
                if not overlap:
                    for nt in new_targets:
                        d = np.hypot(cx - nt['center_km'][0], cy - nt['center_km'][1])
                        if d < nt['radius_km'] + radius:
                            overlap = True
                            break
                if not overlap:
                    placed = True
                    break
            if not placed:
                continue  # 重试耗尽，跳过此新目标
            new_targets.append({
                'id': 100 + i,
                'center_km': (cx, cy),
                'sigma': (random.uniform(2, 5), random.uniform(2, 5)),
                'weight': random.uniform(0.3, 1.0),
                'radius_km': radius,
            })
            all_existing.append(new_targets[-1])  # 后续新目标也要避开此目标

        return {
            'type': 'S2', 'new_targets': new_targets,
            'affected_targets': [t['id'] for t in new_targets],
            'severity': n_new / 3.0,
            'trigger_time_frac': random.uniform(0.3, 0.7),
        }

    def _generate_s3(self, routes, max_retries=30):
        """S3: 突现禁飞区阻断航路，不与任何热点区域重合

        禁飞区仅放置在未飞行的未来航段上。
        根据触发时间比例估算已飞过的目标数，避开已穿越的航段。
        """
        valid_uavs = [(k, r) for k, r in enumerate(routes) if len(r) >= 2]
        if not valid_uavs:
            return self._generate_s1()

        uav_idx, route = random.choice(valid_uavs)
        trigger_frac = random.uniform(0.3, 0.7)
        # 估算已飞过目标数，禁飞区只放在未来航段
        n_done = int(trigger_frac * len(route))
        min_seg = min(n_done, len(route) - 2)
        seg_idx = random.randint(min_seg, len(route) - 2)
        tgt_a = self.original_hotspots[route[seg_idx]]
        tgt_b = self.original_hotspots[route[seg_idx + 1]]

        base_mx = (tgt_a['center_km'][0] + tgt_b['center_km'][0]) / 2
        base_my = (tgt_a['center_km'][1] + tgt_b['center_km'][1]) / 2

        for _ in range(max_retries):
            mx = base_mx + random.uniform(-5, 5)
            my = base_my + random.uniform(-5, 5)
            radius = random.uniform(3, 8)
            # 检查是否与任何热点区域重合
            overlap = False
            for h in self.original_hotspots:
                d = np.hypot(mx - h['center_km'][0], my - h['center_km'][1])
                if d < h['radius_km'] + radius:
                    overlap = True
                    break
            if not overlap:
                return {
                    'type': 'S3',
                    'no_fly_center': (mx, my), 'no_fly_radius': radius,
                    'affected_uav': uav_idx,
                    'affected_segment': (route[seg_idx], route[seg_idx + 1]),
                    'affected_targets': [route[seg_idx], route[seg_idx + 1]],
                    'severity': 1.0,
                    'trigger_time_frac': trigger_frac,
                }

        # 重试耗尽，降低半径兜底
        mx = base_mx + random.uniform(-3, 3)
        my = base_my + random.uniform(-3, 3)
        radius = 2.0
        return {
            'type': 'S3',
            'no_fly_center': (mx, my), 'no_fly_radius': radius,
            'affected_uav': uav_idx,
            'affected_segment': (route[seg_idx], route[seg_idx + 1]),
            'affected_targets': [route[seg_idx], route[seg_idx + 1]],
            'severity': 1.0,
            'trigger_time_frac': trigger_frac,
        }

    def _generate_s4(self, routes):
        """S4: 无人机故障"""
        candidates = [k for k, r in enumerate(routes) if len(r) >= 1]
        if not candidates:
            return self._generate_s1()

        failed_uav = random.choice(candidates)
        lost_targets = routes[failed_uav].copy()

        return {
            'type': 'S4', 'failed_uav': failed_uav,
            'lost_targets': lost_targets,
            'affected_targets': lost_targets, 'severity': 1.0,
            'trigger_time_frac': random.uniform(0.3, 0.7),
        }

    # ==================== 应用突发事件 ====================

    def apply_emergency(self, emergency):
        """将突发事件应用到热点列表

        Returns:
            modified_hotspots, active_uavs, completed_targets
        """
        modified = copy.deepcopy(self.original_hotspots)
        active_uavs = [True] * NUM_UAVS
        completed_targets = set()

        etype = emergency['type']

        if etype == 'S1':
            for shift in emergency.get('shifts', []):
                tid = shift['id']
                for tgt in modified:
                    if tgt['id'] == tid:
                        cx, cy = tgt['center_km']
                        tgt['center_km'] = (cx + shift['dx'], cy + shift['dy'])
                        tgt.pop('spiral_time', None)
                        tgt.pop('spiral_dist', None)
                        tgt.pop('pdet', None)
                        break
        elif etype == 'S2':
            for new_tgt in emergency.get('new_targets', []):
                modified.append(copy.deepcopy(new_tgt))
        elif etype == 'S3':
            pass
        elif etype == 'S4':
            active_uavs[emergency['failed_uav']] = False

        return modified, active_uavs, completed_targets

    # ==================== 飞行仿真 ====================

    def simulate_until_emergency(self, ref_paths, emergency, baseline_routes=None,
                                   route_boundaries=None):
        """沿参考路径推算突发事件触发时的UAV位置，并判断已完成目标

        Args:
            ref_paths: 各UAV参考路径 (m坐标)
            emergency: 事件字典
            baseline_routes: 各UAV的路由 (目标ID列表), 用于推算已完成目标
            route_boundaries: 各UAV每个目标的累积路径距离(米), 精确判定用
        """
        trigger_frac = emergency.get('trigger_time_frac', 0.5)
        uav_positions = np.zeros((NUM_UAVS, 2))
        uav_headings = np.zeros(NUM_UAVS)

        path_lengths = []
        for path in ref_paths:
            if len(path) >= 2:
                seg_lens = np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))
                path_lengths.append(np.sum(seg_lens))
            else:
                path_lengths.append(0.0)

        for k, path in enumerate(ref_paths):
            if len(path) < 2 or path_lengths[k] < 1.0:
                uav_positions[k] = list(self.uav_base_km)
                uav_headings[k] = 0.0
                continue

            target_dist = trigger_frac * path_lengths[k]
            cumsum = 0.0
            for i in range(len(path) - 1):
                seg_len = np.hypot(path[i + 1, 0] - path[i, 0],
                                    path[i + 1, 1] - path[i, 1])
                if cumsum + seg_len >= target_dist:
                    frac = (target_dist - cumsum) / max(seg_len, 1e-9)
                    x = path[i, 0] + frac * (path[i + 1, 0] - path[i, 0])
                    y = path[i, 1] + frac * (path[i + 1, 1] - path[i, 1])
                    uav_positions[k] = np.array([x / 1000.0, y / 1000.0])
                    uav_headings[k] = math.atan2(
                        path[i + 1, 1] - path[i, 1],
                        path[i + 1, 0] - path[i, 0])
                    break
                cumsum += seg_len
            else:
                uav_positions[k] = np.array([path[-1, 0] / 1000.0,
                                              path[-1, 1] / 1000.0])
                uav_headings[k] = 0.0

        # ---- 判断已完成目标 ----
        completed = set()
        if baseline_routes is not None:
            for k, route in enumerate(baseline_routes):
                if not route or path_lengths[k] < 1.0:
                    continue
                trigger_dist = trigger_frac * path_lengths[k]
                if route_boundaries is not None and k < len(route_boundaries) \
                        and len(route_boundaries[k]) == len(route):
                    # 精确判定：使用预计算的累积距离边界
                    for tgt_i, tgt_id in enumerate(route):
                        if trigger_dist >= route_boundaries[k][tgt_i]:
                            completed.add(tgt_id)
                else:
                    # 近似判定：按路径比例均匀分配
                    n = len(route)
                    for tgt_i, tgt_id in enumerate(route):
                        end_frac = (tgt_i + 1) / n
                        if trigger_frac >= end_frac:
                            completed.add(tgt_id)

        remaining = set(h['id'] for h in self.original_hotspots) - completed

        # ---- 计算各UAV已消耗航程 (km) ----
        consumed_ranges = [trigger_frac * pl / 1000.0 for pl in path_lengths]

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

        return uav_positions, uav_headings, completed, remaining, consumed_ranges, active_target_ids, progress_kms

    # ==================== 代价计算 ====================

    def compute_cost_for_assignment(self, assignment_routes, hotspots, active_uavs,
                                     emergency=None, consumed_ranges=None,
                                     uav_positions=None, baseline_routes=None, completed=None):
        """计算给定分配方案的代价，含航程约束校验

        Args:
            assignment_routes: 各UAV目标ID列表
            hotspots: 热点列表
            active_uavs: bool[4]
            emergency: 突发事件信息
            consumed_ranges: 各UAV已消耗航程(km), None则视为0
            uav_positions: (4,2) 各UAV触发时位置(km), 用于计算剩余航程
            baseline_routes: 各UAV基准航线
            completed: 已完成的目标ID集合

        Returns:
            dict: J_max, J_sum, weighted_arrival_sum, routes, finish_times,
                  range_violations, max_range_ratio
        """
        id_to_h = {h['id']: h for h in hotspots}
        nfz_violations = 0
        # 确保所有目标有螺旋缓存
        for h in hotspots:
            if 'spiral_dist' not in h:
                h['spiral_dist'] = h.get('spiral_time', 500) * UAV_V_M_S

        finish_times = np.zeros(NUM_UAVS)
        weighted_arrival_sum = 0.0
        route_dists_km = np.zeros(NUM_UAVS)

        start_positions = uav_positions if uav_positions is not None \
            else [self.uav_base_km] * NUM_UAVS
        base_ranges = consumed_ranges if consumed_ranges is not None \
            else [0.0] * NUM_UAVS

        # 废弃 GA 层，基于真实空间坐标手动推演（修复瞬移 + 螺旋双重计算）
        for k, route in enumerate(assignment_routes):
            if not route or k >= len(active_uavs) or not active_uavs[k]:
                continue

            curr_pos = start_positions[k]
            uav_future_dist = 0.0
            uav_future_time = 0.0

            active_target_id = None
            if baseline_routes and completed is not None and k < len(baseline_routes):
                for b_tid in baseline_routes[k]:
                    if b_tid not in completed:
                        active_target_id = b_tid
                        break

            for i, tgt_id in enumerate(route):
                if tgt_id not in id_to_h:
                    continue
                tgt = id_to_h[tgt_id]

                dist_to_tgt = np.hypot(curr_pos[0] - tgt['center_km'][0],
                                       curr_pos[1] - tgt['center_km'][1])
                spiral_m = tgt['spiral_dist']  # meters
                spiral_km = spiral_m / 1000.0
                spiral_time = tgt.get('spiral_time', spiral_km / UAV_VELOCITY_KM_S)

                actual_dist = dist_to_tgt
                actual_spiral_km = spiral_km
                actual_spiral_time = spiral_time

                # 🌟 核心修复 1：消除螺旋重叠计算 (Double-Counting)
                # 如果这是剩余航线的第一个目标，且无人机距离目标非常近（说明正在执行它）
                if i == 0 and tgt_id == active_target_id and consumed_ranges is not None and baseline_routes is not None:
                    comp_targets = []
                    for b_tid in baseline_routes[k]:
                        if b_tid == active_target_id:
                            break
                        comp_targets.append(b_tid)
                        
                    comp_spiral_km = sum([id_to_h.get(t, {}).get('spiral_dist', 500*50.0)/1000.0 for t in comp_targets])
                    
                    transit_km = 0.0
                    p = self.uav_base_km
                    for t in comp_targets:
                        pos = id_to_h.get(t, {}).get('center_km', p)
                        transit_km += np.hypot(pos[0] - p[0], pos[1] - p[1])
                        p = pos
                    transit_km += np.hypot(tgt['center_km'][0] - p[0], tgt['center_km'][1] - p[1])
                    
                    progress_km = base_ranges[k] - comp_spiral_km - transit_km
                    
                    if progress_km > 0:
                        progress_ratio = min(1.0, progress_km / max(spiral_km, 1e-5))
                        actual_spiral_km = max(0.0, spiral_km - progress_km)
                        actual_spiral_time = actual_spiral_time * (1.0 - progress_ratio)
                        actual_dist = 0.0

                # S3 禁飞区绕飞
                if emergency and emergency['type'] == 'S3':
                    nf_cx, nf_cy = emergency['no_fly_center']
                    nf_r = emergency['no_fly_radius']
                    if _segment_intersects_circle(curr_pos, tgt['center_km'], nf_cx, nf_cy, nf_r):
                        nfz_violations += 1
                        uav_future_dist += 500.0 # 给局部计算施加巨大物理阻力
                        uav_future_time += 10000.0 
                    else:
                        # 只有在边缘情况才应用常规绕飞
                        detour = compute_detour_distance_km(curr_pos, tgt['center_km'], nf_cx, nf_cy, nf_r)
                        uav_future_dist += detour
                        uav_future_time += detour / UAV_VELOCITY_KM_S

                # 紧急度加权（与 GA 公式对齐）
                amplified_weight = 10 ** (4 * tgt['weight'])
                weighted_arrival_sum += uav_future_time * amplified_weight

                curr_pos = tgt['center_km']

            route_dists_km[k] = uav_future_dist
            finish_times[k] = (base_ranges[k] / UAV_VELOCITY_KM_S) + uav_future_time

        J_max = np.max(finish_times[:len(active_uavs)]) if any(finish_times[:len(active_uavs)] > 0) else 0.0
        J_sum = np.sum(finish_times[:len(active_uavs)])

        # ---- 航程约束校验 ----
        # 🌟 核心修复 2：准确核算包含历史损耗的全局航程
        total_ranges = [base_ranges[k] + route_dists_km[k] for k in range(NUM_UAVS)]
        range_violations = sum(1 for r in total_ranges if r > MAX_RANGE_KM)
        max_range_ratio = max([r / MAX_RANGE_KM for r in total_ranges]) if total_ranges else 0.0

        return {
            'J_max': J_max, 'J_sum': J_sum,
            'weighted_arrival_sum': weighted_arrival_sum,
            'finish_times': finish_times,
            'range_violations': range_violations,
            'nfz_violations': nfz_violations, # 新增
            'max_range_ratio': max_range_ratio,
        }

    def compute_oracle_cost(self, remaining_targets, active_uavs,
                            modified_hotspots, weights=(0.5, 0.3, 0.4),
                            consumed_ranges=None):
        """通过GA重优化获取最优代价"""
        n_active = sum(active_uavs)
        if n_active == 0:
            return float('inf')

        ga_targets = []
        for h in modified_hotspots:
            ga_targets.append({
                'id': h['id'], 'pos': h['center_km'],
                'weight': h['weight'], 'radius_km': h['radius_km'],
            })

        if isinstance(remaining_targets, (set, list)):
            remaining_ids = set(remaining_targets if isinstance(remaining_targets, set)
                                else [t['id'] for t in remaining_targets])
            ga_targets = [t for t in ga_targets if t['id'] in remaining_ids]

        if not ga_targets:
            return 0.0

        for tgt in ga_targets:
            matching = [h for h in modified_hotspots if h['id'] == tgt['id']]
            if matching and 'spiral_time' in matching[0]:
                tgt['spiral_time'] = matching[0]['spiral_time']
                tgt['spiral_dist'] = matching[0].get('spiral_dist',
                                                      matching[0]['spiral_time'] * UAV_V_M_S)
            else:
                sol = spiral_search_arc_exact(
                    prob_grid=self.prob_grid, center_km=tgt['pos'],
                    radius_km=tgt['radius_km'], X_km=self.X_km, Y_km=self.Y_km,
                    yaw_at_center=0.0, cfg=self.spiral_cfg,
                )
                tgt['spiral_time'] = sol['totalTime']
                tgt['spiral_dist'] = sol['totalLen']

        # 每架UAV独立d_max：扣除已消耗航程
        full_d_max_s = MAX_RANGE_KM / UAV_VELOCITY_KM_S
        if consumed_ranges is not None and len(consumed_ranges) >= n_active:
            per_uav_d_max = [full_d_max_s - consumed_ranges[k] / UAV_VELOCITY_KM_S
                             for k in range(n_active)]
            d_max_s = min(per_uav_d_max)  # GA用单一d_max，取最紧约束
        else:
            d_max_s = full_d_max_s
        ga.init_ga_env(ga_targets, self.uav_base_km, n_active,
                       prob_grid=None, X_km=None, Y_km=None,
                       spiral_cfg=None, weights=weights,
                       d_max=d_max_s, c_drop=GLOBAL_C_DROP)

        try:
            best_chrom, best_cost, _, _ = ga.run_ga(pop_size=80, generations=150,
                                                     patience=50)
            J_max, J_sum, w_arrival, _, _, _, _ = ga.calculate_raw_costs(best_chrom)
            return {'J_max': J_max, 'J_sum': J_sum,
                    'weighted_arrival_sum': w_arrival, 'cost': best_cost}
        except Exception as e:
            print(f"[WARNING] Oracle GA failed: {e}, using fallback")
            return {'J_max': 30000.0, 'J_sum': 120000.0,
                    'weighted_arrival_sum': 3e8, 'cost': 1.0}

    def compute_step_reward(self, action, target, consumed_ranges, actual_extra_km=0.0):
        """中间步骤即时奖励：基于真实增量物理代价 + 航程pressure"""
        NORM = 5000.0  # 归一化因子

        if action == 0:
            # 放弃 → 机会成本
            return -target.get('weight', 0.3) * 2.0

        uav_idx = action - 1
        
        # 直接使用传入的真实增量距离计算时间
        extra_time = actual_extra_km / UAV_VELOCITY_KM_S

        # 🌟 核心修复: 改用边际压力 (Marginal Pressure)
        range_before = consumed_ranges[uav_idx]
        range_after = consumed_ranges[uav_idx] + actual_extra_km
        
        p_before = max(0.0, range_before - SOFT_RANGE_KM) / max(MAX_RANGE_KM - SOFT_RANGE_KM, 1.0)
        p_after = max(0.0, range_after - SOFT_RANGE_KM) / max(MAX_RANGE_KM - SOFT_RANGE_KM, 1.0)
        
        marginal_pressure = p_after - p_before

        return -(extra_time / NORM + marginal_pressure * 1.0)

    def compute_reward(self, dqn_cost, oracle_cost=None, baseline_val=None,
                        abandoned_targets=None):
        """终端奖励：字典序优先级 (连续梯度版)

        P1a: 航程硬约束 — 越限数量线性惩罚 (-2.0/UAV), 0→安全 4→坠毁
        P1b: 覆盖优先级 — sum(w) * 5.0, 放弃高权重>>弃低权重
        P2:  效能寻优 — 0.5*J_max_norm + 0.3*J_sum_norm + 0.4*w_norm
        """
        range_violations = dqn_cost.get('range_violations', 0)
        nfz_violations = dqn_cost.get('nfz_violations', 0)

        # P1a: 增加禁飞区坠毁惩罚 (极其严厉)
        FATAL_PER_UAV = 4.0
        fatal_penalty = range_violations * FATAL_PER_UAV + nfz_violations * 20.0

        # P1b: 覆盖优先级
        COVERAGE_PENALTY = 5.0
        abandon_penalty = 0.0
        if abandoned_targets:
            abandon_penalty = sum(t.get('weight', 0.3) for t in abandoned_targets) * COVERAGE_PENALTY

        # P2: 效能寻优归一化
        ref_J_max = 10000.0
        ref_J_sum = 40000.0
        # 🌟 修复：将其改为固定的经验常数（推荐 5e7 或 1e8），绝对不能使用 dqn_cost 去动态除自己！
        ref_w = 50000000.0

        if oracle_cost is not None and isinstance(oracle_cost, dict) and oracle_cost.get('J_max', 0) > 0:
            ref_J_max = max(ref_J_max, oracle_cost['J_max'])
            ref_J_sum = max(ref_J_sum, oracle_cost['J_sum'])
            # 如果 GA 给出了更好的（更低的）加权代价，也可以作为基准参考
            if oracle_cost.get('weighted_arrival_sum', 0) > 0:
                ref_w = max(ref_w, oracle_cost['weighted_arrival_sum'])
        elif baseline_val is not None and baseline_val > 0:
            ref_J_max = max(ref_J_max, baseline_val)

        J_max_norm = dqn_cost['J_max'] / max(ref_J_max, 1.0)
        J_sum_norm = dqn_cost['J_sum'] / max(ref_J_sum, 1.0)
        w_norm = dqn_cost.get('weighted_arrival_sum', 0.0) / max(ref_w, 1.0)
        efficiency_cost = 0.5 * J_max_norm + 0.3 * J_sum_norm + 0.4 * w_norm

        reward = -(fatal_penalty + abandon_penalty + efficiency_cost)
        return reward

    # ==================== V2 轮询MDP接口 ====================

    def reset_v2(self, uav_positions_km, consumed_ranges_km, hotspots, active_uavs=None, completed=None,
                 active_target_ids=None, progress_kms=None, emergency=None):
        """
        初始化 V2 版本的 MDP 环境状态，建立物理对齐的初始观测。
        """
        self.emergency = emergency

        # 1. 初始化 4 架无人机的状态矩阵 (4x3: x, y, 剩余航程比)
        self.uav_states = np.zeros((4, 3), dtype=np.float32)
        for k in range(4):
            self.uav_states[k, 0] = uav_positions_km[k][0] / SPAN_KM
            self.uav_states[k, 1] = uav_positions_km[k][1] / SPAN_KM
            # 计算真实的剩余航程比例，确保 DQN 知道自己还有多少“油”
            self.uav_states[k, 2] = (MAX_RANGE_KM - consumed_ranges_km[k]) / MAX_RANGE_KM

        # 2. 构建全局目标池，处理已完成目标的预掩码
        completed_set = set(completed) if completed else set()
        self.targets = []
        for h in hotspots:
            # 🌟 修复点 2：解决 S1/S2 场景下 spiral_dist 缺失导致的“白嫖”搜索问题
            # 如果目标是新增的或位移过的，可能没有预计算的距离，需进行物理对齐的估算
            s_dist = h.get('spiral_dist')
            if s_dist is None:
                # 默认值需与 compute_cost_for_assignment 逻辑严格对齐：时间 * 速度
                s_dist = h.get('spiral_time', 500) * UAV_V_M_S
            self.targets.append({
                'id': h['id'],
                'center_km': h['center_km'],
                'weight': h['weight'],
                'spiral_dist': s_dist,
                'mask': h['id'] in completed_set
            })

        # 🌟 修复点 3：同步物理环境中的存活状态，彻底杜绝 S4 故障无人机在虚拟推演中复活
        self.uav_active = list(active_uavs) if active_uavs is not None else [True] * 4

        # 3. 确定起始决策的无人机
        # 初始只从“物理存活”的无人机中随机抽取一个作为首发
        valid_uavs = [i for i, a in enumerate(self.uav_active) if a]
        self.current_uav_idx = random.choice(valid_uavs) if valid_uavs else 0

        # 4. 处理半途搜索的“硬锁定”逻辑
        self.locked_target_idxs: list = [None] * 4  # 使用 list 声明绕过 Pylance 类型检查
        if active_target_ids and progress_kms:
            # 建立 ID 到当前 targets 列表索引的映射
            id_to_idx = {t['id']: i for i, t in enumerate(self.targets) if 'id' in t}
            for k in range(4):
                tid = active_target_ids[k]
                pkm = progress_kms[k]
                if tid is not None and pkm > 0 and tid not in completed_set:
                    idx = id_to_idx.get(tid)
                    # 🌟 修复点 4：仅当无人机存活时才锁定。S4 故障机的进度作废，目标释放给他人。
                    if self.uav_active[k] and idx is not None:
                        self.locked_target_idxs[k] = idx
                        # 从目标的所需搜索距离中，真实扣除掉已经飞过的里程
                        self.targets[idx]['spiral_dist'] = max(0.0, self.targets[idx]['spiral_dist'] - pkm * 1000.0)

        # 🌟 核心修复点 5：初始合法性校验循环
        # 必须确保第一个开始决策的无人机不是“空油”或“无路可走”的状态。
        # 如果当前无人机无合法动作（比如在禁飞区对面且油不够），则自动流转给下一个有能力的无人机。
        checked = 0
        while checked < 4:
            if self.uav_active[self.current_uav_idx]:
                # 传入 self.emergency，使校验能够识别 S3 禁飞区的阻力
                all_locked = [idx for idx in self.locked_target_idxs if idx is not None]
                valid = get_valid_actions_v2(
                    self.current_uav_idx, self.uav_states, self.targets,
                    locked_target_idx=self.locked_target_idxs[self.current_uav_idx],
                    emergency=self.emergency,
                    all_locked_idxs=all_locked
                )
                if valid.any():
                    break # 找到首个能干活的飞机，跳出
                # 如果没活干了，标记为休眠并轮询下一个
                self.uav_active[self.current_uav_idx] = False
            
            self.current_uav_idx = (self.current_uav_idx + 1) % 4
            checked += 1
            
        return build_state_vector_v2(self.uav_states, self.targets, self.current_uav_idx)

    def step_v2(self, action):
        """执行一步V2 MDP决策"""
        t = self.targets[action]
        dist_km, _ = apply_action_v2(self.current_uav_idx, action, self.uav_states, self.targets, emergency=self.emergency)
        reward = ALPHA_REWARD * t['weight'] - BETA_REWARD * (dist_km / MAX_DIAG_KM)

        # 抢夺惩罚
        cx, cy = t['center_km']
        ux = self.uav_states[self.current_uav_idx][0] * SPAN_KM
        uy = self.uav_states[self.current_uav_idx][1] * SPAN_KM
        d_current = np.hypot(cx - ux, cy - uy)
        d_others = [
            np.hypot(cx - self.uav_states[k][0]*SPAN_KM, cy - self.uav_states[k][1]*SPAN_KM)
            for k in range(4) if k != self.current_uav_idx and self.uav_active[k]
        ]
        if d_others:
            min_other = min(d_others)
            if d_current > min_other * 1.2 and (d_current - min_other) > 2.0:
                reward -= 0.3 * t['weight']

        # 解除当前UAV的锁定
        self.locked_target_idxs[self.current_uav_idx] = None

        done = False
        if all(tgt['mask'] for tgt in self.targets):
            done = True
            reward += R_SUCCESS
        else:
            checked = 0
            while checked < 4:
                self.current_uav_idx = (self.current_uav_idx + 1) % 4
                # 🌟 修复：必须先检查该无人机是否还存活，防止 S4 坠毁飞机复活
                if self.uav_active[self.current_uav_idx]:
                    all_locked = [idx for idx in self.locked_target_idxs if idx is not None]
                    valid = get_valid_actions_v2(
                        self.current_uav_idx, self.uav_states, self.targets,
                        locked_target_idx=self.locked_target_idxs[self.current_uav_idx],
                        emergency=self.emergency,
                        all_locked_idxs=all_locked
                    )
                    if valid.any():
                        break
                    self.uav_active[self.current_uav_idx] = False
                checked += 1
            else:
                done = True

            if done:
                unvisited_w = sum(tgt['weight'] for tgt in self.targets if not tgt['mask'])
                reward -= GAMMA_UNVISITED * unvisited_w

        next_state = build_state_vector_v2(self.uav_states, self.targets, self.current_uav_idx)
        return next_state, reward, done, {'current_uav': self.current_uav_idx}
