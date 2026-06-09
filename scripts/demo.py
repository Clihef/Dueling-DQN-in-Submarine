"""
突发事件临机决策 — 可视化演示 (全物理气动轨迹优化版)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe  # 引入字体描边特效
from matplotlib.patches import Circle

from core import heatmap as hm
from core import ga_allocator as ga
from core.spiral_search import spiral_search_arc_exact
from core.dubins_planner import dubins_curve
from core.uav_navi import uav_navi_traverse
from emergency.agent import EmergencyDQN, NUM_UAVS
from emergency.simulator import EmergencySimulator
from emergency.utils import (
    get_affected_targets,
    replan_around_no_fly, compute_detour_waypoints_km, compute_route_distance_km,
    SPAN_KM, MAX_RANGE_KM, MAX_FLIGHT_TIME_S, GLOBAL_C_DROP,
    STATE_DIM_V2, N_MAX, get_valid_actions_v2
)

# ==================== 全局配置 ====================
UAV_BASE_KM = (0.0, 0.0)
UAV_V_M_S = 50.0
dx_km = 0.1

UNIFIED_HOTSPOTS = [
    {'id': 0, 'center_km': (15, 15), 'sigma': (3, 3), 'weight': 0.25, 'radius_km': 2.5},
    {'id': 1, 'center_km': (25, 10), 'sigma': (4, 4), 'weight': 0.30, 'radius_km': 3.0},
    {'id': 2, 'center_km': (10, 30), 'sigma': (2, 2), 'weight': 0.40, 'radius_km': 2.0},
    {'id': 3, 'center_km': (20, 60), 'sigma': (4, 4), 'weight': 0.50, 'radius_km': 3.5},
    {'id': 4, 'center_km': (30, 80), 'sigma': (6, 6), 'weight': 1.00, 'radius_km': 5.0},
    {'id': 5, 'center_km': (15, 85), 'sigma': (3, 3), 'weight': 0.35, 'radius_km': 2.5},
    {'id': 6, 'center_km': (60, 20), 'sigma': (5, 5), 'weight': 0.60, 'radius_km': 4.0},
    {'id': 7, 'center_km': (55, 40), 'sigma': (4, 4), 'weight': 0.55, 'radius_km': 3.0},
    {'id': 8, 'center_km': (75, 30), 'sigma': (6, 6), 'weight': 0.85, 'radius_km': 4.5},
    {'id': 9, 'center_km': (85, 85), 'sigma': (5, 5), 'weight': 0.95, 'radius_km': 4.0},
    {'id': 10, 'center_km': (75, 90), 'sigma': (3, 3), 'weight': 0.20, 'radius_km': 2.5},
    {'id': 11, 'center_km': (90, 70), 'sigma': (4, 4), 'weight': 0.30, 'radius_km': 3.5},
    {'id': 12, 'center_km': (50, 60), 'sigma': (3, 3), 'weight': 0.45, 'radius_km': 2.5},
    {'id': 13, 'center_km': (40, 20), 'sigma': (2, 2), 'weight': 0.20, 'radius_km': 2.0},
    {'id': 14, 'center_km': (65, 65), 'sigma': (4, 4), 'weight': 0.50, 'radius_km': 3.0},
]

spiral_cfg = {
    'uav': {'Vg': 50.0, 'H': 100.0, 'Rmin': 2000.0},
    'sensor': {'dcty': 600.0, 'hsub': 20.0, 'overlapFrac': 0.1},
    'target': {'vTgt': 5.0, 'epsR': 200.0, 'tau0': 0.0},
    'opt': {'Dmax': 150000.0, 'modeHard': True, 'gamma': 0.01,
            'deltaP_stop': 0.005, 'lambdaR0': 0.03},
    'grid': {'r0interval': 10, 'hinterval': 20, 'NGrid': list(range(1, 6))},
    'traj': {'turnDir': 1}
}


# ==================== 辅助：生成动力学航迹 ====================
def _dedupe_path_m(path_m, min_sep_m=5.0):
    path_m = np.asarray(path_m, dtype=float)
    if len(path_m) < 2:
        return path_m.copy()

    clean_path = [path_m[0]]
    for pt in path_m[1:]:
        if np.hypot(pt[0] - clean_path[-1][0], pt[1] - clean_path[-1][1]) > min_sep_m:
            clean_path.append(pt)
    return np.asarray(clean_path, dtype=float)


def _clip_path_to_demo_bounds_m(path_m, margin_m=200.0):
    """Clip only the visualized trajectory to the demo map bounds."""
    path_m = np.asarray(path_m, dtype=float)
    if len(path_m) == 0:
        return path_m.copy()
    lower = float(margin_m)
    upper = SPAN_KM * 1000.0 - float(margin_m)
    clipped = path_m.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], lower, upper)
    clipped[:, 1] = np.clip(clipped[:, 1], lower, upper)
    return _dedupe_path_m(clipped, min_sep_m=5.0)


def _path_out_of_demo_bounds_m(path_m, margin_m=0.0):
    path_m = np.asarray(path_m, dtype=float)
    if len(path_m) == 0:
        return False
    lower = float(margin_m)
    upper = SPAN_KM * 1000.0 - float(margin_m)
    return bool(
        np.any(path_m[:, 0] < lower)
        or np.any(path_m[:, 0] > upper)
        or np.any(path_m[:, 1] < lower)
        or np.any(path_m[:, 1] > upper)
    )


def _resample_polyline_m(path_m, step_m=200.0):
    """Create a stable visual polyline without invoking the low-level tracker."""
    path_m = np.asarray(path_m, dtype=float)
    if len(path_m) < 2:
        return path_m.copy()

    points = [path_m[0]]
    for p0, p1 in zip(path_m[:-1], path_m[1:]):
        seg = p1 - p0
        seg_len = float(np.hypot(seg[0], seg[1]))
        if seg_len <= 1e-9:
            continue
        n = max(1, int(np.ceil(seg_len / step_m)))
        for j in range(1, n + 1):
            points.append(p0 + (j / n) * seg)
    return np.asarray(points, dtype=float)


def _visual_reference_traj_m(ref_path_m):
    bounded_ref = _clip_path_to_demo_bounds_m(ref_path_m)
    return _resample_polyline_m(bounded_ref, step_m=200.0)


def fill_feasible_baseline_abandoned_targets(sim, routes, hotspots):
    """Deterministically insert GA-abandoned targets when range-feasible.

    The GA chromosome has an abandon bucket. For demo baseline visualization we
    want a coverage-first initial plan, so any target that can be inserted
    without violating the range constraint is restored after GA search.
    """
    routes = [list(route) for route in routes]
    target_ids = {int(h['id']) for h in hotspots}
    weights = {int(h['id']): float(h.get('weight', 0.0)) for h in hotspots}
    active_uavs = [True] * NUM_UAVS
    consumed_ranges = [0.0] * NUM_UAVS
    uav_positions = [UAV_BASE_KM] * NUM_UAVS
    inserted = []

    def route_cost(candidate_routes):
        return sim.compute_cost_for_assignment(
            candidate_routes,
            hotspots,
            active_uavs,
            consumed_ranges=consumed_ranges,
            uav_positions=uav_positions,
        )

    while True:
        assigned = {int(tid) for route in routes for tid in route}
        abandoned = sorted(
            target_ids - assigned,
            key=lambda tid: weights.get(tid, 0.0),
            reverse=True,
        )
        if not abandoned:
            return routes, inserted

        best_insert = None
        for target_id in abandoned:
            for uav_idx in range(NUM_UAVS):
                for pos in range(len(routes[uav_idx]) + 1):
                    trial = [list(route) for route in routes]
                    trial[uav_idx].insert(pos, int(target_id))
                    cost = route_cost(trial)
                    if int(cost.get('range_violations', 0)) != 0:
                        continue
                    if float(cost.get('max_range_ratio', 2.0)) > 1.0:
                        continue
                    total_ranges = np.asarray(cost.get('total_ranges', np.zeros(NUM_UAVS)), dtype=float)
                    candidate = (
                        -weights.get(target_id, 0.0),
                        float(cost.get('J_max', 0.0)),
                        float(cost.get('J_sum', 0.0)),
                        float(np.max(total_ranges)) if total_ranges.size else 0.0,
                        target_id,
                        trial,
                    )
                    if best_insert is None or candidate < best_insert:
                        best_insert = candidate

        if best_insert is None:
            return routes, inserted

        _, _, _, _, target_id, routes = best_insert
        inserted.append(int(target_id))


def generate_aero_traj(ref_path_m, v_uav):
    """提取稀疏参考点，传入动力学模型解算实际飞行轨迹。

    This function is used only for demo visualization. If the aerodynamic
    tracker creates a visibly invalid path (out of map bounds or truncated), we
    fall back to a bounded reference trajectory so the figure reflects the
    planned route order without visual artifacts.
    """
    if len(ref_path_m) < 2:
        return np.empty((0, 2))

    ref_path_clean = _dedupe_path_m(ref_path_m, min_sep_m=5.0)
    visual_ref_path = _clip_path_to_demo_bounds_m(ref_path_clean)
    if len(visual_ref_path) < 2:
        return np.empty((0, 2))

    # 估算总时间和步长。连续动力学跟踪会在 Dubins/螺旋密集航点处产生
    # 额外转向耗时，时间预算过紧会截断最后一个目标的搜索轨迹。
    total_dist = np.sum(np.hypot(np.diff(visual_ref_path[:, 0]), np.diff(visual_ref_path[:, 1])))
    total_sim_time = (total_dist / v_uav) * 2.0 + 120.0

    actual_traj = uav_navi_traverse(visual_ref_path, total_sim_time, 0.02, v_uav, 150.0)
    if len(actual_traj) == 0:
        return _visual_reference_traj_m(visual_ref_path)

    terminal_error = np.hypot(
        actual_traj[-1, 0] - visual_ref_path[-1, 0],
        actual_traj[-1, 1] - visual_ref_path[-1, 1],
    )
    visual_bad = (
        _path_out_of_demo_bounds_m(actual_traj)
        or terminal_error > 500.0
    )
    if visual_bad:
        print(
            f"[WARN] Demo visualization fallback: "
            f"terminal error = {terminal_error / 1000.0:.2f} km, "
            f"out_of_bounds = {_path_out_of_demo_bounds_m(actual_traj)}"
        )
        return _visual_reference_traj_m(visual_ref_path)

    return _clip_path_to_demo_bounds_m(actual_traj)


# ==================== 演示主函数 ====================
def trim_polyline_prefix(path_m, trim_m):
    """从完整螺旋搜索路径中移除已搜索的举例，沿多段线累计长度裁掉前缀，并在裁剪点做线性插值，保证剩余路径从真实续搜点开始，而不是粗暴从某个离散点开始"""
    path_m = np.asarray(path_m, dtype=float)
    if len(path_m) < 2 or trim_m <= 0:
        return path_m

    remaining = float(trim_m)
    for i in range(len(path_m) - 1):
        p0 = path_m[i]
        p1 = path_m[i + 1]
        seg_len = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if seg_len <= 1e-9:
            continue
        if remaining <= seg_len:
            frac = remaining / seg_len
            start = p0 + frac * (p1 - p0)
            return np.vstack([start, path_m[i + 1:]])
        remaining -= seg_len

    return path_m[-1:].copy()


def polyline_length_m(path_m):
    path_m = np.asarray(path_m, dtype=float)
    if len(path_m) < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(path_m[:, 0]), np.diff(path_m[:, 1]))))


def slice_polyline_by_distance(path_m, start_m, end_m):
    """Return a path slice between two cumulative distances on a polyline."""
    path_m = np.asarray(path_m, dtype=float)
    if len(path_m) < 2:
        return path_m.copy()

    start_m = max(0.0, float(start_m))
    end_m = min(float(end_m), polyline_length_m(path_m))
    if end_m <= start_m:
        return path_m[-1:].copy()

    points = []
    cumsum = 0.0
    for i in range(len(path_m) - 1):
        p0 = path_m[i]
        p1 = path_m[i + 1]
        seg = p1 - p0
        seg_len = float(np.hypot(seg[0], seg[1]))
        if seg_len <= 1e-9:
            continue
        next_cumsum = cumsum + seg_len

        if next_cumsum >= start_m and cumsum <= end_m:
            local_start = max(start_m, cumsum)
            local_end = min(end_m, next_cumsum)
            a = (local_start - cumsum) / seg_len
            b = (local_end - cumsum) / seg_len
            start_pt = p0 + a * seg
            end_pt = p0 + b * seg
            if not points or np.hypot(*(start_pt - points[-1])) > 1e-6:
                points.append(start_pt)
            points.append(end_pt)

        if next_cumsum >= end_m:
            break
        cumsum = next_cumsum

    return np.asarray(points, dtype=float)

def run_emergency_demo(emergency_type='random', model_path='outputs/models/emergency_dqn_model.pt',
                       show=True):
    # ---- 0. 全局绘图规范配置 ----
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'axes.labelsize': 13,
        'axes.titlesize': 15,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 18
    })
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"突发事件演示模式 (全物理气动版)")
    print(f"类型: {emergency_type}, 设备: {device}")

    # ---- 1. 初始化环境 ----
    span_km = SPAN_KM
    x_km = np.arange(0, span_km + dx_km / 2, dx_km)
    y_km = np.arange(0, span_km + dx_km / 2, dx_km)
    X_km, Y_km = np.meshgrid(x_km, y_km)

    hm_hotspots = [{'center': h['center_km'], 'sigma': h['sigma'],
                    'weight': h['weight']} for h in UNIFIED_HOTSPOTS]
    np.random.seed(42)
    prob_grid = hm.generate_controlled_prob_field(X_km, Y_km, hm_hotspots, alpha=0.15)

    sim = EmergencySimulator(UNIFIED_HOTSPOTS, NUM_UAVS, UAV_BASE_KM, spiral_cfg,
                             prob_grid, X_km, Y_km)

    # ---- 2. 加载DQN模型 ----
    dqn = EmergencyDQN(state_dim=STATE_DIM_V2, num_actions=N_MAX).to(device)
    if os.path.exists(model_path):
        dqn.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        print(f"DQN模型已加载: {model_path}")
    else:
        print(f"警告: 未找到模型 {model_path}, 使用未训练权重")

    # ---- 3. 运行GA获取原始计划 ----
    print("运行GA获取原始分配...")
    ga_targets = [{'id': h['id'], 'pos': h['center_km'],
                   'weight': h['weight'], 'radius_km': h['radius_km']}
                  for h in UNIFIED_HOTSPOTS]
    ga.init_ga_env(ga_targets, UAV_BASE_KM, NUM_UAVS,
                   prob_grid=prob_grid, X_km=X_km, Y_km=Y_km,
                   spiral_cfg=spiral_cfg, weights=(0.5, 0.3, 0.4), d_max=MAX_FLIGHT_TIME_S, c_drop=GLOBAL_C_DROP)
    best_chrom, best_cost, _, norm_factors = ga.run_ga(pop_size=200, generations=300, patience=100)
    _, raw_routes, _ = ga.evaluate_chromosome(best_chrom, norm_factors)
    baseline_routes = raw_routes[:NUM_UAVS]
    baseline_routes, baseline_inserted = fill_feasible_baseline_abandoned_targets(
        sim, baseline_routes, UNIFIED_HOTSPOTS
    )
    if baseline_inserted:
        print(f"GA baseline feasible-fill inserted targets: {baseline_inserted}")
    print(f"原始分配: {baseline_routes}")

    # ---- 4. 生成基线参考路径 & 动力学解算 (供左图展示) ----
    ref_paths = []
    route_boundaries_list = []
    for k, route in enumerate(baseline_routes):
        if not route:
            ref_paths.append(np.empty((0, 2)))
            route_boundaries_list.append([])
            continue
        uav_path = []
        boundaries = []
        cumulative_dist = 0.0
        curr_pose = [UAV_BASE_KM[0] * 1000.0, UAV_BASE_KM[1] * 1000.0, 0.0]
        
        for tgt_idx in route:
            tgt = UNIFIED_HOTSPOTS[tgt_idx]
            target_m = [tgt['center_km'][0] * 1000.0, tgt['center_km'][1] * 1000.0]
            yaw_to_tgt = math.atan2(target_m[1] - curr_pose[1], target_m[0] - curr_pose[0])
            
            spiral_sol = spiral_search_arc_exact(
                prob_grid, tgt['center_km'], tgt['radius_km'],
                X_km, Y_km, yaw_to_tgt, spiral_cfg
            )
            s_path = spiral_sol['path'][::10]
            s_start_pos = spiral_sol['path'][0]
            s_start_yaw = math.atan2(spiral_sol['path'][1][1] - spiral_sol['path'][0][1],
                                     spiral_sol['path'][1][0] - spiral_sol['path'][0][0])
            
            dubins_goal = [s_start_pos[0], s_start_pos[1], s_start_yaw]
            d_path, d_len = dubins_curve(curr_pose, dubins_goal, r=spiral_cfg['uav']['Rmin'], stepsize=200.0)
            if d_path is not None:
                uav_path.extend(d_path[:, 0:2].tolist())
                cumulative_dist += d_len
                
            uav_path.extend(s_path.tolist())
            cumulative_dist += spiral_sol['totalLen']
            boundaries.append(cumulative_dist)
            curr_pose = [spiral_sol['pEnd'][0], spiral_sol['pEnd'][1], spiral_sol['yawEnd']]
            
        ref_paths.append(np.array(uav_path))
        route_boundaries_list.append(boundaries)

    print("解算基线规划的动力学实体轨迹...")
    actual_baseline_trajectories = [generate_aero_traj(p, UAV_V_M_S) for p in ref_paths]

    # ---- 5. 生成突发事件与推算触发位置 ----
    if emergency_type == 'random':
        emergency = sim.generate_random_emergency(baseline_routes, ref_paths)
    elif emergency_type == 'S1': emergency = sim._generate_s1()
    elif emergency_type == 'S2': emergency = sim._generate_s2()
    elif emergency_type == 'S3': emergency = sim._generate_s3(baseline_routes)
    elif emergency_type == 'S4': emergency = sim._generate_s4(baseline_routes)

    etype = emergency['type']
    print(f"\n突发事件: {etype}, 严重度: {emergency.get('severity', 'N/A')}")

    uav_positions, uav_headings, completed, remaining, consumed_ranges, active_target_ids, progress_kms = \
        sim.simulate_until_emergency(ref_paths, emergency, baseline_routes, route_boundaries_list)

    if etype == 'S1':
        active_ids = {tid for tid in active_target_ids if tid is not None}
        shift_eligible = set(remaining) - active_ids
        if not shift_eligible:
            shift_eligible = set(remaining)
        emergency = sim._generate_s1(
            eligible_ids=shift_eligible,
            trigger_time_frac=emergency.get('trigger_time_frac', 0.5),
        )

    modified_hotspots, active_uavs, _ = sim.apply_emergency(emergency)
    remaining_ids = set(remaining)
    if etype == 'S2':
        for nt in emergency.get('new_targets', []): remaining_ids.add(nt['id'])

    # ---- 6. DQN轮询MDP决策 ----
    dqn_routes = [[tid for tid in r if tid not in completed] for r in baseline_routes]
    if etype == 'S4': dqn_routes[emergency['failed_uav']] = []

    abandoned = []
    decisions = []
    id_to_hotspot = {h['id']: h for h in modified_hotspots}
    for h in UNIFIED_HOTSPOTS:
        if h['id'] not in id_to_hotspot: id_to_hotspot[h['id']] = h

    local_detour_feasible = False
    if etype == 'S3':
        clean_routes, local_detour_feasible, _ = sim.s3_local_detour_routes(
            dqn_routes, active_uavs, consumed_ranges,
            uav_positions=uav_positions, emergency=emergency,
            baseline_routes=baseline_routes, completed=completed,
            active_target_ids=active_target_ids, progress_kms=progress_kms,
        )

    if not local_detour_feasible:
        state = sim.reset_v2(uav_positions, consumed_ranges, modified_hotspots,
                             active_uavs=active_uavs, completed=completed,
                             active_target_ids=active_target_ids, progress_kms=progress_kms,
                             emergency=emergency)
        done = False
        while not done:
            all_locked = [idx for idx in sim.locked_target_idxs if idx is not None]
            valid_mask = get_valid_actions_v2(sim.current_uav_idx, sim.uav_states, sim.targets,
                                              locked_target_idx=sim.locked_target_idxs[sim.current_uav_idx],
                                              emergency=emergency, all_locked_idxs=all_locked)
            if not valid_mask.any(): break
            with torch.no_grad():
                q_values = dqn(torch.FloatTensor(state).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
            action = int(np.argmax(np.where(valid_mask, q_values, -np.inf)))
            t = sim.targets[action]
            decisions.append((t, sim.current_uav_idx))
            state, _, done, _ = sim.step_v2(action)

        clean_routes = [[] for _ in range(NUM_UAVS)]
        for t, uav_idx in decisions:
            clean_routes[uav_idx].append(t['id'])

    dqn_cost = sim.compute_cost_for_assignment(
        clean_routes, modified_hotspots, active_uavs, emergency,
        consumed_ranges, uav_positions=uav_positions, baseline_routes=baseline_routes,
        completed=completed, active_target_ids=active_target_ids,
        progress_kms=progress_kms)
    print(f"DQN分配 (重排序后): {clean_routes}")

    # ---- 7. DQN 重规划参考路径 & 动力学解算 (供右图展示) ----
    print("解算DQN规划的动力学实体轨迹...")
    detour_paths = []
    if etype == 'S3':
        detour_paths = replan_around_no_fly(clean_routes, uav_positions, emergency['no_fly_center'], emergency['no_fly_radius'], id_to_hotspot)

    actual_dqn_trajectories = []
    for k, route in enumerate(clean_routes):
        if not route:
            actual_dqn_trajectories.append(np.empty((0,2)))
            continue
        
        curr_pos_m = [uav_positions[k][0] * 1000.0, uav_positions[k][1] * 1000.0]
        curr_yaw = uav_headings[k]
        curr_pose = [curr_pos_m[0], curr_pos_m[1], curr_yaw]
        full_path_m = [curr_pos_m]

        # 遍历目标生成 Dubins + 螺旋线
        for route_order, tgt_idx in enumerate(route):
            tgt = next((h for h in modified_hotspots if h['id'] == tgt_idx), None)
            if tgt is None: continue
            
            target_m = [tgt['center_km'][0] * 1000.0, tgt['center_km'][1] * 1000.0]
            continued_search = (
                route_order == 0
                and k < len(active_target_ids)
                and active_target_ids[k] == tgt_idx
                and k < len(progress_kms)
                and progress_kms[k] > 0
            )

            if continued_search:
                ref_path = ref_paths[k] if k < len(ref_paths) else np.empty((0, 2))
                baseline_route = baseline_routes[k] if k < len(baseline_routes) else []
                if len(ref_path) >= 2 and tgt_idx in baseline_route:
                    baseline_order = baseline_route.index(tgt_idx)
                    if k < len(route_boundaries_list) and baseline_order < len(route_boundaries_list[k]):
                        trigger_dist_m = emergency.get('trigger_time_frac', 0.5) * polyline_length_m(ref_path)
                        target_done_dist_m = route_boundaries_list[k][baseline_order]
                        remaining_active_path = slice_polyline_by_distance(
                            ref_path, trigger_dist_m, target_done_dist_m
                        )
                        if len(remaining_active_path) >= 2:
                            remaining_active_path[0] = np.array(curr_pose[:2], dtype=float)
                            full_path_m.extend(remaining_active_path[1:].tolist())
                            tail = remaining_active_path[-1]
                            prev = remaining_active_path[-2]
                            curr_pose = [
                                tail[0], tail[1],
                                math.atan2(tail[1] - prev[1], tail[0] - prev[0])
                            ]
                            continue

            if etype == 'S3' and not continued_search:
                curr_km = (curr_pose[0] / 1000.0, curr_pose[1] / 1000.0)
                detour_waypoints = compute_detour_waypoints_km(
                    curr_km, tgt['center_km'],
                    emergency['no_fly_center'][0],
                    emergency['no_fly_center'][1],
                    emergency['no_fly_radius'],
                    min_turn_radius_km=spiral_cfg['uav']['Rmin'] / 1000.0,
                )
                if detour_waypoints:
                    detour_points_m = [
                        [wp[0] * 1000.0, wp[1] * 1000.0]
                        for wp in detour_waypoints
                    ]
                    full_path_m.extend(detour_points_m)
                    last_pt = detour_points_m[-1]
                    yaw = math.atan2(target_m[1] - last_pt[1], target_m[0] - last_pt[0])
                    curr_pose = [last_pt[0], last_pt[1], yaw]

            yaw_to_tgt = math.atan2(target_m[1] - curr_pose[1], target_m[0] - curr_pose[0])
            spiral_sol = spiral_search_arc_exact(prob_grid, tgt['center_km'], tgt['radius_km'], X_km, Y_km, yaw_to_tgt, spiral_cfg)
            if continued_search:
                s_full = trim_polyline_prefix(spiral_sol['path'], progress_kms[k] * 1000.0)
            else:
                s_full = spiral_sol['path']
            if len(s_full) < 2:
                continue

            s_path = s_full[::10]
            if np.hypot(s_path[-1][0] - s_full[-1][0], s_path[-1][1] - s_full[-1][1]) > 1e-6:
                s_path = np.vstack([s_path, s_full[-1]])
            s_start_yaw = math.atan2(s_path[1][1] - s_path[0][1], s_path[1][0] - s_path[0][0])

            if not continued_search or np.hypot(s_path[0][0] - curr_pose[0], s_path[0][1] - curr_pose[1]) > 50.0:
                dubins_goal = [s_path[0][0], s_path[0][1], s_start_yaw]
                d_path, _ = dubins_curve(curr_pose, dubins_goal, r=spiral_cfg['uav']['Rmin'], stepsize=200.0)
                if d_path is not None: full_path_m.extend(d_path[:, 0:2].tolist())
            full_path_m.extend(s_path.tolist())
            
            curr_pose = [s_path[-1][0], s_path[-1][1], math.atan2(s_path[-1][1] - s_path[-2][1], s_path[-1][0] - s_path[-2][0])]
            
        actual_dqn_trajectories.append(generate_aero_traj(full_path_m, UAV_V_M_S))

    # ---- 8. 顶级科研级绘图展示 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11), facecolor='white')
    uav_colors = ['#e63946', '#1d3557', '#2a9d8f', '#9b5de5']
    abandoned_set = set(abandoned)
    original_pos = {h['id']: h['center_km'] for h in UNIFIED_HOTSPOTS}

    # 共用描边特效：黑色字，外绕白边
    text_outline = [pe.withStroke(linewidth=3, foreground='white')]

    for ax, is_baseline in [(ax1, True), (ax2, False)]:
        # 绘制半透明热力图底色
        ax.imshow(prob_grid, extent=(0, 100, 0, 100), origin='lower', cmap='plasma', alpha=0.35)

        for tgt in (UNIFIED_HOTSPOTS if is_baseline else modified_hotspots):
            cx, cy = tgt['center_km']
            is_remaining = tgt['id'] in remaining_ids
            is_abandoned = tgt['id'] in abandoned_set
            
            if is_abandoned and not is_baseline:
                color, alpha = 'black', 0.25
            elif is_remaining and not is_baseline:
                color, alpha = '#e63946', 0.75
            else:
                color, alpha = '#457b9d', 0.6
                
            ax.add_patch(Circle((cx, cy), tgt['radius_km'], color=color, fill=True, alpha=0.15))
            ax.scatter(cx, cy, s=70, color=color, edgecolors='black', linewidth=1.0, alpha=alpha, zorder=5)

            # S1 原位置与位移方向
            if not is_baseline and etype == 'S1' and tgt['id'] in original_pos:
                ox, oy = original_pos[tgt['id']]
                if abs(cx - ox) > 0.5 or abs(cy - oy) > 0.5:
                    ax.add_patch(Circle((ox, oy), tgt['radius_km'], fill=False,
                                        color='#6c757d', alpha=0.85,
                                        linestyle='--', linewidth=1.6, zorder=6))
                    ax.scatter(ox, oy, s=55, facecolors='white', edgecolors='#6c757d',
                               linewidth=1.5, marker='o', zorder=16)
                    ax.annotate(
                        '', xy=(cx, cy), xytext=(ox, oy),
                        arrowprops=dict(
                            arrowstyle='-|>', color='#b45f06', lw=2.2,
                            alpha=0.95, linestyle='--', mutation_scale=14,
                        ),
                        zorder=17,
                    )
                    ax.text(ox + 1.0, oy - 2.0, f"Old T{tgt['id']}",
                            fontsize=9, color='#5c5c5c', fontweight='bold',
                            path_effects=text_outline, zorder=18)
                    ax.text(cx + 1.0, cy - 2.2, "Shifted",
                            fontsize=9, color='#9a4d00', fontweight='bold',
                            path_effects=text_outline, zorder=18)

            # 绘制区域 ID（删去白色底框，采用描边特效）
            label = f"T{tgt['id']}" + ('(X)' if is_abandoned and not is_baseline else '')
            ax.text(cx + 1.2, cy + 1.2, label, fontsize=11, fontweight='bold', color='black', 
                    path_effects=text_outline, zorder=10)

        ax.plot(UAV_BASE_KM[0], UAV_BASE_KM[1], 'k^', markersize=14, label='Base Station', zorder=10)

        # 禁飞区绘制
        if not is_baseline and etype == 'S3':
            ax.add_patch(Circle(emergency['no_fly_center'], emergency['no_fly_radius'], color='#ff9f1c', fill=True, alpha=0.35, label='No-Fly Zone'))

        # 核心修改：修复 S4 事件无人机损坏位置未标注的问题
        # 标记突发事件断点位置 (移除对 active_uavs 的强过滤)
        if not is_baseline or (is_baseline and etype != 'none'):
            interrupt_labeled = False
            for k in range(NUM_UAVS):
                # 如果是 S4 事件中损坏的那架无人机
                if etype == 'S4' and k == emergency.get('failed_uav'):
                    # 用显眼的黑白交叉标记，并且标注 Crash
                    ax.scatter(uav_positions[k][0], uav_positions[k][1],
                               s=250, color='#333333', marker='X', edgecolors='white', linewidth=1.5, zorder=25, 
                               label='Failure Point' if k == emergency.get('failed_uav') else "")
                    ax.text(uav_positions[k][0] + 1.5, uav_positions[k][1] - 2.5, "Crash", 
                            color='#333333', fontweight='bold', fontsize=11, path_effects=text_outline, zorder=25)
                else:
                    # 其他正常打断的无人机，依然使用星号
                    lbl = 'Interrupt Point' if not interrupt_labeled else ""
                    ax.scatter(uav_positions[k][0], uav_positions[k][1],
                               s=280, color=uav_colors[k], marker='*', edgecolors='black', linewidth=1.2, zorder=20, label=lbl)
                    interrupt_labeled = True

        ax.set_xlabel('X Coordinate (km)')
        ax.set_ylabel('Y Coordinate (km)')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle=':', alpha=0.6)

    # 左图渲染：基线实际轨迹与顺序标签
    ax1.set_title(f'Original Baseline Plan (GA)\nRoutes: {baseline_routes}')
    for k, traj in enumerate(actual_baseline_trajectories):
        if len(traj) > 0:
            traj_km = traj / 1000.0
            ax1.plot(traj_km[:, 0], traj_km[:, 1], color=uav_colors[k], linestyle='-', linewidth=2.2, alpha=0.85, label=f'UAV {k+1} Traj', zorder=8)
            end_pt = traj_km[-1]
            prev_pt = traj_km[-2] if len(traj_km) > 1 else traj_km[-1]
            ax1.annotate('', xy=(end_pt[0], end_pt[1]), xytext=(prev_pt[0], prev_pt[1]), arrowprops=dict(arrowstyle='->', color=uav_colors[k], lw=2.0))
            
            for step, tgt_idx in enumerate(baseline_routes[k]):
                tgt = UNIFIED_HOTSPOTS[tgt_idx]
                ax1.text(tgt['center_km'][0] - 2.5, tgt['center_km'][1] - 3.5, f"[{step+1}]", 
                         color=uav_colors[k], fontweight='bold', fontsize=11, path_effects=text_outline, zorder=15)

    ax1.legend(loc='upper right', framealpha=0.9)

    # 右图渲染：DQN重规划轨迹与顺序标签
    ax2.set_title(f'DQN Reallocation (Event: {etype})\nRoutes: {clean_routes}')
    for k, traj in enumerate(actual_dqn_trajectories):
        if len(traj) > 0:
            traj_km = traj / 1000.0
            ax2.plot(traj_km[:, 0], traj_km[:, 1], color=uav_colors[k], linestyle='-', linewidth=2.2, alpha=0.85, zorder=8)
            end_pt = traj_km[-1]
            prev_pt = traj_km[-2] if len(traj_km) > 1 else traj_km[-1]
            ax2.annotate('', xy=(end_pt[0], end_pt[1]), xytext=(prev_pt[0], prev_pt[1]), arrowprops=dict(arrowstyle='->', color=uav_colors[k], lw=2.0))
            
            for step, tgt_idx in enumerate(clean_routes[k]):
                tgt = next((h for h in modified_hotspots if h['id'] == tgt_idx), None)
                if tgt:
                    ax2.text(tgt['center_km'][0] - 2.5, tgt['center_km'][1] - 3.5, f"[{step+1}]", 
                             color=uav_colors[k], fontweight='bold', fontsize=11, path_effects=text_outline, zorder=15)
                    
    # 右图：标出 S3 引导避障点
    if etype == 'S3':
        for pts in detour_paths:
            for pt in pts[1:-1]:
                is_tgt = any(np.hypot(pt[0]-tc[0], pt[1]-tc[1]) < 1e-3 for tc in [t['center_km'] for t in modified_hotspots])
                if not is_tgt:
                    ax2.scatter(pt[0], pt[1], s=40, color='yellow', edgecolors='black', marker='D', zorder=15)

    ax2.legend(loc='upper right', framealpha=0.9)

    # 航程文本框对比
    base_dists = compute_route_distance_km(baseline_routes, [UAV_BASE_KM]*NUM_UAVS, id_to_hotspot, UAV_BASE_KM=UAV_BASE_KM)
    dqn_dists = compute_route_distance_km(clean_routes, uav_positions, id_to_hotspot, emergency=emergency, consumed_ranges=consumed_ranges, baseline_routes=baseline_routes, completed=completed, UAV_BASE_KM=UAV_BASE_KM)
    
    ga_text = '\n'.join(f'UAV{k+1}: {base_dists[k]:.0f} km' + (' ⚠' if base_dists[k] > MAX_RANGE_KM else '') for k in range(NUM_UAVS))
    ax1.text(0.02, 0.98, f'Total Scheduled Range:\n{ga_text}', transform=ax1.transAxes, fontsize=10, fontfamily='monospace', va='top', bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

    dqn_text = '\n'.join(f'UAV{k+1}: {consumed_ranges[k]:.0f} + {dqn_dists[k]:.0f} = {consumed_ranges[k]+dqn_dists[k]:.0f} km' + (' ⚠' if (consumed_ranges[k]+dqn_dists[k]) > MAX_RANGE_KM else '') for k in range(NUM_UAVS))
    ax2.text(0.02, 0.98, f'Range (Used + Remain):\n{dqn_text}', transform=ax2.transAxes, fontsize=10, fontfamily='monospace', va='top', bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

    fig.suptitle(f'Emergency Reallocation Demo (Continuous Aerodynamic Trajectory) | MAX= {MAX_RANGE_KM:.0f} km\n'
                 f'DQN J_max: {dqn_cost["J_max"]:.0f}s | J_sum: {dqn_cost["J_sum"]:.0f}s | Violations: {dqn_cost.get("range_violations", 0)}',
                 fontsize=16, y=0.98)

    os.makedirs('outputs/eval', exist_ok=True)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output_path = f'outputs/eval/demo_phys_{etype}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Demo figure saved: {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['S1', 'S2', 'S3', 'S4', 'random', 'all'], default='random',
                        help='突发事件类型；all 会依次运行 S1/S2/S3/S4')
    parser.add_argument('--model-path', type=str, default='outputs/models/emergency_dqn_model.pt')
    parser.add_argument('--no-show', action='store_true',
                        help='只保存图片，不弹出 Matplotlib 窗口；批量运行时推荐开启')
    args = parser.parse_args()
    if args.type == 'all':
        for etype in ['S1', 'S2', 'S3', 'S4']:
            run_emergency_demo(etype, args.model_path, show=not args.no_show)
    else:
        run_emergency_demo(args.type, args.model_path, show=not args.no_show)
