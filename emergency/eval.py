"""
突发事件临机决策 — 统一评估脚本 (V2 架构)
对比 DQN(轮询多目标) / GA重优化 / 最近邻启发式 / 按权重贪心 / 僵化执行 五种方法
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math

from core import heatmap as hm
from core import ga_allocator as ga
from core.spiral_search import spiral_search_arc_exact
from core.dubins_planner import dubins_curve
from emergency.agent import EmergencyDQN, NUM_UAVS
from emergency.simulator import EmergencySimulator
from emergency.utils import (
    SPAN_KM, MAX_RANGE_KM, UAV_VELOCITY_KM_S, GLOBAL_C_DROP,
    STATE_DIM_V2, N_MAX, get_valid_actions_v2
)

# ==================== 全局配置 ====================
UAV_BASE_KM = (0.0, 0.0)
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


# ==================== 启发式方法 ====================

def nearest_neighbor_assignment(uav_positions, hotspots, active_uavs, remaining_ids):
    routes = [[] for _ in range(NUM_UAVS)]
    remaining = list(remaining_ids)
    uav_last_pos = {k: np.array(pos) for k, pos in enumerate(uav_positions) if active_uavs[k]}

    # 按权重降序处理
    remaining.sort(key=lambda tid: next((h['weight'] for h in hotspots if h['id'] == tid), 0), reverse=True)

    for tid in remaining:
        tgt = next((h for h in hotspots if h['id'] == tid), None)
        if tgt is None or not uav_last_pos: continue
        tgt_pos = np.array(tgt['center_km'])
        # 寻找距离最近的活跃无人机
        best_uav = min(uav_last_pos.keys(),
                       key=lambda k: np.hypot(uav_last_pos[k][0] - tgt_pos[0],
                                               uav_last_pos[k][1] - tgt_pos[1]))
        routes[best_uav].append(tid)
        uav_last_pos[best_uav] = tgt_pos

    return routes

def greedy_weight_assignment(uav_positions, hotspots, active_uavs, remaining_ids):
    routes = [[] for _ in range(NUM_UAVS)]
    remaining = list(remaining_ids)
    remaining.sort(key=lambda tid: next((h['weight'] for h in hotspots if h['id'] == tid), 0), reverse=True)

    for tid in remaining:
        tgt = next((h for h in hotspots if h['id'] == tid), None)
        if tgt is None: continue
        tgt_pos = np.array(tgt['center_km'])
        best_uav, best_dist = None, float('inf')
        
        for k in range(NUM_UAVS):
            if not active_uavs[k]: continue
            last_pos = uav_positions[k]
            if routes[k]:
                last_tgt = next((h for h in hotspots if h['id'] == routes[k][-1]), None)
                if last_tgt: last_pos = np.array(last_tgt['center_km'])
                
            d = np.hypot(last_pos[0] - tgt_pos[0], last_pos[1] - tgt_pos[1])
            if d < best_dist:
                best_dist, best_uav = d, k
                
        if best_uav is not None:
            routes[best_uav].append(tid)

    return routes

def rigid_baseline_assignment(baseline_routes, emergency, completed):
    # 剔除已完成目标，死板地按原计划飞行
    routes = [[tid for tid in r if tid not in completed] for r in baseline_routes]
    if emergency['type'] == 'S4':
        routes[emergency['failed_uav']] = []
    return routes


# ==================== 统一物理评估接口 ====================

def calc_unified_metrics(routes, sim, modified_hotspots, active_uavs, emergency,
                         consumed_ranges, uav_positions, baseline_routes, completed, remaining_ids):
    """确保所有方法都在相同的物理评价层（含禁飞区绕飞和进度扣减）进行核算"""
    cost = sim.compute_cost_for_assignment(
        routes, modified_hotspots, active_uavs, emergency,
        consumed_ranges, uav_positions, baseline_routes=baseline_routes, completed=completed
    )
    assigned_ids = set([tid for r in routes for tid in r])
    abandoned_ids = set(remaining_ids) - assigned_ids
    
    abandoned_weights = []
    for tid in abandoned_ids:
        w = next((h['weight'] for h in modified_hotspots if h['id'] == tid), 0)
        abandoned_weights.append(w)
        
    cost['abandoned_weights'] = abandoned_weights
    return cost


# ==================== 评估主函数 ====================

def evaluate_all_methods(num_scenarios=50, model_path='models/emergency_dqn_model.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"评估设备: {device}")
    
    # ---- 初始化仿真环境 ----
    print("初始化仿真环境...")
    x_km = np.arange(0, SPAN_KM + dx_km / 2, dx_km)
    y_km = np.arange(0, SPAN_KM + dx_km / 2, dx_km)
    X_km, Y_km = np.meshgrid(x_km, y_km)

    hm_hotspots = [{'center': h['center_km'], 'sigma': h['sigma'], 'weight': h['weight']} for h in UNIFIED_HOTSPOTS]
    np.random.seed(42)
    prob_grid = hm.generate_controlled_prob_field(X_km, Y_km, hm_hotspots, alpha=0.15)

    sim = EmergencySimulator(UNIFIED_HOTSPOTS, NUM_UAVS, UAV_BASE_KM, spiral_cfg, prob_grid, X_km, Y_km)

    # ---- 加载DQN模型 ----
    dqn = EmergencyDQN(state_dim=STATE_DIM_V2, num_actions=N_MAX).to(device)
    if os.path.exists(model_path):
        dqn.load_state_dict(torch.load(model_path, map_location=device))
        print(f"DQN模型已加载: {model_path}")
    else:
        print(f"警告: 模型文件 {model_path} 不存在，使用未训练模型")
    dqn.eval()

    # ---- 运行初始全局GA ----
    print("运行基准GA获取初始参考路径...")
    ga_targets = [{'id': h['id'], 'pos': h['center_km'], 'weight': h['weight'], 'radius_km': h['radius_km']}
                  for h in UNIFIED_HOTSPOTS]
    ga.init_ga_env(ga_targets, UAV_BASE_KM, NUM_UAVS, prob_grid=prob_grid, X_km=X_km, Y_km=Y_km,
                   spiral_cfg=spiral_cfg, weights=(0.5, 0.3, 0.4))
    best_chrom, _, _, norm_factors = ga.run_ga(pop_size=200, generations=300, patience=100)
    _, raw_routes, _ = ga.evaluate_chromosome(best_chrom, norm_factors)
    baseline_routes = raw_routes[:NUM_UAVS]

    # 生成物理边界 (为了半途锁定推演)
    ref_paths, route_boundaries_list = [], []
    for k, route in enumerate(baseline_routes):
        if not route:
            ref_paths.append(np.empty((0, 2))); route_boundaries_list.append([])
            continue
        uav_path, boundaries = [], []
        cumulative_dist = 0.0
        curr_pose = [UAV_BASE_KM[0] * 1000.0, UAV_BASE_KM[1] * 1000.0, 0.0]
        
        for tgt_idx in route:
            tgt = UNIFIED_HOTSPOTS[tgt_idx]
            yaw_to_tgt = math.atan2(tgt['center_km'][1]*1000.0 - curr_pose[1], tgt['center_km'][0]*1000.0 - curr_pose[0])
            spiral_sol = spiral_search_arc_exact(prob_grid, tgt['center_km'], tgt['radius_km'], X_km, Y_km, yaw_to_tgt, spiral_cfg)
            d_path, d_len = dubins_curve(curr_pose, [spiral_sol['path'][0][0], spiral_sol['path'][0][1], 
                                         math.atan2(spiral_sol['path'][1][1]-spiral_sol['path'][0][1], spiral_sol['path'][1][0]-spiral_sol['path'][0][0])],
                                         r=spiral_cfg['uav']['Rmin'], stepsize=200.0)
            if d_path is not None:
                uav_path.extend(d_path[:, 0:2].tolist())
                cumulative_dist += d_len
            uav_path.extend(spiral_sol['path'][::10].tolist())
            cumulative_dist += spiral_sol['totalLen']
            boundaries.append(cumulative_dist)
            curr_pose = [spiral_sol['pEnd'][0], spiral_sol['pEnd'][1], spiral_sol['yawEnd']]
            
        ref_paths.append(np.array(uav_path))
        route_boundaries_list.append(boundaries)

    # ---- 评估循环 ----
    results = {'DQN': [], 'GA_oracle': [], 'NearestNeighbor': [], 'GreedyWeight': [], 'Rigid': []}
    decision_times = {k: [] for k in results.keys()}
    scenario_types = []
    per_type_costs = {m: {et: [] for et in ['S1', 'S2', 'S3', 'S4']} for m in results.keys()}

    print(f"\n开始评估 {num_scenarios} 个突发事件场景...")
    print("-" * 70)

    for scenario_idx in range(num_scenarios):
        # 生成并应用紧急事件
        emergency = sim.generate_random_emergency(baseline_routes, ref_paths)
        etype = emergency['type']
        scenario_types.append(etype)

        uav_positions, _, completed, remaining, consumed_ranges, active_target_ids, progress_kms = \
            sim.simulate_until_emergency(ref_paths, emergency, baseline_routes, route_boundaries_list)

        if etype == 'S1': # 剥离已完成目标的位移
            emergency['shifts'] = [s for s in emergency.get('shifts', []) if s['id'] not in completed]
            emergency['affected_targets'] = [tid for tid in emergency.get('affected_targets', []) if tid not in completed]

        modified_hotspots, active_uavs, _ = sim.apply_emergency(emergency)
        
        if etype == 'S2': # 新增目标计入剩余
            for nt in emergency.get('new_targets', []): remaining.add(nt['id'])

        remaining_ids = set(remaining)

        # ---------------- 1. DQN 方法 ----------------
        t0 = time.time()
        state = sim.reset_v2(uav_positions, consumed_ranges, modified_hotspots,
                             active_uavs=active_uavs, completed=completed,
                             active_target_ids=active_target_ids, progress_kms=progress_kms,
                             emergency=emergency)
        done = False
        decisions = []
        
        while not done:
            valid_mask = get_valid_actions_v2(sim.current_uav_idx, sim.uav_states, sim.targets,
                                              locked_target_idx=sim.locked_target_idxs[sim.current_uav_idx],
                                              emergency=emergency)
            if not valid_mask.any(): break
                
            with torch.no_grad():
                q_values = dqn(torch.FloatTensor(state).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
            
            action = int(np.argmax(np.where(valid_mask, q_values, -np.inf)))
            t_selected = sim.targets[action]
            decisions.append((t_selected, sim.current_uav_idx))
            state, _, done, _ = sim.step_v2(action)

        # 重建精确路由
        dqn_routes = [[] for _ in range(NUM_UAVS)]
        for tgt, uav_idx in decisions:
            dqn_routes[uav_idx].append(tgt['id'])

        dqn_cost = calc_unified_metrics(dqn_routes, sim, modified_hotspots, active_uavs, emergency,
                                        consumed_ranges, uav_positions, baseline_routes, completed, remaining_ids)
        decision_times['DQN'].append(time.time() - t0)
                
        # ---------------- 2. GA Oracle 方法 ----------------
        t0 = time.time()
        n_active = sum(active_uavs)
        ga_targets_mod = [{'id': h['id'], 'pos': h['center_km'], 'weight': h['weight'], 'radius_km': h['radius_km']}
                          for h in modified_hotspots if h['id'] in remaining_ids]
                          
        if ga_targets_mod and n_active > 0:
            for tgt in ga_targets_mod:
                matching = [h for h in modified_hotspots if h['id'] == tgt['id']]
                tgt['spiral_time'] = matching[0].get('spiral_time', 500.0) if matching else 500.0

            # 扣除各机已消耗的时间约束
            full_d_max_s = MAX_RANGE_KM / UAV_VELOCITY_KM_S
            per_uav_d_max = [full_d_max_s - consumed_ranges[k] / UAV_VELOCITY_KM_S for k in range(NUM_UAVS) if active_uavs[k]]
            ga.init_ga_env(ga_targets_mod, UAV_BASE_KM, n_active, weights=(0.5, 0.3, 0.4),
                           d_max=min(per_uav_d_max) if per_uav_d_max else full_d_max_s, c_drop=GLOBAL_C_DROP)
            try:
                best_c, _, _, _ = ga.run_ga(pop_size=80, generations=100, patience=30)
                _, _, _, ga_raw_routes, _, _, _ = ga.calculate_raw_costs(best_c)
                
                # 将 GA 局部索引还原为全局物理 ID
                ga_routes = [[] for _ in range(NUM_UAVS)]
                active_uav_indices = [i for i, a in enumerate(active_uavs) if a]
                for i, r in enumerate(ga_raw_routes[:n_active]):
                    real_uav_idx = active_uav_indices[i]
                    ga_routes[real_uav_idx] = [ga_targets_mod[tgt_idx]['id'] for tgt_idx in r]
                    
                ga_cost = calc_unified_metrics(ga_routes, sim, modified_hotspots, active_uavs, emergency,
                                               consumed_ranges, uav_positions, baseline_routes, completed, remaining_ids)
            except Exception:
                ga_cost = {'J_max': 1e9, 'J_sum': 1e9, 'weighted_arrival_sum': 1e9, 'abandoned_weights': []}
        else:
            ga_cost = {'J_max': 0, 'J_sum': 0, 'weighted_arrival_sum': 0, 'abandoned_weights': []}
        decision_times['GA_oracle'].append(time.time() - t0)

        # ---------------- 3. 最近邻 (NN) ----------------
        t0 = time.time()
        nn_routes = nearest_neighbor_assignment(uav_positions, modified_hotspots, active_uavs, remaining_ids)
        nn_cost = calc_unified_metrics(nn_routes, sim, modified_hotspots, active_uavs, emergency,
                                       consumed_ranges, uav_positions, baseline_routes, completed, remaining_ids)
        decision_times['NearestNeighbor'].append(time.time() - t0)

        # ---------------- 4. 权重贪心 (GW) ----------------
        t0 = time.time()
        gw_routes = greedy_weight_assignment(uav_positions, modified_hotspots, active_uavs, remaining_ids)
        gw_cost = calc_unified_metrics(gw_routes, sim, modified_hotspots, active_uavs, emergency,
                                       consumed_ranges, uav_positions, baseline_routes, completed, remaining_ids)
        decision_times['GreedyWeight'].append(time.time() - t0)

        # ---------------- 5. 僵化执行基线 (Rigid) ----------------
        t0 = time.time()
        rigid_routes = rigid_baseline_assignment(baseline_routes, emergency, completed)
        rigid_cost = calc_unified_metrics(rigid_routes, sim, modified_hotspots, active_uavs, emergency,
                                          consumed_ranges, uav_positions, baseline_routes, completed, remaining_ids)
        decision_times['Rigid'].append(time.time() - t0)

        # ---- 收集并记录数据 ----
        results['DQN'].append(dqn_cost)
        results['GA_oracle'].append(ga_cost)
        results['NearestNeighbor'].append(nn_cost)
        results['GreedyWeight'].append(gw_cost)
        results['Rigid'].append(rigid_cost)

        for method, cost in zip(results.keys(), [dqn_cost, ga_cost, nn_cost, gw_cost, rigid_cost]):
            per_type_costs[method][etype].append(cost)

        if (scenario_idx + 1) % 10 == 0:
            print(f"  已评估 {scenario_idx + 1}/{num_scenarios} 场景...")

    # ==================== 结果汇总及可视化生成 ====================
    print("\n" + "=" * 70)
    print("评估结果汇总")
    print("=" * 70)

    metrics_names = ['J_max', 'J_sum', 'weighted_arrival_sum']
    method_names = ['DQN', 'GA_oracle', 'NearestNeighbor', 'GreedyWeight', 'Rigid']

    header = f"{'Method':<18} {'J_max':>12} {'J_sum':>12} {'W_Arrival':>12} {'Fatal%':>8} {'Time(ms)':>10} {'vsGA':>8}"
    print(header)
    print("-" * 70)

    for method in method_names:
        costs = results.get(method, [])
        if not costs: continue
        avg_times = np.mean(decision_times.get(method, [0.001])) * 1000.0

        jmax_vals = [c['J_max'] for c in costs if c['J_max'] < 1e8]
        jsum_vals = [c['J_sum'] for c in costs if c['J_sum'] < 1e8]
        warr_vals = [c.get('weighted_arrival_sum', 0) for c in costs if c.get('weighted_arrival_sum', 0) < 1e8]
        fatal_vals = [1 if c.get('range_violations', 0) > 0 else 0 for c in costs]

        fatal_rate = np.mean(fatal_vals) * 100.0 if fatal_vals else 0.0
        ga_jmax = np.mean([c['J_max'] for c in results.get('GA_oracle', []) if c['J_max'] < 1e8] or [1.0])
        vs_ga = np.mean(jmax_vals) / ga_jmax if ga_jmax > 0 else 0.0

        print(f"{method:<18} {np.mean(jmax_vals):>12.1f} {np.mean(jsum_vals):>12.1f} {np.mean(warr_vals):>12.1f} {fatal_rate:>7.1f}% {avg_times:>10.1f} {vs_ga:>8.2f}x")

    print("-" * 70)
    print(f"Fatal% = 任何UAV超 {MAX_RANGE_KM} km航程的场景占比")

    # ---- 绘图设置 ----
    TOL_COLORS = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#AA3377']
    METHOD_COLORS = dict(zip(method_names, TOL_COLORS))
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'axes.titlesize': 13})
    os.makedirs('outputs', exist_ok=True)

    # 1. 对比箱线图
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax_idx, metric in enumerate(metrics_names):
        data, labels = [], []
        for method in method_names:
            vals = [c[metric] for c in results[method] if c.get(metric, 0) < 1e8]
            if vals: data.append(vals); labels.append(method)
        bp = axes[ax_idx].boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
        for patch, method in zip(bp['boxes'], labels):
            patch.set_facecolor(METHOD_COLORS.get(method, '#888'))
            patch.set_alpha(0.75)
        axes[ax_idx].set_title(metric.replace('_', ' ').title(), fontweight='bold')
        axes[ax_idx].tick_params(axis='x', rotation=20)
        axes[ax_idx].grid(True, linestyle=':', alpha=0.4, axis='y')
    plt.tight_layout()
    plt.savefig('outputs/emergency_eval_comparison.png', dpi=200)
    plt.close()

    # 2. 放弃目标权重分布直方图
    fig, ax = plt.subplots(figsize=(10, 5))
    for method in method_names:
        abandoned_weights = []
        for c in results[method]:
            if 'abandoned_weights' in c: abandoned_weights.extend(c['abandoned_weights'])
        if abandoned_weights:
            ax.hist(abandoned_weights, bins=12, alpha=0.5, label=method, color=METHOD_COLORS.get(method, '#888'))
    ax.set_xlabel('Target Weight', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Abandoned Target Weight Distribution Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.4, axis='y')
    plt.tight_layout()
    plt.savefig('outputs/emergency_eval_abandon.png', dpi=200)
    plt.close()
    
    print("\n✅ 所有学术图表已保存至 outputs/ 目录。")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios', type=int, default=50)
    parser.add_argument('--model-path', type=str, default='models/emergency_dqn_model.pt')
    args = parser.parse_args()
    evaluate_all_methods(num_scenarios=args.scenarios, model_path=args.model_path)