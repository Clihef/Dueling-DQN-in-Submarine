"""
突发事件临机决策 — 可视化演示 (5动作逐目标决策)
展示原始计划 vs DQN重分配计划的对比
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import copy
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from core import heatmap as hm
from core import ga_allocator as ga
from core.spiral_search import spiral_search_arc_exact
from core.dubins_planner import dubins_curve
from core.uav_navi import uav_navi_traverse
from emergency.agent import (
    EmergencyDQN, get_valid_actions, select_action, NUM_UAVS
)
from emergency.simulator import EmergencySimulator
from emergency.utils import (
    build_state_vector, get_affected_targets,
    apply_decision,
    replan_around_no_fly, compute_route_distance_km,
    STATE_DIM, SPAN_KM, MAX_RANGE_KM, MAX_FLIGHT_TIME_S, GLOBAL_C_DROP,
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


# ==================== 演示主函数 ====================

def run_emergency_demo(emergency_type='random', model_path='models/emergency_dqn_model.pt'):
    """运行单个突发事件场景的可视化演示

    Args:
        emergency_type: 'S1'/'S2'/'S3'/'S4'/'random'
        model_path: 训练好的DQN模型路径
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"突发事件演示模式 (5动作逐目标决策)")
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
    best_chrom, best_cost, _, norm_factors = ga.run_ga(
        pop_size=200, generations=300, patience=100
    )
    _, raw_routes, baseline_times = ga.evaluate_chromosome(best_chrom, norm_factors)
    
    # 🌟 新增：截断分离
    baseline_routes = raw_routes[:NUM_UAVS]
    print(f"原始分配: {baseline_routes}")

    # ---- 4. 生成参考路径 ----
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
            yaw_to_tgt = math.atan2(target_m[1] - curr_pose[1],
                                    target_m[0] - curr_pose[0])
            spiral_sol = spiral_search_arc_exact(
                prob_grid, tgt['center_km'], tgt['radius_km'],
                X_km, Y_km, yaw_to_tgt, spiral_cfg
            )
            s_path = spiral_sol['path'][::10]
            s_start_pos = spiral_sol['path'][0]
            s_start_yaw = math.atan2(
                spiral_sol['path'][1][1] - spiral_sol['path'][0][1],
                spiral_sol['path'][1][0] - spiral_sol['path'][0][0]
            )
            dubins_goal = [s_start_pos[0], s_start_pos[1], s_start_yaw]
            d_path, d_len = dubins_curve(
                curr_pose, dubins_goal,
                r=spiral_cfg['uav']['Rmin'], stepsize=200.0
            )
            if d_path is not None:
                uav_path.extend(d_path[:, 0:2].tolist())
                cumulative_dist += d_len
            uav_path.extend(s_path.tolist())
            cumulative_dist += spiral_sol['totalLen']
            boundaries.append(cumulative_dist)
            curr_pose = [spiral_sol['pEnd'][0], spiral_sol['pEnd'][1],
                         spiral_sol['yawEnd']]
        ref_paths.append(np.array(uav_path))
        route_boundaries_list.append(boundaries)

    # ---- 5. 生成突发事件 ----
    if emergency_type == 'random':
        emergency = sim.generate_random_emergency(baseline_routes, ref_paths)
    elif emergency_type == 'S1':
        emergency = sim._generate_s1()
    elif emergency_type == 'S2':
        emergency = sim._generate_s2()
    elif emergency_type == 'S3':
        emergency = sim._generate_s3(baseline_routes)
    elif emergency_type == 'S4':
        emergency = sim._generate_s4(baseline_routes)

    etype = emergency['type']
    print(f"突发事件: {etype}, 严重度: {emergency.get('severity', 'N/A')}")

    # ---- 6. 推算触发位置 ----
    uav_positions, uav_headings, completed, remaining, consumed_ranges, active_target_ids, progress_kms = \
        sim.simulate_until_emergency(ref_paths, emergency,
                                     baseline_routes, route_boundaries_list)

    # S1: 已搜索目标不再位移
    if etype == 'S1':
        emergency['shifts'] = [s for s in emergency.get('shifts', [])
                               if s['id'] not in completed]
        emergency['affected_targets'] = [tid for tid in emergency.get('affected_targets', [])
                                          if tid not in completed]

    # ---- 7. 应用突发事件 ----
    modified_hotspots, active_uavs, _ = sim.apply_emergency(emergency)
    remaining_ids = set(remaining)
    if etype == 'S2':
        for nt in emergency.get('new_targets', []):
            remaining_ids.add(nt['id'])

    print(f"活跃UAV: {active_uavs}, 剩余目标: {remaining_ids}")

    # ---- 8. DQN逐目标决策 ----
    affected = get_affected_targets(emergency, baseline_routes, modified_hotspots)
    affected = [t for t in affected if t['id'] not in completed]
    print(f"受影响目标 (按权重降序, 排除已完成): {[(t['id'], t['weight']) for t in affected]}")

    dqn_routes = [[tid for tid in r if tid not in completed] for r in baseline_routes]
    if etype == 'S4':
        dqn_routes[emergency['failed_uav']] = []

    abandoned = []
    decisions = []  # 记录每步决策 (用于可视化标注)

    id_to_hotspot = {h['id']: h for h in modified_hotspots}
    for h in UNIFIED_HOTSPOTS:
        if h['id'] not in id_to_hotspot:
            id_to_hotspot[h['id']] = h

    # ---- V2 MDP 决策循环 ----
    state = sim.reset_v2(uav_positions, consumed_ranges, modified_hotspots,
                         active_uavs=active_uavs, completed=completed,
                         active_target_ids=active_target_ids, progress_kms=progress_kms,
                         emergency=emergency)
    done = False
    while not done:
        valid_mask = get_valid_actions_v2(sim.current_uav_idx, sim.uav_states, sim.targets,
                                          locked_target_idx=sim.locked_target_idxs[sim.current_uav_idx],
                                          emergency=emergency)
        if not valid_mask.any():
            break
        with torch.no_grad():
            q_values = dqn(torch.FloatTensor(state).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
        action = int(np.argmax(np.where(valid_mask, q_values, -np.inf)))
        t = sim.targets[action]
        print(f"  UAV{sim.current_uav_idx+1} -> target{t['center_km']} (w={t['weight']:.2f})  Q={q_values[action]:.3f}")
        decisions.append((t, sim.current_uav_idx))
        state, _, done, _ = sim.step_v2(action)

    # 重建 clean_routes（targets 现在有 id 字段）
    clean_routes = [[] for _ in range(NUM_UAVS)]
    for t, uav_idx in decisions:
        clean_routes[uav_idx].append(t['id'])
    abandoned = []

    dqn_cost = sim.compute_cost_for_assignment(
        clean_routes, modified_hotspots, active_uavs, emergency,
        consumed_ranges, uav_positions=uav_positions, baseline_routes=baseline_routes, completed=completed
)
    print(f"DQN分配 (重排序后): {clean_routes}")
    print(f"放弃目标: {abandoned}")
    print(f"DQN代价: J_max={dqn_cost['J_max']:.1f}, J_sum={dqn_cost['J_sum']:.1f}, "
          f"越限={dqn_cost.get('range_violations', 0)}, "
          f"最大航程比={dqn_cost.get('max_range_ratio', 0):.2f}")

    # ---- 9. 可视化 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), facecolor='white')

    uav_colors = ['red', 'blue', 'green', 'magenta']
    abandoned_set = set(abandoned)

    # 构建原始位置映射 (用于S1位移对比)
    original_pos = {h['id']: h['center_km'] for h in UNIFIED_HOTSPOTS}

    for ax, title_prefix in [(ax1, 'Original Plan'), (ax2, 'DQN Reallocation')]:
        ax.imshow(prob_grid, extent=(0, 100, 0, 100), origin='lower',
                  cmap='plasma', alpha=0.35)

        for i, tgt in enumerate(modified_hotspots):
            cx, cy = tgt['center_km']
            is_remaining = tgt['id'] in remaining_ids
            is_abandoned = tgt['id'] in abandoned_set
            if is_abandoned and ax == ax2:
                color = 'black'
                alpha = 0.3
            elif is_remaining:
                color = 'red'
                alpha = 0.7
            else:
                color = 'gray'
                alpha = 0.3
            circle = Circle((cx, cy), tgt['radius_km'], color=color,
                            fill=True, alpha=0.12)
            ax.add_patch(circle)
            ax.scatter(cx, cy, s=60, color=color, edgecolors='black',
                       linewidth=0.8, alpha=alpha)

            # S1位移标注：虚线画原位置 + 位移箭头 (左右图均显示)
            if etype == 'S1' and tgt['id'] in original_pos:
                ox, oy = original_pos[tgt['id']]
                if abs(cx - ox) > 0.5 or abs(cy - oy) > 0.5:  # 有实质位移
                    old_circle = Circle((ox, oy), tgt['radius_km'],
                                        fill=False, color='blue', alpha=0.5,
                                        linestyle='--', linewidth=1.2)
                    ax.add_patch(old_circle)
                    ax.annotate('', xy=(cx, cy), xytext=(ox, oy),
                                arrowprops=dict(arrowstyle='->', color='blue',
                                                lw=1.2, alpha=0.6, linestyle='dashed'))

            if is_remaining or is_abandoned:
                label = f'{tgt["id"]}'
                if is_abandoned and ax == ax2:
                    label += '(X)'
                ax.text(cx + 0.8, cy + 0.8, label, fontsize=7,
                        fontweight='bold', color='black',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none',
                                  pad=1.5))

        ax.plot(UAV_BASE_KM[0], UAV_BASE_KM[1], 'k^', markersize=12,
                label='Base Station')

        if etype == 'S3':
            nf_c = emergency['no_fly_center']
            nf_r = emergency['no_fly_radius']
            nofly = Circle(nf_c, nf_r, color='orange', fill=True, alpha=0.3,
                           label='No-Fly Zone')
            ax.add_patch(nofly)

        for k in range(NUM_UAVS):
            if active_uavs[k]:
                ax.scatter(uav_positions[k][0], uav_positions[k][1],
                           s=100, color=uav_colors[k], marker='s',
                           edgecolors='black', linewidth=1.5, zorder=10)
                ax.text(uav_positions[k][0] + 1, uav_positions[k][1] + 1,
                        f'UAV{k + 1}', fontsize=8, fontweight='bold',
                        color=uav_colors[k])

        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

    # 原始计划 (左图)
    ax1.set_title(f'Original GA Plan\nRoutes: {baseline_routes}', fontsize=11)
    for k, route in enumerate(baseline_routes):
        if not route:
            continue
        prev = UAV_BASE_KM
        for tgt_idx in route:
            tgt = UNIFIED_HOTSPOTS[tgt_idx]
            curr = tgt['center_km']
            ax1.annotate('', xy=curr, xytext=prev,
                         arrowprops=dict(arrowstyle='->', color=uav_colors[k],
                                         lw=1.5, alpha=0.6, linestyle='dashed'))
            prev = curr

    # DQN重分配 (右图) — 显示逐目标决策结果
    decision_desc = ', '.join(
        f"({t['center_km'][0]:.0f},{t['center_km'][1]:.0f})->U{a+1}"
        for t, a in decisions
    )
    ax2.set_title(f'DQN Reallocation ({etype})\nRoutes: {clean_routes}\n'
                  f'Decisions: {decision_desc}', fontsize=9)

    # S3绕行路径绘制
    if etype == 'S3':
        detour_paths = replan_around_no_fly(
            clean_routes, uav_positions,
            emergency['no_fly_center'], emergency['no_fly_radius'],
            id_to_hotspot
        )
        for k, pts in enumerate(detour_paths):
            if len(pts) < 2:
                continue
            # 绘制含绕行点的完整路径
            for i in range(len(pts) - 1):
                ax2.plot([pts[i][0], pts[i + 1][0]],
                         [pts[i][1], pts[i + 1][1]],
                         color=uav_colors[k], lw=2.0, alpha=0.9)
                ax2.annotate('', xy=pts[i + 1], xytext=pts[i],
                             arrowprops=dict(arrowstyle='->', color=uav_colors[k],
                                             lw=2.0, alpha=0.9))
            # 标注绕行点（非UAV位置、非目标位置的中间点）
            for pt in pts[1:-1]:
                ax2.scatter(pt[0], pt[1], s=50, color='yellow',
                            edgecolors='black', linewidth=0.5, zorder=15,
                            marker='D')
    else:
        for k, route in enumerate(clean_routes):
            if not route:
                continue
            prev_uav = uav_positions[k]
            for tgt_idx in route:
                tgt = next((h for h in modified_hotspots if h['id'] == tgt_idx), None)
                if tgt is None:
                    continue
                curr = tgt['center_km']
                ax2.annotate('', xy=curr, xytext=(prev_uav[0], prev_uav[1]),
                             arrowprops=dict(arrowstyle='->', color=uav_colors[k],
                                             lw=2.0, alpha=0.9))
                prev_uav = curr

    # ---- 计算各UAV实际总航程 ----
    base_dists = compute_route_distance_km(baseline_routes, [UAV_BASE_KM] * NUM_UAVS, id_to_hotspot, UAV_BASE_KM=UAV_BASE_KM)

    # 🌟 修复：补齐 emergency, consumed_ranges, baseline_routes, completed，激活半途进度扣减逻辑
    dqn_dists = compute_route_distance_km(
        clean_routes, uav_positions, id_to_hotspot,
        emergency=emergency, consumed_ranges=consumed_ranges,
        baseline_routes=baseline_routes, completed=completed, UAV_BASE_KM=UAV_BASE_KM
    )
    dqn_total = [consumed_ranges[k] + dqn_dists[k] for k in range(NUM_UAVS)]

    # 左图：GA 基线总航程
    ga_text = '\n'.join(
        f'UAV{k+1}: {base_dists[k]:.0f} km' + (' !' if base_dists[k] > MAX_RANGE_KM else '')
        for k in range(NUM_UAVS)
    )
    ax1.text(0.02, 0.97, f'Total Range:\n{ga_text}',
             transform=ax1.transAxes, fontsize=7, fontfamily='monospace',
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', pad=4))

    # 右图：DQN 总航程 (已消耗 + 剩余)
    dqn_text = '\n'.join(
        f'UAV{k+1}: {consumed_ranges[k]:.0f}+{dqn_dists[k]:.0f}={dqn_total[k]:.0f} km'
        + (' !' if dqn_total[k] > MAX_RANGE_KM else '')
        for k in range(NUM_UAVS)
    )
    ax2.text(0.02, 0.97, f'Range (used+remain):\n{dqn_text}',
             transform=ax2.transAxes, fontsize=7, fontfamily='monospace',
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', pad=4))

    # 总标题
    fig.suptitle(
        f'Emergency Decision-Making Demo (5-Action DQN) | Type: {etype} | '
        f'MAX={MAX_RANGE_KM:.0f}km | '
        f'DQN J_max: {dqn_cost["J_max"]:.0f}s | J_sum: {dqn_cost["J_sum"]:.0f}s | '
        f'Abandoned: {abandoned}\n'
        f'Violations: {dqn_cost.get("range_violations", 0)} | '
        f'Max range ratio: {dqn_cost.get("max_range_ratio", 0):.2f}',
        fontsize=11, fontweight='bold'
    )

    # 从CSV日志读取训练回合数，用于文件命名
    csv_path = model_path.replace('.pt', '_log.csv')
    ep_count = 0
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            lines = f.readlines()
            if len(lines) > 1:
                ep_count = int(lines[-1].split(',')[0])

    os.makedirs('outputs', exist_ok=True)
    output_path = f'outputs/emergency_demo_{etype}_ep{ep_count}.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"演示图已保存: {output_path}")
    plt.show()

    print("演示完成")


# ==================== CLI入口 ====================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DQN突发事件临机决策演示 (5动作)')
    parser.add_argument('--type', choices=['S1', 'S2', 'S3', 'S4', 'random'],
                        default='random', help='突发事件类型')
    parser.add_argument('--model-path', type=str, default='models/emergency_dqn_model.pt')
    args = parser.parse_args()

    run_emergency_demo(
        emergency_type=args.type,
        model_path=args.model_path,
    )
