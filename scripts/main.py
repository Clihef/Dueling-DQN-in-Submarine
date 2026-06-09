import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter
import os 


# ================= 导入核心模块 =================
from core import heatmap as hm
from core import ga_allocator as ga
from core.spiral_search import spiral_search_arc_exact
from core.dubins_planner import dubins_curve
from core.uav_navi import uav_navi_traverse

# ================= 1. 统一的全局配置与目标定义 =================
UAV_BASE_KM = (0.0, 0.0)
NUM_UAVS = 4
UAV_V_M_S = 50.0  # 飞行速度 50m/s
UAV_VELOCITY_KM_S = 0.05
MAX_RANGE_KM = 350.0       # 每架UAV最大航程 (km)

dx_km = 0.1          # 分辨率 0.1 km (100 m)
span_km = 100.0      # 范围 0~100 km

# 遗传算法的偏好权重 (W_MAX_TIME, W_TOTAL_TIME, W_URGENCY)
GA_PREFERENCE_WEIGHTS = (0.1, 1.0, 0.0) # 极端偏好总时间
# GA_PREFERENCE_WEIGHTS = (0.0, 0.1, 1.0) # 极端偏好紧急程度，
# GA_PREFERENCE_WEIGHTS = (1.0, 0.1, 0.0) # 极端偏好协同度
# GA_PREFERENCE_WEIGHTS = (0.5, 0.3, 0.4) # 综合权重，平衡总时间和紧急程度


# 统一的热点定义库 (同时满足 Heatmap 的 sigma 和 GA 的 area 需求)
# 热点
UNIFIED_HOTSPOTS = [
    # 簇1：近距离区 (距离起点近，但权重低，测试无人机会不会"路过顺手搜")
    {'id': 0, 'center_km': (15, 15), 'sigma': (3, 3), 'weight': 0.25, 'radius_km': 2.5},
    {'id': 1, 'center_km': (25, 10), 'sigma': (4, 4), 'weight': 0.30, 'radius_km': 3.0},
    {'id': 2, 'center_km': (10, 30), 'sigma': (2, 2), 'weight': 0.40, 'radius_km': 2.0},
    
    # 簇2：中距离左侧 (包含一个极高危热点)
    {'id': 3, 'center_km': (20, 60), 'sigma': (4, 4), 'weight': 0.50, 'radius_km': 3.5},
    {'id': 4, 'center_km': (30, 80), 'sigma': (6, 6), 'weight': 1.00, 'radius_km': 5.0}, # 高危！
    {'id': 5, 'center_km': (15, 85), 'sigma': (3, 3), 'weight': 0.35, 'radius_km': 2.5},

    # 簇3：中距离右侧 (权重中等，适合测试负载均衡)
    {'id': 6, 'center_km': (60, 20), 'sigma': (5, 5), 'weight': 0.60, 'radius_km': 4.0},
    {'id': 7, 'center_km': (55, 40), 'sigma': (4, 4), 'weight': 0.55, 'radius_km': 3.0},
    {'id': 8, 'center_km': (75, 30), 'sigma': (6, 6), 'weight': 0.85, 'radius_km': 4.5}, # 次高危！

    # 簇4：远距离右上角 (距离最远，包含一个极高危热点，测试无人机会不会不顾一切先飞过去)
    {'id': 9,  'center_km': (85, 85), 'sigma': (5, 5), 'weight': 0.95, 'radius_km': 4.0}, # 高危！
    {'id': 10, 'center_km': (75, 90), 'sigma': (3, 3), 'weight': 0.20, 'radius_km': 2.5},
    {'id': 11, 'center_km': (90, 70), 'sigma': (4, 4), 'weight': 0.30, 'radius_km': 3.5},

    # 簇5：散落的孤立点 (用于填补地图空白，增加航线规划难度)
    {'id': 12, 'center_km': (50, 60), 'sigma': (3, 3), 'weight': 0.45, 'radius_km': 2.5},
    {'id': 13, 'center_km': (40, 20), 'sigma': (2, 2), 'weight': 0.20, 'radius_km': 2.0},
    {'id': 14, 'center_km': (65, 65), 'sigma': (4, 4), 'weight': 0.50, 'radius_km': 3.0},
]

# 螺旋搜索相关配置 (透传给 spiral_search_generator)
spiral_cfg = {
    'uav': {'Vg': UAV_V_M_S, 'H': 100.0, 'Rmin': 2000.0},
    'sensor': {'dcty': 600.0, 'hsub': 20.0, 'overlapFrac': 0.1, 'width_model': 'sqrt'},
    'target': {'vTgt': 5.0, 'epsR': 200.0, 'tau0': 0.0},
    'opt': {'Dmax': 150000.0, 'modeHard': True, 'gamma': 0.01, 'deltaP_stop': 0.005, 'lambdaR0': 0.03},
    'grid': {'r0interval': 10, 'hinterval': 20, 'NGrid': list(range(1, 6))},
    'traj': {'turnDir': 1}
}


def save_dynamic_trajectory_animation(prob_grid, actual_trajectories, hotspots, base_km, span_km,
                                      weights, output_path, interval_ms=60):
    """保存多机轨迹的动态播放 GIF。"""
    valid_trajs = [traj for traj in actual_trajectories if len(traj) > 0]
    if not valid_trajs:
        print(">>> 无可用轨迹，跳过动态展示图生成。")
        return

    max_steps = max(len(traj) for traj in valid_trajs)
    if max_steps < 2:
        print(">>> 轨迹点不足，跳过动态展示图生成。")
        return

    # 控制总帧数，避免 GIF 体积过大
    max_frames = 700
    step_stride = max(1, int(np.ceil(max_steps / max_frames)))
    frame_steps = list(range(1, max_steps + 1, step_stride))
    if frame_steps[-1] != max_steps:
        frame_steps.append(max_steps)

    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    img = ax.imshow(prob_grid, extent=(0, span_km, 0, span_km), origin='lower', cmap='plasma', alpha=0.45)
    fig.colorbar(img, ax=ax, label='Target Probability Density')

    # 背景元素：热点区域与基地
    for tgt in hotspots:
        cx, cy = tgt['center_km']
        circle = Circle((cx, cy), tgt['radius_km'], color='red', fill=True, alpha=0.16)
        ax.add_patch(circle)
        ax.scatter(cx, cy, s=85, color='red', edgecolors='black', linewidth=1.2, alpha=0.85)

    ax.plot(base_km[0], base_km[1], 'k^', markersize=14, label='Base Station')

    uav_colors = ['red', 'blue', 'green', 'magenta']
    lines = []
    markers = []
    for k in range(len(actual_trajectories)):
        color = uav_colors[k % len(uav_colors)]
        line, = ax.plot([], [], linestyle='-', linewidth=2, color=color, label=f'UAV {k+1}')
        marker, = ax.plot([], [], marker='o', markersize=6, color=color)
        lines.append(line)
        markers.append(marker)

    progress_text = ax.text(
        0.02,
        0.98,
        'Progress: 0.0%',
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.88, edgecolor='black', boxstyle='round,pad=0.3')
    )

    ax.set_title(f'Multi-UAV Dynamic Trajectory (Weights: {weights})')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_xlim(0, span_km)
    ax.set_ylim(0, span_km)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right')

    def init():
        for line, marker in zip(lines, markers):
            line.set_data([], [])
            marker.set_data([], [])
        progress_text.set_text('Progress: 0.0%')
        return lines + markers + [progress_text]

    def update(step):
        for k, traj in enumerate(actual_trajectories):
            if len(traj) == 0:
                continue

            end_idx = min(step, len(traj))
            part = traj[:end_idx] / 1000.0
            lines[k].set_data(part[:, 0], part[:, 1])
            markers[k].set_data([part[-1, 0]], [part[-1, 1]])

        progress = step / max_steps * 100.0
        progress_text.set_text(f'Progress: {progress:.1f}%')
        return lines + markers + [progress_text]

    ani = FuncAnimation(
        fig,
        update,
        frames=frame_steps,
        init_func=init,
        interval=max(1, int(interval_ms)),
        blit=False,
        repeat=False
    )

    fps = max(5, int(round(1000.0 / max(1, int(interval_ms)))))
    writer = PillowWriter(fps=fps)
    ani.save(output_path, writer=writer)
    plt.close(fig)
    print(f">>> 动态轨迹图已保存: {output_path}")


def show_dynamic_trajectory_animation(prob_grid, actual_trajectories, hotspots, base_km, span_km,
                                      weights, interval_ms=60):
    """窗口实时播放多机轨迹，不落盘。"""
    valid_trajs = [traj for traj in actual_trajectories if len(traj) > 0]
    if not valid_trajs:
        print(">>> 无可用轨迹，跳过实时播放。")
        return

    max_steps = max(len(traj) for traj in valid_trajs)
    if max_steps < 2:
        print(">>> 轨迹点不足，跳过实时播放。")
        return

    # 实时播放同样做抽帧，避免窗口刷新过慢
    max_frames = 700
    step_stride = max(1, int(np.ceil(max_steps / max_frames)))
    frame_steps = list(range(1, max_steps + 1, step_stride))
    if frame_steps[-1] != max_steps:
        frame_steps.append(max_steps)

    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    img = ax.imshow(prob_grid, extent=(0, span_km, 0, span_km), origin='lower', cmap='plasma', alpha=0.45)
    fig.colorbar(img, ax=ax, label='Target Probability Density')

    for tgt in hotspots:
        cx, cy = tgt['center_km']
        circle = Circle((cx, cy), tgt['radius_km'], color='red', fill=True, alpha=0.16)
        ax.add_patch(circle)
        ax.scatter(cx, cy, s=85, color='red', edgecolors='black', linewidth=1.2, alpha=0.85)

    ax.plot(base_km[0], base_km[1], 'k^', markersize=14, label='Base Station')

    uav_colors = ['red', 'blue', 'green', 'magenta']
    lines = []
    markers = []
    for k in range(len(actual_trajectories)):
        color = uav_colors[k % len(uav_colors)]
        line, = ax.plot([], [], linestyle='-', linewidth=2, color=color, label=f'UAV {k+1}')
        marker, = ax.plot([], [], marker='o', markersize=6, color=color)
        lines.append(line)
        markers.append(marker)

    progress_text = ax.text(
        0.02,
        0.98,
        'Progress: 0.0%',
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.88, edgecolor='black', boxstyle='round,pad=0.3')
    )

    ax.set_title(f'Multi-UAV Live Trajectory (Weights: {weights})')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_xlim(0, span_km)
    ax.set_ylim(0, span_km)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right')

    def init():
        for line, marker in zip(lines, markers):
            line.set_data([], [])
            marker.set_data([], [])
        progress_text.set_text('Progress: 0.0%')
        return lines + markers + [progress_text]

    def update(step):
        for k, traj in enumerate(actual_trajectories):
            if len(traj) == 0:
                continue

            end_idx = min(step, len(traj))
            part = traj[:end_idx] / 1000.0
            lines[k].set_data(part[:, 0], part[:, 1])
            markers[k].set_data([part[-1, 0]], [part[-1, 1]])

        progress = step / max_steps * 100.0
        progress_text.set_text(f'Progress: {progress:.1f}%')
        return lines + markers + [progress_text]

    ani = FuncAnimation(
        fig,
        update,
        frames=frame_steps,
        init_func=init,
        interval=max(1, int(interval_ms)),
        blit=False,
        repeat=False
    )

    # 保持局部变量引用直到窗口关闭，避免动画对象被提前回收
    _ = ani
    plt.show()
    plt.close(fig)
    print(">>> 实时轨迹播放结束。")

def run_simulation(weights, output_dir='figure', enable_dynamic_plot=False,
                   enable_live_plot=False, animation_interval_ms=60,
                   save_trajectory_data=False):
    # 设置权重
    GA_PREFERENCE_WEIGHTS = weights
    
    print(f">>> 开始运行仿真，权重: {weights}")
    
    # 1. 生成底图环境
    x_km = np.arange(0, span_km + dx_km/2, dx_km)
    y_km = np.arange(0, span_km + dx_km/2, dx_km)
    X_km, Y_km = np.meshgrid(x_km, y_km)
     
    hm_hotspots = [{'center': h['center_km'], 'sigma': h['sigma'], 'weight': h['weight']} for h in UNIFIED_HOTSPOTS]
    
    np.random.seed(42)
    prob_grid = hm.generate_controlled_prob_field(X_km, Y_km, hm_hotspots, alpha=0.15)

    # 预计算每个热点区域的概率质量，用于后续覆盖率统计
    region_prob_mass = {}
    for tgt in UNIFIED_HOTSPOTS:
        cx, cy = tgt['center_km']
        mask_region = np.hypot(X_km - cx, Y_km - cy) <= tgt['radius_km']
        region_prob_mass[tgt['id']] = float(np.sum(prob_grid[mask_region]))
    total_hotspot_prob_mass = float(sum(region_prob_mass.values()))
    
    # 保存静态热力图 (仅第一次)
    if not os.path.exists(f'{output_dir}/static_heatmap.png'):
        plt.figure(figsize=(10, 8))
        img = plt.imshow(prob_grid, extent=(0, 100, 0, 100), origin='lower', cmap='plasma', alpha=0.8)
        plt.colorbar(img, label='Target Probability Density')
        plt.title('Initial Static Heatmap')
        plt.xlabel('X (km)')
        plt.ylabel('Y (km)')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.savefig(f'{output_dir}/static_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 保存带热点标注的热力图 (仅第一次)
    if not os.path.exists(f'{output_dir}/static_heatmap_annotated.png'):
        plt.figure(figsize=(10, 8), facecolor='white')
        img = plt.imshow(prob_grid, extent=(0, 100, 0, 100), origin='lower', cmap='plasma', alpha=0.8)
        plt.colorbar(img, label='Target Probability Density')
        # 遍历所有热点，进行圈图和标注
        for tgt in UNIFIED_HOTSPOTS:
            cx, cy = tgt['center_km']
            r_km = tgt['radius_km']
            weight = tgt['weight']
            tgt_id = tgt['id']
            # 画圆圈表示热点的有效搜索半径
            circle = Circle((cx, cy), r_km, color='black', fill=False, linestyle='--', linewidth=1.5, alpha=0.8)
            plt.gca().add_patch(circle)
            # 标出热点中心
            plt.plot(cx, cy, 'k*', markersize=8)
            # 在右上方一点的位置标注 ID 和 权重，添加半透明白色底框防止与热力图颜色冲突
            plt.text(cx + 1.5, cy + 1.5, f'ID: {tgt_id}\nW: {weight:.2f}', 
                     color='black', fontsize=9, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2.0))           
        plt.title('Annotated Static Heatmap (Hotspots & Priorities)')
        plt.xlabel('X (km)')
        plt.ylabel('Y (km)')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.savefig(f'{output_dir}/static_heatmap_annotated.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 2. 运行GA
    ga_targets = [{'id': h['id'], 'pos': h['center_km'], 'weight': h['weight'], 'radius_km': h['radius_km']} for h in UNIFIED_HOTSPOTS] # GA需要的目标格式
    d_max_s = MAX_RANGE_KM / UAV_VELOCITY_KM_S  # 350km / 0.05 = 7000s # 最大飞行时间 2小时，单位秒
    ga.init_ga_env(ga_targets, UAV_BASE_KM, NUM_UAVS, prob_grid, X_km, Y_km, spiral_cfg, weights=GA_PREFERENCE_WEIGHTS, d_max=d_max_s, c_drop=18000.0) # 预计算每个区域螺旋搜索耗时
    
    best_chrom, best_cost, history, norm_factors = ga.run_ga(pop_size=200, generations=300, patience=100) 
    _, best_routes, best_times = ga.evaluate_chromosome(best_chrom, norm_factors) # 评估得到最优路径和时间
    
    print(f"最优分配: {best_routes}")
    
    # 保存GA适应度曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history, 'b-', linewidth=2)
    plt.title(f'GA Fitness Curve (Weights: {weights})')
    plt.xlabel('Generation')
    plt.ylabel('Total Cost')
    plt.grid(True, linestyle='--', alpha=0.5)
    weights_str = f"{weights[0]}_{weights[1]}_{weights[2]}"
    plt.savefig(f'{output_dir}/ga_fitness_curve_{weights_str}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 生成路径
    uav_global_reference_paths = []
    region_spiral_stats = []
    uav_plan_total_len_m = np.zeros(NUM_UAVS)
    uav_plan_spiral_len_m = np.zeros(NUM_UAVS)
    uav_plan_dubins_len_m = np.zeros(NUM_UAVS)

    # 🌟 新增：分离真实无人机航线与被放弃的目标（垃圾桶）
    real_routes = best_routes[:NUM_UAVS]
    dropped_targets = best_routes[NUM_UAVS] if len(best_routes) > NUM_UAVS else []
    if dropped_targets:
        print(f"\n⚠️ 以下目标因航程超限或代价过高被放弃 (放入垃圾桶): {dropped_targets}\n")

    # 遍历每个无人机的分配路径，生成全局参考路径
    for k, route in enumerate(real_routes):
        # 如果无人机没有分配到任何区域，直接添加一个空路径占位，继续下一架无人机的处理
        if not route:
            uav_global_reference_paths.append(np.empty((0,2)))
            continue
        uav_path = []
        curr_pose = [UAV_BASE_KM[0]*1000.0, UAV_BASE_KM[1]*1000.0, 0.0]
        # 为每个分配到的目标区域生成螺旋搜索路径，并连接成完整的参考路径
        for visit_order, tgt_idx in enumerate(route, start=1):
            tgt = UNIFIED_HOTSPOTS[tgt_idx]
            target_m = [tgt['center_km'][0]*1000.0, tgt['center_km'][1]*1000.0]
            # 计算当前无人机位置到目标中心的航向角，作为螺旋搜索的初始朝向
            yaw_to_tgt = math.atan2(target_m[1] - curr_pose[1], target_m[0] - curr_pose[0])
            # 螺旋搜索路径生成
            spiral_sol = spiral_search_arc_exact(prob_grid, tgt['center_km'], tgt['radius_km'], X_km, Y_km, yaw_to_tgt, spiral_cfg)
            s_path = spiral_sol['path'][::10] # 稀疏取点，减少后续处理压力

            # 连接当前无人机位置到螺旋搜索起点的Dubins路径
            s_start_pos = spiral_sol['path'][0]
            s_start_yaw = math.atan2(spiral_sol['path'][1][1] - spiral_sol['path'][0][1], 
                                     spiral_sol['path'][1][0] - spiral_sol['path'][0][0])
            dubins_goal = [s_start_pos[0], s_start_pos[1], s_start_yaw]
            d_path, d_len = dubins_curve(curr_pose, dubins_goal, r=spiral_cfg['uav']['Rmin'], stepsize=200.0)

            # 将Dubins路径和螺旋搜索路径连接起来，形成完整的参考路径
            if d_path is not None:
                uav_path.extend(d_path[:, 0:2].tolist())
                uav_plan_total_len_m[k] += float(d_len)
                uav_plan_dubins_len_m[k] += float(d_len)
            uav_path.extend(s_path.tolist())
            uav_plan_total_len_m[k] += float(spiral_sol['totalLen'])
            uav_plan_spiral_len_m[k] += float(spiral_sol['totalLen'])

            pdet = float(spiral_sol.get('Pdet', 0.0))
            tgt_prob_mass = region_prob_mass.get(tgt['id'], 0.0)
            covered_prob_mass_est = pdet * tgt_prob_mass
            region_spiral_stats.append({
                'uav': k + 1,
                'visit_order': visit_order,
                'tgt_id': tgt['id'],
                'center_km': tgt['center_km'],
                'radius_km': float(tgt['radius_km']),
                'N': int(spiral_sol['N']),
                'r0_km': float(spiral_sol['r0'] / 1000.0),
                'h_km': float(spiral_sol['h'] / 1000.0),
                'spiral_len_km': float(spiral_sol['totalLen'] / 1000.0),
                'spiral_time_s': float(spiral_sol['totalTime']),
                'Pdet': pdet,
                'region_prob_mass': float(tgt_prob_mass),
                'covered_prob_mass_est': float(covered_prob_mass_est)
            })

            # 更新当前无人机位置为螺旋搜索结束位置
            curr_pose = [spiral_sol['pEnd'][0], spiral_sol['pEnd'][1], spiral_sol['yawEnd']]

        # 将当前无人机的完整参考路径添加到全局列表中    
        uav_global_reference_paths.append(np.array(uav_path))
    
    # 保存单个区域路径 (仅第一次，以ID=4为例)
    if not os.path.exists(f'{output_dir}/single_region_path.png'):
        tgt_example = UNIFIED_HOTSPOTS[4]
        yaw_example = 0.0
        spiral_example = spiral_search_arc_exact(prob_grid, tgt_example['center_km'], tgt_example['radius_km'], X_km, Y_km, yaw_example, spiral_cfg)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(prob_grid, extent=(0, 100, 0, 100), origin='lower', cmap='plasma', alpha=0.5)
        plt.colorbar(label='Target Probability Density')
        path_km = spiral_example['path'] / 1000.0
        plt.plot(path_km[:, 0], path_km[:, 1], 'r-', linewidth=2, label='Spiral Search Path')
        circle = Circle(tgt_example['center_km'], tgt_example['radius_km'], color='blue', fill=False, linestyle='--', linewidth=2, alpha=0.7, label='Search Area')
        plt.gca().add_patch(circle)
        plt.plot(tgt_example['center_km'][0], tgt_example['center_km'][1], 'ko', markersize=10, label='Hotspot Center')

        # 标出螺旋算法的切入点并编号 1,2,...
        cut_ins_km = spiral_example['cutIns'] / 1000.0
        if len(cut_ins_km) > 0:
            plt.scatter(cut_ins_km[:, 0], cut_ins_km[:, 1], s=48, c='white', edgecolors='black',
                        linewidths=0.9, zorder=5, label='Cut-in Points')
            for idx, (px, py) in enumerate(cut_ins_km, start=1):
                plt.text(px + 0.15, py + 0.15, f'{idx}', fontsize=9, color='black', fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1.5), zorder=6)

        # 标注优化参数 N、r0、h（r0/h 转换为 km）
        param_text = (
            f'N = {spiral_example["N"]}\n'
            f'r0 = {spiral_example["r0"] / 1000.0:.2f} km\n'
            f'h = {spiral_example["h"] / 1000.0:.2f} km'
        )
        plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes,
                 va='top', ha='left', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.88, edgecolor='black', boxstyle='round,pad=0.35'))

        plt.title(f'Single Region Search Path (Hotspot ID {tgt_example["id"]})')
        plt.xlabel('X (km)')
        plt.ylabel('Y (km)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        cx, cy = tgt_example['center_km']
        r_km = tgt_example['radius_km']
        margin = 2.0  # 在圆圈外留出 2km 的视觉边距
        ax = plt.gca()
        # 强制设置视口界限（必须放在图表元素都添加完毕后）
        ax.set_xlim(cx - r_km - margin, cx + r_km + margin)
        ax.set_ylim(cy - r_km - margin, cy + r_km + margin)        
        # 保持真实的物理长宽比 1:1，但限定在框内，不扩展坐标系
        ax.set_aspect('equal', adjustable='box')
        plt.savefig(f'{output_dir}/single_region_path.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. 执行导航仿真，根据动力学模型，生成实际轨迹
    actual_trajectories = []
    uav_actual_time_s = np.zeros(NUM_UAVS)
    uav_actual_dist_m = np.zeros(NUM_UAVS)
    # 遍历每个无人机的全局参考路径，执行导航仿真，得到实际飞行轨迹
    for k, ref_path in enumerate(uav_global_reference_paths):
        # 如果无人机没有分配到任何区域，ref_path会是一个空数组，此时直接添加一个空轨迹占位，继续下一架无人机的处理
        if len(ref_path) == 0:
            actual_trajectories.append(np.empty((0,2)))
            continue
        clean_path = [ref_path[0]]
        # 进行稀疏化处理，删去过于密集的点，减少计算压力
        for pt in ref_path[1:]:
            if np.hypot(pt[0] - clean_path[-1][0], pt[1] - clean_path[-1][1]) > 5.0:
                clean_path.append(pt)
        ref_path_clean = np.array(clean_path)
        # 总距离计算；总时间乘以一个系数模拟实飞中的绕飞和调整
        total_dist = np.sum(np.hypot(np.diff(ref_path_clean[:, 0]), np.diff(ref_path_clean[:, 1])))
        total_sim_time = (total_dist / UAV_V_M_S) * 1.2 
        uav_actual_dist_m[k] = float(total_dist)
        uav_actual_time_s[k] = float(total_sim_time)
        
        actual_traj = uav_navi_traverse(ref_path_clean, total_sim_time, 0.02, UAV_V_M_S, 70.0)
        actual_trajectories.append(actual_traj)
    
    # 5. 绘制最终图
    plt.figure(figsize=(12, 10), facecolor='white')
    img = plt.imshow(prob_grid, extent=(0, 100, 0, 100), origin='lower', cmap='plasma', alpha=0.5)
    plt.colorbar(img, label='Target Probability Density')
    
    uav_colors = ['red', 'blue', 'green', 'magenta']
    
    for k in range(NUM_UAVS):
        ref_path = uav_global_reference_paths[k]
        act_traj = actual_trajectories[k]
        if len(ref_path) > 0:
            plt.plot(ref_path[:, 0]/1000.0, ref_path[:, 1]/1000.0, 
                     color=uav_colors[k], linestyle=':', linewidth=1, alpha=0.6)
            plt.plot(act_traj[:, 0]/1000.0, act_traj[:, 1]/1000.0, 
                     color=uav_colors[k], linestyle='-', linewidth=2, label=f'UAV {k+1}')
            
    for tgt in UNIFIED_HOTSPOTS:
        cx, cy = tgt['center_km']
        # weight_color = cm.Reds(tgt['weight'])
        plt.scatter(cx, cy, s=100, color='red', edgecolors='black', linewidth=1.5, alpha=0.8)
        plt.text(cx + 1, cy + 1, f'ID:{tgt["id"]}', fontsize=10, fontweight='bold', color='black', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        circle = Circle((cx, cy), tgt['radius_km'], color='red', fill=True, alpha=0.2)
        plt.gca().add_patch(circle)
        
    plt.plot(UAV_BASE_KM[0], UAV_BASE_KM[1], 'k^', markersize=15, label='Base Station')
    
    for k, route in enumerate(real_routes):
        if not route:
            continue
        prev_pos = UAV_BASE_KM
        for i, tgt_idx in enumerate(route):
            tgt = UNIFIED_HOTSPOTS[tgt_idx]
            curr_pos = tgt['center_km']
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            dist = np.hypot(dx, dy)
            if dist > 0:
                dx /= dist
                dy /= dist
            plt.arrow(prev_pos[0], prev_pos[1], dx*2, dy*2, head_width=1, head_length=1, 
                      fc=uav_colors[k], ec=uav_colors[k], alpha=0.8, linewidth=2)
            plt.text((prev_pos[0] + curr_pos[0])/2, (prev_pos[1] + curr_pos[1])/2, 
                     f'{i+1}', fontsize=12, color=uav_colors[k], fontweight='bold')
            prev_pos = curr_pos
    
    plt.title(f'Multi-UAV Simulation (Weights: {weights})', fontsize=14)
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    weights_str = f"{weights[0]}_{weights[1]}_{weights[2]}"
    plt.savefig(f'{output_dir}/final_result_{weights_str}.png', dpi=300, bbox_inches='tight')
    plt.close()

    if enable_live_plot:
        show_dynamic_trajectory_animation(
            prob_grid=prob_grid,
            actual_trajectories=actual_trajectories,
            hotspots=UNIFIED_HOTSPOTS,
            base_km=UAV_BASE_KM,
            span_km=span_km,
            weights=weights,
            interval_ms=animation_interval_ms,
        )

    if enable_dynamic_plot:
        dynamic_plot_path = f'{output_dir}/dynamic_trajectory_{weights_str}.gif'
        save_dynamic_trajectory_animation(
            prob_grid=prob_grid,
            actual_trajectories=actual_trajectories,
            hotspots=UNIFIED_HOTSPOTS,
            base_km=UAV_BASE_KM,
            span_km=span_km,
            weights=weights,
            output_path=dynamic_plot_path,
            interval_ms=animation_interval_ms,
        )

    # 6. 保存文本统计报告
    uav_plan_time_s = np.zeros(NUM_UAVS)
    best_times_arr = np.array(best_times, dtype=float).flatten()
    if best_times_arr.size > 0:
        copy_n = min(NUM_UAVS, best_times_arr.size)
        uav_plan_time_s[:copy_n] = best_times_arr[:copy_n]

    def calc_jain_fairness(values):
        positive = np.array([v for v in values if v > 1e-12], dtype=float)
        if positive.size == 0:
            return 0.0
        return float((np.sum(positive) ** 2) / (positive.size * np.sum(positive ** 2)))

    def calc_cv(values):
        positive = np.array([v for v in values if v > 1e-12], dtype=float)
        if positive.size == 0:
            return 0.0
        mean_v = float(np.mean(positive))
        if mean_v < 1e-12:
            return 0.0
        return float(np.std(positive) / mean_v)

    max_covered_prob_by_region = {}
    for row in region_spiral_stats:
        tid = row['tgt_id']
        max_covered_prob_by_region[tid] = max(max_covered_prob_by_region.get(tid, 0.0), row['covered_prob_mass_est'])

    covered_prob_mass_est = float(sum(max_covered_prob_by_region.values()))
    coverage_ratio_hotspot_est = covered_prob_mass_est / max(total_hotspot_prob_mass, np.finfo(float).eps)

    mission_time_plan_s = float(np.max(uav_plan_time_s)) if np.any(uav_plan_time_s > 0) else 0.0
    mission_time_actual_s = float(np.max(uav_actual_time_s)) if np.any(uav_actual_time_s > 0) else 0.0
    # 时间公平性计算，越接近0越不公平，越接近1越公平
    plan_time_fairness = calc_jain_fairness(uav_plan_time_s)
    actual_time_fairness = calc_jain_fairness(uav_actual_time_s)
    # CV计算，越小表示时间分布越集中，越大表示时间分布越分散
    plan_time_cv = calc_cv(uav_plan_time_s)
    actual_time_cv = calc_cv(uav_actual_time_s)

    report_path = f'{output_dir}/run_summary_{weights_str}.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('=== Multi-UAV Run Summary ===\n')
        f.write(f'weights = {weights}\n')
        f.write(f'output_dir = {output_dir}\n\n')

        f.write('[Key Settings]\n')
        f.write(f'NUM_UAVS = {NUM_UAVS}\n')
        f.write(f'UAV_BASE_KM = {UAV_BASE_KM}\n')
        f.write(f'UAV_V_M_S = {UAV_V_M_S:.3f}\n')
        f.write(f'dx_km = {dx_km:.3f}, span_km = {span_km:.3f}\n')
        f.write(f'spiral_cfg = {spiral_cfg}\n')
        f.write(f'ga_best_cost = {ga.format_objective(best_cost)}\n')
        f.write(f'ga_norm_factors = {norm_factors}\n')
        f.write(f'ga_real_routes = {real_routes}\n')
        f.write(f'ga_dropped_targets = {dropped_targets}\n\n')  # 🌟 明确记录被放弃的目标

        f.write('[Global Coverage & Collaboration]\n')
        f.write(f'hotspot_prob_mass_total = {total_hotspot_prob_mass:.8f}\n')
        f.write(f'covered_prob_mass_est = {covered_prob_mass_est:.8f}\n')
        f.write(f'hotspot_coverage_ratio_est = {coverage_ratio_hotspot_est:.4%}\n')
        f.write(f'mission_time_plan_s = {mission_time_plan_s:.3f}\n')
        f.write(f'mission_time_actual_s = {mission_time_actual_s:.3f}\n')
        f.write(f'plan_time_jain_fairness = {plan_time_fairness:.6f}\n')
        f.write(f'actual_time_jain_fairness = {actual_time_fairness:.6f}\n')
        f.write(f'plan_time_cv = {plan_time_cv:.6f}\n')
        f.write(f'actual_time_cv = {actual_time_cv:.6f}\n\n')

        f.write('[UAV Statistics]\n')
        for k in range(NUM_UAVS):
            # 🌟 替换为 real_routes
            route = real_routes[k] if k < len(real_routes) else []
            f.write(
                f'UAV {k+1}: route={route}, plan_time_s={uav_plan_time_s[k]:.3f}, '
                f'actual_time_s={uav_actual_time_s[k]:.3f}, plan_len_km={uav_plan_total_len_m[k]/1000.0:.3f}, '
                f'spiral_len_km={uav_plan_spiral_len_m[k]/1000.0:.3f}, dubins_len_km={uav_plan_dubins_len_m[k]/1000.0:.3f}, '
                f'actual_dist_km={uav_actual_dist_m[k]/1000.0:.3f}\n'
            )
        f.write('\n')

        f.write('[Per-Region Spiral Optimization]\n')
        f.write('columns: uav, visit_order, hotspot_id, center_km, radius_km, N, r0_km, h_km, spiral_len_km, spiral_time_s, Pdet, region_prob_mass, covered_prob_mass_est\n')
        for row in region_spiral_stats:
            f.write(
                f"{row['uav']}, {row['visit_order']}, {row['tgt_id']}, {row['center_km']}, {row['radius_km']:.3f}, "
                f"{row['N']}, {row['r0_km']:.3f}, {row['h_km']:.3f}, {row['spiral_len_km']:.3f}, "
                f"{row['spiral_time_s']:.3f}, {row['Pdet']:.6f}, {row['region_prob_mass']:.8f}, {row['covered_prob_mass_est']:.8f}\n"
            )
    
    if save_trajectory_data:
        # 可选保存轨迹数据用于离线回放或二次分析
        import json
        trajectory_data = {
            'uavs': [],
            'hotspots': [{'center_km': h['center_km'], 'radius_km': h['radius_km']}
                         for h in UNIFIED_HOTSPOTS],
            'base': list(UAV_BASE_KM),
            'span_km': span_km,
            'dropped_targets': dropped_targets  # 🌟 增加垃圾桶字段供前端读取
        }

        for k in range(NUM_UAVS):
            trajectory_data['uavs'].append({
                'id': k + 1,
                'trajectory': actual_trajectories[k].tolist(),
                'route': real_routes[k]  # 🌟 替换为 real_routes
            })

        traj_json_path = f'{output_dir}/trajectory_data.json'
        with open(traj_json_path, 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, indent=2)
        print(f">>> 轨迹数据已保存: {traj_json_path}")
    else:
        print(">>> 已跳过 trajectory_data.json 保存。")

    print(f">>> 仿真完成，结果保存至 {output_dir}")

def main():
    import os
    import datetime
    import argparse

    parser = argparse.ArgumentParser(description='Multi-UAV simulation runner')
    parser.add_argument(
        '--dynamic-plot',
        action='store_true',
        help='生成并保存动态轨迹 GIF（默认关闭）'
    )
    parser.add_argument(
        '--live-plot',
        action='store_true',
        help='窗口实时播放动态轨迹（不保存 GIF）'
    )
    parser.add_argument(
        '--animation-interval-ms',
        type=int,
        default=60,
        help='动态轨迹图帧间隔（毫秒），默认 60ms'
    )
    parser.add_argument(
        '--save-trajectory-data',
        action='store_true',
        help='保存 trajectory_data.json（默认关闭）'
    )
    parser.add_argument(
        '--emergency-mode',
        choices=['none', 'train', 'eval', 'demo'],
        default='none',
        help='突发事件临机决策模式: train(训练DQN), eval(评估对比), demo(可视化演示)'
    )
    parser.add_argument(
        '--emergency-type',
        choices=['S1', 'S2', 'S3', 'S4', 'random', 'all'],
        default='random',
        help='突发事件类型 (demo模式, 默认random)'
    )
    parser.add_argument(
        '--emergency-episodes',
        type=int,
        default=10000,
        help='训练回合数 (train模式, 默认10000)'
    )
    parser.add_argument(
        '--emergency-eval-scenarios',
        type=int,
        default=50,
        help='评估场景数 (eval模式, 默认50)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='outputs/models/emergency_dqn_model.pt',
        help='DQN模型路径 (默认 outputs/models/emergency_dqn_model.pt)'
    )
    parser.add_argument(
        '--emergency-eval-scenarios-per-type',
        type=int,
        default=None,
        help='每类突发事件评估场景数；优先级高于 --emergency-eval-scenarios'
    )
    parser.add_argument(
        '--emergency-eval-seed',
        type=int,
        default=42,
        help='eval模式随机种子 (默认42)'
    )
    parser.add_argument(
        '--emergency-eval-output-dir',
        type=str,
        default='outputs/eval',
        help='eval模式输出目录 (默认 outputs/eval)'
    )
    parser.add_argument(
        '--emergency-eval-ga-pop',
        type=int,
        default=160,
        help='eval模式GA_Replan种群规模 (默认160)'
    )
    parser.add_argument(
        '--emergency-eval-ga-generations',
        type=int,
        default=250,
        help='eval模式GA_Replan迭代代数 (默认250)'
    )
    parser.add_argument(
        '--emergency-eval-ga-patience',
        type=int,
        default=60,
        help='eval模式GA_Replan早停耐心 (默认60)'
    )
    parser.add_argument(
        '--emergency-eval-no-plots',
        action='store_true',
        help='eval模式只输出CSV/JSON，不生成图表'
    )
    parser.add_argument(
        '--demo-no-show',
        action='store_true',
        help='demo模式只保存图片，不弹出Matplotlib窗口；批量all时推荐开启'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default='auto',
        help='恢复训练: auto(自动最新), none(从头), 或指定ckpt路径'
    )
    args = parser.parse_args()

    # ---- 突发事件模式 ----
    if args.emergency_mode == 'train':
        from scripts.train import train as emergency_train
        emergency_train(
            num_episodes=args.emergency_episodes,
            model_save_path=args.model_path,
            resume_path=args.resume,
        )
        return

    if args.emergency_mode == 'eval':
        from scripts.eval import evaluate_all_methods
        eval_args = argparse.Namespace(
            model_path=args.model_path,
            scenarios_per_type=(
                args.emergency_eval_scenarios_per_type
                if args.emergency_eval_scenarios_per_type is not None
                else args.emergency_eval_scenarios
            ),
            seed=args.emergency_eval_seed,
            output_dir=args.emergency_eval_output_dir,
            ga_pop=args.emergency_eval_ga_pop,
            ga_generations=args.emergency_eval_ga_generations,
            ga_patience=args.emergency_eval_ga_patience,
            no_plots=args.emergency_eval_no_plots,
        )
        evaluate_all_methods(eval_args)
        return

    if args.emergency_mode == 'demo':
        from scripts.demo import run_emergency_demo
        if args.emergency_type == 'all':
            for etype in ['S1', 'S2', 'S3', 'S4']:
                run_emergency_demo(
                    emergency_type=etype,
                    model_path=args.model_path,
                    show=not args.demo_no_show,
                )
        else:
            run_emergency_demo(
                emergency_type=args.emergency_type,
                model_path=args.model_path,
                show=not args.demo_no_show,
            )
        return

    # ---- 原有仿真流程 ----
    # 获取当前运行的时间戳，格式为: 20260327_164830
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建子文件夹
    weight_sets = [
        (0.5, 0.3, 0.4),
        (0.0, 0.1, 1.0),
        (0.1, 1.0, 0.0),
        (1.0, 0.1, 0.0),
        # (0.5, 0.5, 0.0),
        # (0.3, 0.5, 0.6)
    ]

    for weights in weight_sets:
        output_dir = f'outputs/figures/run_{timestamp}/weights_{weights[0]}_{weights[1]}_{weights[2]}'
        os.makedirs(output_dir, exist_ok=True)
        run_simulation(
            weights,
            output_dir,
            enable_dynamic_plot=args.dynamic_plot,
            enable_live_plot=args.live_plot,
            animation_interval_ms=args.animation_interval_ms,
            save_trajectory_data=args.save_trajectory_data,
        )

if __name__ == "__main__":
    main()
