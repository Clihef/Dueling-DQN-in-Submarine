import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
import math

# 导入底图和螺旋生成模块以进行预计算
from core import heatmap as hm
from core.spiral_search import spiral_search_arc_exact

# ================= 仿真参数设置 =================
UAV_VELOCITY = 0.05         # 无人机速度 50 m/s = 0.05 km/s
SENSOR_WIDTH = 1.0          # 传感器扫描宽度 (km)

# 权重配置 (调节任务偏好)
W_MAX_TIME = 0.1           # 侧重木桶效应，要求协同完成任务
W_TOTAL_TIME = 0.0          # 侧重总能耗
W_URGENCY = 1.0           # 侧重尽早到达高概率(高权重)热点

# 环境初始化与精确代价预计算
def init_ga_env(targets_list, uav_base_km, num_uavs, prob_grid=None, X_km=None, Y_km=None, spiral_cfg=None, weights=(0.5, 0.5, 0.0), d_max=None, c_drop=None):
    # 定义全局变量，供 GA 内部函数调用
    global TARGETS, NUM_TARGETS, UAV_BASE, NUM_UAVS, W_MAX_TIME, W_TOTAL_TIME, W_URGENCY
    global GA_REAL_UAVS, GA_D_MAX, GA_C_DROP, GA_USE_TRASH
    UAV_BASE = uav_base_km
    GA_REAL_UAVS = num_uavs          # 真实无人机数（不含垃圾桶）
    GA_D_MAX = d_max                  # 航程硬约束 (秒)，None=无约束
    GA_C_DROP = c_drop                # 放弃惩罚系数，None=不启用垃圾桶
    GA_USE_TRASH = (c_drop is not None)
    NUM_UAVS = num_uavs + 1 if GA_USE_TRASH else num_uavs   # 启用垃圾桶时+1
    TARGETS = targets_list
    NUM_TARGETS = len(TARGETS)
    W_MAX_TIME, W_TOTAL_TIME, W_URGENCY = weights

    # 预计算每个热点的精确搜索时间和覆盖率
    if prob_grid is not None and spiral_cfg is not None:
        print("\n>>> [GA Init] 正在为各热点预计算精确螺旋搜索代价...")
        for tgt in TARGETS:
            if 'spiral_time' in tgt:
                continue  # 已有预计算值，跳过
            sol = spiral_search_arc_exact(
                prob_grid=prob_grid,
                center_km=tgt['pos'],
                radius_km=tgt['radius_km'],
                X_km=X_km, Y_km=Y_km,
                yaw_at_center=0.0,
                cfg=spiral_cfg
            )
            tgt['spiral_dist'] = sol['totalLen']
            tgt['spiral_time'] = sol['totalTime']
            tgt['pdet'] = sol['Pdet']
    elif not all('spiral_time' in tgt for tgt in TARGETS):
        print("\n>>> [GA Init] 警告：未传入网格环境参数且目标无预计算螺旋代价，将回退使用粗略面积估算。")

# ================= 染色体解码与物理计算 =================
def calculate_raw_costs(chrom):
    """
    纯物理计算层：解码 + 航程硬约束 + 放弃惩罚
    返回 (J_max, J_sum, weighted_arrival_sum, routes, finish_times, abandon_penalty)
    """
    routes = [[] for _ in range(NUM_UAVS)]
    uav_idx = 0
    for gene in chrom:
        if gene >= NUM_TARGETS:
            uav_idx += 1
        else:
            routes[uav_idx].append(gene)

    # 分离真实无人机与垃圾桶
    real_routes = routes[:GA_REAL_UAVS]
    trash_route = routes[GA_REAL_UAVS] if GA_USE_TRASH and len(routes) > GA_REAL_UAVS else []

    finish_times = np.zeros(NUM_UAVS)
    weighted_arrival_sum = 0.0

    # 计算每架无人机的完成时间和加权到达时间
    for k, route in enumerate(routes):
        if not route:
            continue
        curr_pos = UAV_BASE
        t = 0.0
        for tgt_idx in route:
            tgt = TARGETS[tgt_idx]
            # 修改后：直线 + Dubins 转弯惩罚预估
            linear_dist = np.hypot(curr_pos[0] - tgt['pos'][0], curr_pos[1] - tgt['pos'][1])

            # 估算转弯代价：假设平均每次转场需要进行约 90~180 度的偏航角调整
            # 半径 R = 2.0 km，半个圆周的长度大概是 0.75 * pi * R ≈ 4.7 km，飞行时间约 4.7 / 0.05 = 94s，作为转弯惩罚加入总时间
            R_MIN = 2.0 # km
            turn_penalty = math.pi * 0.75 * R_MIN 

            dist = linear_dist + turn_penalty
            t += dist / UAV_VELOCITY
            amplified_weight = 10 ** (4 * tgt['weight'])
            weighted_arrival_sum += t * amplified_weight
            if 'spiral_time' in tgt:
                t += tgt['spiral_time']
            else:
                search_dist = tgt['area'] / SENSOR_WIDTH
                t += search_dist / UAV_VELOCITY
            curr_pos = tgt['pos']
        finish_times[k] = t

    # ---- 航程超限量：累积超限秒数（有梯度，不直接淘汰） ----
    over_time = 0.0
    if GA_D_MAX is not None:
        for k, route in enumerate(real_routes):
            if finish_times[k] > GA_D_MAX:
                over_time += finish_times[k] - GA_D_MAX

    # ---- 放弃惩罚 ----
    abandon_penalty = 0.0
    if GA_USE_TRASH and trash_route:
        abandon_penalty = sum(TARGETS[t]['weight'] * GA_C_DROP for t in trash_route)

    J_max = np.max(finish_times[:GA_REAL_UAVS]) if any(finish_times[:GA_REAL_UAVS] > 0) else 0.0
    J_sum = np.sum(finish_times[:GA_REAL_UAVS])

    return J_max, J_sum, weighted_arrival_sum, routes, finish_times, abandon_penalty, over_time

# ================= 动态归一化代价评估 =================
def evaluate_chromosome(chrom, norm_factors):
    """
    代价评估层：接收动态归一化基准，计算最终的适应度代价
    """
    J_max, J_sum, w_arrival, routes, finish_times, abandon_penalty, over_time = calculate_raw_costs(chrom)

    max_J_max, max_J_sum, max_w_arrival, max_abandon, max_over_time = norm_factors

    # 归一化处理
    norm_J_max = J_max / max(max_J_max, 1e-5)
    norm_J_sum = J_sum / max(max_J_sum, 1e-5)
    norm_W_arrival = w_arrival / max(max_w_arrival, 1e-5)
    norm_abandon = abandon_penalty / max(max_J_max, 1e-5) if max_J_max > 0 else 0.0
    norm_over = over_time / max(max_over_time, 1e-5) if max_over_time > 0 else 0.0

    ABANDON_WEIGHT = 1.0
    OVER_WEIGHT = 10.0  # 超限惩罚：确保"合法但烂" >> "超限但快"

    total_cost = (W_MAX_TIME * norm_J_max + W_TOTAL_TIME * norm_J_sum
                  + W_URGENCY * norm_W_arrival + ABANDON_WEIGHT * norm_abandon
                  + OVER_WEIGHT * norm_over)

    return total_cost, routes, finish_times

# ================= 遗传算法 (GA) 操作 =================
def create_individual():
    # 生成一条随机染色体并打乱
    chrom = list(range(NUM_TARGETS + NUM_UAVS - 1)) # 包含目标区域和分隔符
    random.shuffle(chrom) # 打乱顺序
    return chrom

def crossover(p1, p2):
    """顺序交叉 (Order Crossover, OX)，专门用于保证不产生重复基因的排列"""
    size = len(p1)
    start, end = sorted(random.sample(range(size), 2)) # 随机选择交叉段
    child = [-1] * size # 初始化子代为无效值
    # 继承父代1的一段
    child[start:end+1] = p1[start:end+1]
    
    # 用父代2的剩余元素按顺序填充
    fill_elements = [item for item in p2 if item not in child]
    ptr = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = fill_elements[ptr]
            ptr += 1
    return child

def mutate(chrom, mutation_rate=0.2):
    if random.random() < mutation_rate:
        # 抛硬币决定使用哪种变异方式：50%交换，50%逆转
        if random.random() < 0.5:
            # 交换突变 (Swap)
            idx1, idx2 = random.sample(range(len(chrom)), 2)
            chrom[idx1], chrom[idx2] = chrom[idx2], chrom[idx1]
        else:
            # 逆转突变 (Inversion) -> 能有效解开航线交叉
            idx1, idx2 = sorted(random.sample(range(len(chrom)), 2))
            chrom[idx1:idx2+1] = list(reversed(chrom[idx1:idx2+1]))
    return chrom

def run_ga(pop_size=100, generations=300, patience=60):
    # 种群大小为100，迭代次数为300，如果连续60代没有改进则提前停止
    # 1. 生成第一代随机种群
    population = [create_individual() for _ in range(pop_size)]

    # ------------------------------------------------------------------
    # [GA] 动态归一化基准扫描 (Generation 0 预处理)
    # 过滤航程超限个体，仅用有效个体建立归一化基准
    # ------------------------------------------------------------------
    raw_results = [calculate_raw_costs(chrom) for chrom in population]

    max_J_max = max(res[0] for res in raw_results)
    max_J_sum = max(res[1] for res in raw_results)
    max_w_arrival = max(res[2] for res in raw_results)
    max_abandon = max(res[5] for res in raw_results) if any(r[5] > 0 for r in raw_results) else 1.0
    max_over_time = max(res[6] for res in raw_results) if any(r[6] > 0 for r in raw_results) else 1.0

    norm_factors = (max_J_max, max_J_sum, max_w_arrival, max_abandon, max_over_time)
    print(f"[GA] 动态归一化基准: Max_J_max={max_J_max:.1f}, Max_J_sum={max_J_sum:.1f}, "
          f"Max_W_arrival={max_w_arrival:.2e}, Max_Abandon={max_abandon:.1f}, "
          f"Max_OverTime={max_over_time:.1f}s")
    # ------------------------------------------------------------------
        
    best_chrom = None
    best_cost = float('inf') # 初始化最优解为无穷大
    cost_history = []
    no_improve_count = 0      
    # 基础变异率
    base_mut_rate = 0.2

    for gen in range(generations):
        # 评估适应度 (代价的倒数)
        scored_pop = [] # 存储 (代价, 染色体) 的列表，便于排序和选择

        # 2. 评估每个染色体的代价，并记录最优解
        for chrom in population:
            cost, _, _ = evaluate_chromosome(chrom, norm_factors)
            scored_pop.append((cost, chrom))
            
            if cost < best_cost:
                best_cost = cost
                best_chrom = chrom
                no_improve_count = 0
                
        cost_history.append(best_cost) # 记录每代的最优代价，便于后续绘制收敛曲线
        no_improve_count += 1

        # 3. 如果连续 15 代没有进步，开始逐渐提高变异率，刺激种群跳出局部最优
        if no_improve_count > 15:
            current_mut_rate = min(0.6, base_mut_rate + 0.05 * (no_improve_count // 10)) # 每10代增加5%，最高不超过60%
        else:
            current_mut_rate = base_mut_rate

        # 4. 选择与淘汰
        scored_pop.sort(key=lambda x: x[0])
        next_gen = [x[1] for x in scored_pop[:4]] # 保留代价最低的前4个个体直接进入下一代
        
        # 5. 交叉与变异生成下一代，直至达到种群大小
        while len(next_gen) < pop_size:
            # 随机抽5个，选最好的作为父母
            p1 = min(random.sample(scored_pop, 5), key=lambda x: x[0])[1]
            p2 = min(random.sample(scored_pop, 5), key=lambda x: x[0])[1]
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate=current_mut_rate)
            next_gen.append(child)

        # 6. 更新种群    
        population = next_gen
        
        # 每50代打印一次当前最优解的代价，观察收敛情况
        if gen % 50 == 0:
            print(f"Generation {gen}, Best Cost: {best_cost:.2f}, Mutation Rate: {current_mut_rate:.2f}")

        if no_improve_count >= patience:
            print(f"Early stopping at generation {gen} due to no improvement.")
            break
            
    return best_chrom, best_cost, cost_history, norm_factors

# ================= 可视化 =================
def plot_results(routes, finish_times, cost_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- 图 1：航迹分配图 ---
    colors = ['r', 'b', 'g', 'm']
    ax1.plot(UAV_BASE[0], UAV_BASE[1], 'k^', markersize=12, label='UAV Base')
    
    # 画目标点（圆圈大小代表搜索面积，透明度/颜色代表权重概率）
    for tgt in TARGETS:
        ax1.scatter(tgt['pos'][0], tgt['pos'][1], s=tgt['area']*20, 
                    c='orange', alpha=tgt['weight'], edgecolors='k')
        ax1.text(tgt['pos'][0]+2, tgt['pos'][1]+2, f"ID:{tgt['id']}\nW:{tgt['weight']:.2f}")

    # 画航线
    for k, route in enumerate(routes):
        if not route: continue
        route_pts = [UAV_BASE] + [TARGETS[idx]['pos'] for idx in route]
        xs = [p[0] for p in route_pts]
        ys = [p[1] for p in route_pts]
        ax1.plot(xs, ys, color=colors[k], linestyle='-', linewidth=2, 
                 marker='o', label=f'UAV {k+1} (Time: {finish_times[k]:.0f}s)')
        
        # 标出访问顺序
        for step, tgt_idx in enumerate(route):
            tgt_pos = TARGETS[tgt_idx]['pos']
            ax1.text(tgt_pos[0]-3, tgt_pos[1]-4, f"[{k+1}]-No.{step+1}", 
                     color=colors[k], fontweight='bold')

    ax1.set_title("Multi-UAV Cooperative Task Allocation")
    ax1.set_xlabel("X (km)")
    ax1.set_ylabel("Y (km)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # --- 图 2：收敛曲线 ---
    ax2.plot(cost_history, 'b-', linewidth=2)
    ax2.set_title("GA Convergence Curve")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Total Cost (J)")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    # save_path = 'uav_allocation_result.png'
    # plt.savefig(save_path, dpi=300)
    # print(f"\n✅ 分配结果已保存至：{save_path}")

if __name__ == "__main__":
    print("开始运行四机协同任务分配遗传算法 (GA)...")
    best_chrom, best_cost, history, final_norm_factors = run_ga(pop_size=200, generations=600, patience=100)
    
    _, best_routes, best_times = evaluate_chromosome(best_chrom, final_norm_factors)
    
    print("\n--- 最优任务分配结果 ---")
    for k in range(NUM_UAVS):
        print(f"无人机 {k+1}: 分配目标 ID -> {best_routes[k]}, 预计耗时: {best_times[k]:.1f} 秒")
        
    plot_results(best_routes, best_times, history)