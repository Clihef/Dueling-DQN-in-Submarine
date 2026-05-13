"""
突发事件 — 工具函数、状态向量、FSM、路由操作
从 simulator.py 拆分出来，供 train/eval/demo 使用
"""
import numpy as np
import copy

# ==================== 全局常量 ====================
NUM_UAVS = 4
NUM_TARGETS = 15
MAX_NEW_TARGETS = 3
STATE_DIM = 38
SPAN_KM = 100.0
UAV_V_M_S = 50.0
UAV_VELOCITY_KM_S = 0.05
MAX_RANGE_KM = 350.0       # 每架UAV最大航程 (km)
MAX_FLIGHT_TIME_S = MAX_RANGE_KM / UAV_VELOCITY_KM_S # 7000秒
GLOBAL_C_DROP = 18000.0   # 统一的放弃惩罚
SOFT_RANGE_KM = 245.0       # 软约束阈值 (70% MAX_RANGE, 350*0.7=245)
MAX_DIAG_KM = 141.0  # sqrt(100^2 + 100^2)
MAX_RADIUS_KM = 5.0

# ==================== V2 MDP 常量 ====================
N_MAX = 20              # 目标槽位上限 (15初始 + 3突发 + 2预留)
STATE_DIM_V2 = 116      # 12(UAV矩阵) + 20×5(目标+competitive_ratio) + 4(决策者one-hot)
ALPHA_REWARD = 1.0      # 目标权重系数
BETA_REWARD = 0.8       # 距离惩罚系数（归一化距离 dist/MAX_DIAG_KM），使远距离低权重目标收益为负
R_SUCCESS = 10.0        # 全局通关奖励
GAMMA_UNVISITED = 5.0   # 未访问目标权重惩罚系数


# ==================== 路由/染色体转换 ====================

def routes_to_chromosome(routes, num_targets, num_uavs):
    """将路由列表转换为GA染色体格式"""
    chrom = []
    separator_val = num_targets
    for k in range(num_uavs): 
        if k > 0:
            chrom.append(separator_val)
            separator_val += 1
        chrom.extend(routes[k])
    return chrom


def chromosome_to_routes(chrom, num_targets, num_uavs):
    """将GA染色体解码为路由列表"""
    routes = [[] for _ in range(num_uavs)]
    uav_idx = 0
    for gene in chrom:
        if gene >= num_targets:
            uav_idx += 1
        else:
            if uav_idx < num_uavs:
                routes[uav_idx].append(gene)
    return routes


# ==================== 基于综合代价的贪心插入 ====================

def _cheapest_insertion(target, uav_idx, routes, uav_positions, id_to_hotspot, emergency=None,
                        consumed_ranges=None, baseline_routes=None, completed=None, UAV_BASE_KM=(0.0, 0.0)):
    """纯物理距离驱动的贪心插入，保持与评价函数严格一致的螺旋扣减逻辑"""
    route = routes[uav_idx]
    if not route:
        route.append(target['id'])
        return

    start_pos = uav_positions[uav_idx]
    best_idx = 0
    best_cost = float('inf')

    active_target_id = None
    if baseline_routes and completed is not None and uav_idx < len(baseline_routes):
        for b_tid in baseline_routes[uav_idx]:
            if b_tid not in completed:
                active_target_id = b_tid
                break

    for i in range(len(route) + 1):
        temp_route = route[:i] + [target['id']] + route[i:]
        
        dist_sum = 0.0
        curr_pos = start_pos
        for step_i, tid in enumerate(temp_route):
            tgt = id_to_hotspot.get(tid)
            if tgt is None: continue
            
            dist = np.hypot(curr_pos[0] - tgt['center_km'][0], curr_pos[1] - tgt['center_km'][1])
            spiral_km = tgt.get('spiral_dist', tgt.get('spiral_time', 500) * 50.0) / 1000.0
            actual_dist = dist
            actual_spiral = spiral_km
            
            # 精确打折逻辑
            if step_i == 0 and tid == active_target_id and consumed_ranges is not None and baseline_routes is not None:
                comp_targets = []
                for b_tid in baseline_routes[uav_idx]:
                    if b_tid == active_target_id:
                        break
                    comp_targets.append(b_tid)
                    
                comp_spiral_len = sum([id_to_hotspot.get(t, {}).get('spiral_dist', 500*50.0)/1000.0 for t in comp_targets])
                
                transit_len = 0.0
                p = UAV_BASE_KM
                for t in comp_targets:
                    pos = id_to_hotspot.get(t, {}).get('center_km', p)
                    transit_len += np.hypot(pos[0] - p[0], pos[1] - p[1])
                    p = pos
                transit_len += np.hypot(tgt['center_km'][0] - p[0], tgt['center_km'][1] - p[1])
                
                progress_km = consumed_ranges[uav_idx] - comp_spiral_len - transit_len
                if progress_km > 0:
                    actual_spiral = max(0.0, spiral_km - progress_km)
                    actual_dist = 0.0
            
            # 禁飞区硬隔离（确保在分配时避开明显穿越禁区的目标串）
            if emergency and emergency.get('type') == 'S3':
                nf_cx, nf_cy = emergency['no_fly_center']
                nf_r = emergency['no_fly_radius']
                if _segment_intersects_circle(curr_pos, tgt['center_km'], nf_cx, nf_cy, nf_r):
                    actual_dist += 500.0 # 插入阶段拒绝穿越
                    
            dist_sum += actual_dist + actual_spiral
            curr_pos = tgt['center_km']
        
        if dist_sum < best_cost:
            best_cost = dist_sum
            best_idx = i
            
    routes[uav_idx].insert(best_idx, target['id'])

def apply_decision(action, target, routes, abandoned_list, uav_positions=None, id_to_hotspot=None, emergency=None,
                   consumed_ranges=None, baseline_routes=None, completed=None, UAV_BASE_KM=(0.0,0.0)):
    """
    执行单步DQN决策，并传入 emergency 用于禁飞区避让
    """
    target_id = target['id']
    if action == 0:
        abandoned_list.append(target_id)
    else:
        uav_idx = action - 1
        for k in range(len(routes)):
            if target_id in routes[k]:
                routes[k].remove(target_id)
                
        if uav_positions is not None and id_to_hotspot is not None:
            # 🌟 透传 emergency
            _cheapest_insertion(target, uav_idx, routes, uav_positions, id_to_hotspot, emergency=None,
                        consumed_ranges=None, baseline_routes=None, completed=None, UAV_BASE_KM=(0.0, 0.0))
        else:
            routes[uav_idx].append(target_id)


# ==================== S3禁飞区绕行 ====================

def replan_around_no_fly(routes, uav_positions, no_fly_center, no_fly_radius,
                          id_to_hotspot):
    """对每条UAV路由检测禁飞区穿越，插入绕行航路点

    修改方式：为每条被阻断航段插入2个绕行点（圆两侧切点），
    返回带绕行点的"虚拟路径点列表"，供可视化用。原始路由不变。

    Args:
        routes: list of lists, 每架UAV的目标ID序列
        uav_positions: (4,2) 各UAV当前位置
        no_fly_center: (x, y) 禁飞区中心
        no_fly_radius: float 禁飞区半径
        id_to_hotspot: dict, target_id → hotspot

    Returns:
        waypoint_paths: list of lists, 每架UAV的路径点序列 [(x,y), ...]
            (包含目标位置 + 绕行点，可直接用于画线)
    """
    nf_cx, nf_cy = no_fly_center
    waypoint_paths = []

    for k, route in enumerate(routes):
        points = [tuple(uav_positions[k])]  # 起始点：UAV当前位置
        for tgt_id in route:
            h = id_to_hotspot.get(tgt_id)
            if h is None:
                continue
            tgt_pos = h['center_km']
            # 检测当前段 (points[-1] → tgt_pos) 是否穿越禁飞区
            if _segment_intersects_circle(points[-1], tgt_pos,
                                          nf_cx, nf_cy, no_fly_radius):
                # 计算绕行点：在圆心到线段垂线的反方向偏移
                detour = _compute_detour_waypoint(
                    points[-1], tgt_pos, nf_cx, nf_cy, no_fly_radius
                )
                if detour is not None:
                    points.append(detour)
            points.append(tgt_pos)
        waypoint_paths.append(points)

    return waypoint_paths


def _segment_intersects_circle(p1, p2, cx, cy, r):
    """线段与圆是否相交 (同 simulator._line_intersects_circle)"""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    fx = x1 - cx
    fy = y1 - cy
    a = dx * dx + dy * dy
    if a < 1e-9:
        return np.hypot(fx, fy) <= r
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    disc = np.sqrt(disc)
    t1 = (-b - disc) / (2 * a)
    t2 = (-b + disc) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 <= 0 and t2 >= 1)


def _compute_detour_waypoint(p1, p2, cx, cy, r):
    """计算绕行航路点：在圆心到线段垂线方向，偏移到圆外"""
    x1, y1 = p1
    x2, y2 = p2
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx_c = mx - cx
    dy_c = my - cy
    dist_c = np.hypot(dx_c, dy_c)
    if dist_c < 1e-9:
        dx_c = -(y2 - y1)
        dy_c = x2 - x1
        dist_c = np.hypot(dx_c, dy_c)
    if dist_c < 1e-9:
        return None
    scale = (r * 1.2) / dist_c
    return (cx + dx_c * scale, cy + dy_c * scale)


def compute_detour_distance_km(p1, p2, cx, cy, r):
    """计算绕飞禁飞区的实际额外距离 (km)

    通过在圆外插入绕行点，计算 p1→waypoint→p2 的实际路径长度，
    减去直线距离 p1→p2，得到绕飞额外代价。

    Args:
        p1, p2: (x_km, y_km) 航段两端
        cx, cy, r: 禁飞区圆心和半径 (km)

    Returns:
        extra_km: 绕飞额外距离 (km), 0 表示无需绕飞或无法计算
    """
    if not _segment_intersects_circle(p1, p2, cx, cy, r):
        return 0.0
    wp = _compute_detour_waypoint(p1, p2, cx, cy, r)
    if wp is None:
        return 2.5 * r  # 回退到工程估算
    d_straight = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
    d_detour = np.hypot(wp[0] - p1[0], wp[1] - p1[1]) + \
               np.hypot(p2[0] - wp[0], p2[1] - wp[1])
    return max(d_detour - d_straight, 0.0)


# ==================== 路由负载均衡 ====================

def _uav_load(routes_k, id_to_hotspot):
    """估算单架UAV的任务负载（螺旋搜索时间，秒）"""
    total = 0.0
    for tid in routes_k:
        h = id_to_hotspot.get(tid)
        if h and 'spiral_time' in h:
            total += h['spiral_time']
        elif h:
            total += h.get('radius_km', 2.0) * 100.0  # 粗略回退
    return total


# ==================== 航程计算 ====================

def compute_route_distance_km(routes, uav_positions, id_to_hotspot, emergency=None, 
                              consumed_ranges=None, baseline_routes=None, completed=None, UAV_BASE_KM=(0.0, 0.0)):
    """计算每架UAV路由的总飞行距离，引入精确螺旋进度扣减"""
    total_distances = []
    for k, route in enumerate(routes):
        total = 0.0
        prev_pos = tuple(uav_positions[k])
        
        # 1. 精确定位该架 UAV 突发时正在执行的“活跃目标”
        active_target_id = None
        if baseline_routes and completed is not None and k < len(baseline_routes):
            for tid in baseline_routes[k]:
                if tid not in completed:
                    active_target_id = tid
                    break
        
        for i, tid in enumerate(route):
            h = id_to_hotspot.get(tid)
            if h is None:
                continue
            tgt_pos = h['center_km']
            dist = np.hypot(tgt_pos[0] - prev_pos[0], tgt_pos[1] - prev_pos[1])
            spiral_km = h.get('spiral_dist', h.get('spiral_time', 500) * 50.0) / 1000.0
            
            actual_dist = dist
            actual_spiral = spiral_km
            
            # 2. 真实搜索长度进度计算核心逻辑
            if i == 0 and tid == active_target_id and consumed_ranges is not None and baseline_routes is not None:
                comp_targets = []
                for b_tid in baseline_routes[k]:
                    if b_tid == active_target_id:
                        break
                    comp_targets.append(b_tid)
                    
                # 减去：所有已经搜索完成的目标需要的螺旋路径长度和
                comp_spiral_len = sum([id_to_hotspot.get(t, {}).get('spiral_dist', 500*50.0)/1000.0 for t in comp_targets])
                
                # 减去：走到此处的转场路径长度和
                transit_len = 0.0
                p = UAV_BASE_KM
                for t in comp_targets:
                    pos = id_to_hotspot.get(t, {}).get('center_km', p)
                    transit_len += np.hypot(pos[0] - p[0], pos[1] - p[1])
                    p = pos
                transit_len += np.hypot(tgt_pos[0] - p[0], tgt_pos[1] - p[1])
                
                progress_km = consumed_ranges[k] - comp_spiral_len - transit_len
                
                # 如果 progress_km > 0，说明已经进入了该螺旋搜索圈
                if progress_km > 0:
                    actual_spiral = max(0.0, spiral_km - progress_km)
                    actual_dist = 0.0 # 已经到达该目标区域内，转场距离归零
                    
            # 3. S3 禁飞区阻力（即时惩罚）
            if emergency and emergency.get('type') == 'S3':
                nf_cx, nf_cy = emergency['no_fly_center']
                nf_r = emergency['no_fly_radius']
                detour = compute_detour_distance_km(prev_pos, tgt_pos, nf_cx, nf_cy, nf_r)
                actual_dist += detour
                
            total += actual_dist + actual_spiral
            prev_pos = tgt_pos
        total_distances.append(total)
    return total_distances

# ==================== 状态向量构建 ====================

def build_state_vector(current_target, uav_positions, active_uavs, uav_routes,
                       affected_targets, processed_count, emergency_info,
                       consumed_ranges=None):
    """构建38维状态向量，聚焦当前决策目标

    Args:
        current_target: dict, 当前决策目标 (含 id, center_km, weight, radius_km)
        uav_positions: (4, 2) 每架UAV的 (x_km, y_km)
        active_uavs: bool[4] 哪些UAV活跃
        uav_routes: list of lists, 每架UAV当前路由 (目标ID列表)
        affected_targets: list of dict, 所有受影响目标
        processed_count: int, 已处理目标数
        emergency_info: dict, 突发事件信息
        consumed_ranges: list[float]|None, 各UAV已消耗航程(km), None则填0

    Returns:
        state: (38,) float32 numpy array
    """
    state = np.zeros(STATE_DIM, dtype=np.float32)
    idx = 0

    # --- UAV模块 (4×5=20) ---
    for k in range(NUM_UAVS):
        state[idx] = uav_positions[k][0] / SPAN_KM
        state[idx + 1] = uav_positions[k][1] / SPAN_KM
        idx += 2

    for k in range(NUM_UAVS):
        state[idx] = 1.0 if active_uavs[k] else 0.0
        idx += 1

    tgt_x, tgt_y = current_target['center_km']
    for k in range(NUM_UAVS):
        d = np.hypot(uav_positions[k][0] - tgt_x,
                     uav_positions[k][1] - tgt_y)
        state[idx] = d / MAX_DIAG_KM
        idx += 1

    for k in range(NUM_UAVS):
        state[idx] = min(len(uav_routes[k]) / NUM_TARGETS, 1.0)
        idx += 1

    # --- UAV航程利用率 (4维) ---
    ranges = consumed_ranges if consumed_ranges is not None else [0.0] * NUM_UAVS
    for k in range(NUM_UAVS):
        state[idx] = min(ranges[k] / MAX_RANGE_KM, 1.0)
        idx += 1

    # --- 当前目标模块 (5维) ---
    state[idx] = tgt_x / SPAN_KM
    idx += 1
    state[idx] = tgt_y / SPAN_KM
    idx += 1
    state[idx] = current_target.get('weight', 0.0)
    idx += 1
    state[idx] = current_target.get('radius_km', 2.0) / MAX_RADIUS_KM
    idx += 1
    state[idx] = 1.0 if current_target.get('id', 0) >= 100 else 0.0
    idx += 1

    # --- 突发事件类型 one-hot (4维) ---
    etype = emergency_info.get('type', 'S1')
    onehot = {'S1': [1, 0, 0, 0], 'S2': [0, 1, 0, 0],
              'S3': [0, 0, 1, 0], 'S4': [0, 0, 0, 1]}
    for v in onehot.get(etype, [0, 0, 0, 0]):
        state[idx] = float(v)
        idx += 1

    # --- 禁飞区信息 (3维, 仅S3非零) ---
    if etype == 'S3':
        nf_c = emergency_info.get('no_fly_center', (0, 0))
        state[idx] = nf_c[0] / SPAN_KM
        state[idx + 1] = nf_c[1] / SPAN_KM
        state[idx + 2] = emergency_info.get('no_fly_radius', 0) / SPAN_KM
    idx += 3

    # --- 序列上下文 (2维) ---
    total_affected = max(len(affected_targets), 1)
    remaining = total_affected - processed_count
    state[idx] = remaining / total_affected
    idx += 1

    if remaining > 0:
        remaining_targets = affected_targets[processed_count:]
        avg_w = np.mean([t.get('weight', 0) for t in remaining_targets])
    else:
        avg_w = 0.0
    state[idx] = avg_w

    return state


# ==================== 受影响目标提取 ====================

def get_affected_targets(emergency, routes, hotspots):
    """从突发事件提取受影响目标列表 (按权重降序)

    Args:
        emergency: dict, 突发事件信息
        routes: list of lists, 当前各UAV路由
        hotspots: list of dict, 热点列表

    Returns:
        affected_targets: list of dict, 受影响目标 (含完整信息, 按权重降序)
    """
    etype = emergency['type']
    id_to_hotspot = {h['id']: h for h in hotspots}
    affected = []

    if etype == 'S1':
        for shift in emergency.get('shifts', []):
            tid = shift['id']
            h = id_to_hotspot.get(tid)
            if h:
                tgt = copy.deepcopy(h)
                tgt['center_km'] = (tgt['center_km'][0] + shift['dx'],
                                    tgt['center_km'][1] + shift['dy'])
                affected.append(tgt)

    elif etype == 'S2':
        for nt in emergency.get('new_targets', []):
            affected.append(copy.deepcopy(nt))

    elif etype == 'S3':
        nf_cx, nf_cy = emergency.get('no_fly_center', (0,0))
        nf_r = emergency.get('no_fly_radius', 0)
        
        for route in routes:
            for i in range(len(route) - 1):
                p1 = id_to_hotspot[route[i]]['center_km']
                p2 = id_to_hotspot[route[i+1]]['center_km']
                # 检测到阻断
                if _segment_intersects_circle(p1, p2, nf_cx, nf_cy, nf_r):
                    # 将该节点及其之后的所有目标全部剥离重新分配
                    for tid in route[i:]:
                        h = id_to_hotspot.get(tid)
                        if h and h not in affected:
                            affected.append(copy.deepcopy(h))
                    break # 跳出当前 route 的检测
        
        # 兜底逻辑
        if not affected:
            for tid in emergency.get('affected_targets', []):
                h = id_to_hotspot.get(tid)
                if h and h not in affected:
                    affected.append(copy.deepcopy(h))

    elif etype == 'S4':
        for tid in emergency.get('lost_targets', []):
            h = id_to_hotspot.get(tid)
            if h:
                affected.append(copy.deepcopy(h))

    affected.sort(key=lambda t: t.get('weight', 0), reverse=True)
    return affected


# ==================== FSM 状态机 ====================

class EmergencyFSM:
    """突发事件有限状态机

    状态:
        NORMAL (0) — GA方案正常执行中
        EMERGENCY (1) — 事件触发，DQN逐目标决策中

    转换:
        NORMAL → EMERGENCY: 事件触发
        EMERGENCY → NORMAL: 所有受影响目标处理完毕
    """

    NORMAL = 0
    EMERGENCY = 1

    def __init__(self):
        self.state = self.NORMAL
        self.state_names = {self.NORMAL: 'NORMAL', self.EMERGENCY: 'EMERGENCY'}

    def trigger_emergency(self, emergency):
        if self.state != self.NORMAL:
            return False
        self.state = self.EMERGENCY
        return True

    def resolve_emergency(self):
        if self.state != self.EMERGENCY:
            return False
        self.state = self.NORMAL
        return True

    def is_normal(self):
        return self.state == self.NORMAL

    def is_emergency(self):
        return self.state == self.EMERGENCY

    @property
    def current_state(self):
        return self.state_names.get(self.state, 'UNKNOWN')


# ==================== V2 MDP 函数 ====================

def build_state_vector_v2(uav_states, targets, current_uav_idx):
    """构建116维全局状态向量 (V2 轮询MDP)"""
    state = np.zeros(STATE_DIM_V2, dtype=np.float32)
    state[0:12] = uav_states.flatten()
    ux = uav_states[current_uav_idx][0] * SPAN_KM
    uy = uav_states[current_uav_idx][1] * SPAN_KM
    for j in range(N_MAX):
        base = 12 + j * 5
        if j < len(targets):
            t = targets[j]
            cx, cy = t['center_km']
            state[base]     = cx / SPAN_KM
            state[base + 1] = cy / SPAN_KM
            state[base + 2] = t['weight']
            state[base + 3] = float(t['mask'])
            d_current = np.hypot(cx - ux, cy - uy) + 1e-6
            d_others = [
                np.hypot(cx - uav_states[k][0]*SPAN_KM, cy - uav_states[k][1]*SPAN_KM)
                for k in range(4) if k != current_uav_idx
            ]
            raw_ratio = min(d_others) / d_current if d_others else 1.0
            state[base + 4] = np.clip(raw_ratio, 0.0, 5.0)
    state[112 + current_uav_idx] = 1.0
    return state


def get_valid_actions_v2(uav_idx, uav_states, targets, locked_target_idx=None, emergency=None):
    """返回当前UAV的合法动作掩码 (V2 轮询MDP)"""
    if locked_target_idx is not None:
        mask = np.zeros(N_MAX, dtype=np.bool_)
        if locked_target_idx < len(targets) and not targets[locked_target_idx]['mask']:
            mask[locked_target_idx] = True
        return mask
    valid_mask = np.zeros(N_MAX, dtype=np.bool_)
    remaining_km = uav_states[uav_idx][2] * MAX_RANGE_KM
    ux = uav_states[uav_idx][0] * SPAN_KM
    uy = uav_states[uav_idx][1] * SPAN_KM

    for j in range(min(len(targets), N_MAX)):
        t = targets[j]
        if t['mask']:
            continue
        cx, cy = t['center_km']
        dist = np.hypot(cx - ux, cy - uy)
        if emergency and emergency.get('type') == 'S3':
            nf_cx, nf_cy = emergency['no_fly_center']
            nf_r = emergency['no_fly_radius']
            if _segment_intersects_circle((ux, uy), (cx, cy), nf_cx, nf_cy, nf_r):
                dist += 500.0
            else:
                dist += compute_detour_distance_km((ux, uy), (cx, cy), nf_cx, nf_cy, nf_r)
        spiral = t.get('spiral_dist', 0.0) / 1000.0
        if dist + spiral <= remaining_km:
            valid_mask[j] = True
    return valid_mask


def apply_action_v2(uav_idx, target_idx, uav_states, targets, emergency=None):
    """执行轮询MDP动作，原地更新状态 (V2)"""
    t = targets[target_idx]
    cx, cy = t['center_km']
    ux = uav_states[uav_idx][0] * SPAN_KM
    uy = uav_states[uav_idx][1] * SPAN_KM
    dist_km = np.hypot(cx - ux, cy - uy)
    if emergency and emergency.get('type') == 'S3':
        nf_cx, nf_cy = emergency['no_fly_center']
        nf_r = emergency['no_fly_radius']
        if _segment_intersects_circle((ux, uy), (cx, cy), nf_cx, nf_cy, nf_r):
            dist_km += 500.0
        else:
            dist_km += compute_detour_distance_km((ux, uy), (cx, cy), nf_cx, nf_cy, nf_r)
    spiral = t.get('spiral_dist', 0.0) / 1000.0
    total_cost_km = dist_km + spiral
    uav_states[uav_idx][0] = cx / SPAN_KM
    uav_states[uav_idx][1] = cy / SPAN_KM
    uav_states[uav_idx][2] -= total_cost_km / MAX_RANGE_KM
    t['mask'] = True
    return dist_km, total_cost_km


# ==================== 自测 ====================

if __name__ == '__main__':
    from core import heatmap as hm

    print("=== 突发事件工具函数自测 ===\n")

    span_km = 100.0
    dx_km = 0.1
    x_km = np.arange(0, span_km + dx_km / 2, dx_km)
    y_km = np.arange(0, span_km + dx_km / 2, dx_km)
    X_km, Y_km = np.meshgrid(x_km, y_km)

    hotspots = [
        {'id': 0, 'center_km': (15, 15), 'weight': 0.25, 'radius_km': 2.5},
        {'id': 1, 'center_km': (25, 10), 'weight': 0.30, 'radius_km': 3.0},
        {'id': 2, 'center_km': (10, 30), 'weight': 0.40, 'radius_km': 2.0},
    ]

    # 1. 状态向量
    print("--- 状态向量测试 ---")
    state = build_state_vector(
        {'id': 0, 'center_km': (15, 15), 'weight': 0.25, 'radius_km': 2.5},
        np.array([[10, 20], [30, 40], [50, 60], [0, 0]], dtype=float),
        [True, True, True, True],
        [[0], [1], [2], []],
        [{'id': 1, 'center_km': (25, 10), 'weight': 0.30, 'radius_km': 3.0},
         {'id': 0, 'center_km': (15, 15), 'weight': 0.25, 'radius_km': 2.5}],
        0,
        {'type': 'S1', 'severity': 0.5},
    )
    assert state.shape[0] == STATE_DIM, f"维度错误: {state.shape[0]} != {STATE_DIM}"
    print(f"[OK] 状态向量维度: {STATE_DIM}\n")

    # 2. FSM
    print("--- FSM测试 ---")
    fsm = EmergencyFSM()
    assert fsm.is_normal()
    fsm.trigger_emergency({'type': 'S1'})
    assert fsm.is_emergency()
    fsm.resolve_emergency()
    assert fsm.is_normal()
    print("[OK] FSM: NORMAL → EMERGENCY → NORMAL\n")

    print("=== 自测全部通过 ===")
