import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

# ================= 工具与几何辅助函数 =================
def wrap_to_pi(angle):
    """将角度限制在 [-pi, pi] 之间"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def dedup_polyline(XY, tol=1e-9):
    """移除多段线中过近的重复点"""
    if len(XY) < 2:
        return XY
    d = np.hypot(np.diff(XY[:, 0]), np.diff(XY[:, 1]))
    keep = np.insert(d > tol, 0, True)
    XY_clean = XY[keep]
    if len(XY_clean) < 2:
        XY_clean = np.vstack([XY_clean, XY_clean[0] + [tol, 0]]) # 加入一个微小偏移点以保证至少两个点
    return XY_clean

def path_len(XY):
    """计算路径总长度"""
    if len(XY) < 2: return 0.0
    return np.sum(np.hypot(np.diff(XY[:, 0]), np.diff(XY[:, 1])))

def norm_arc_angles(a0, a1, turn_dir):
    """规范化圆弧的起止角度"""
    a0 = math.atan2(math.sin(a0), math.cos(a0))
    a1 = math.atan2(math.sin(a1), math.cos(a1))
    if turn_dir > 0:
        while a1 <= a0: a1 += 2 * np.pi
    else:
        while a1 >= a0: a1 -= 2 * np.pi
    return a0, a1

def sample_circle(O, R, th0, th1, turn_dir):
    """生成圆弧轨迹点集"""
    th0, th1 = norm_arc_angles(th0, th1, turn_dir) # 规范切入与切出角度
    dth = np.radians(1) if turn_dir > 0 else -np.radians(1)
    
    # 构建角度序列
    if turn_dir > 0:
        th = np.arange(th0, th1, dth)
        if len(th) == 0 or th[-1] < th1 - 1e-12:
            th = np.append(th, th1)
    else:
        th = np.arange(th0, th1, dth)
        if len(th) == 0 or th[-1] > th1 + 1e-12:
            th = np.append(th, th1)
            
    XY = np.column_stack([O[0] + R * np.cos(th), O[1] + R * np.sin(th)]) # 转换角度序列为二维点集
    return dedup_polyline(XY) # 去除过近的重复点

# ================= 核心螺旋片段生成 =================
def enter_first_ring_3quarter(C, r0, turn_dir, phi0):
    # C 是中心点，r0 是首圈半径，turn_dir 是转向，phi0 是切入角度
    """生成切入首圈的 3/4 桥接圆和直线"""
    n = np.array([math.cos(phi0), math.sin(phi0)]) # 圆心指向切入点的单位法向量
    t = np.array([-math.sin(phi0), math.cos(phi0)]) # 圆心指向切入点的单位切向量 (右手旋转90度)
    Oe = C - r0 * t # 切入圆心
    th0 = math.atan2(C[1] - Oe[1], C[0] - Oe[0])
    th1 = th0 + turn_dir * 3 * np.pi / 2
    
    arc3q_XY = sample_circle(Oe, r0, th0, th1, turn_dir)
    P3 = arc3q_XY[-1] # 3/4 圆弧的切出点
    P4 = C + r0 * n # 直线切入点
    line_XY = np.vstack([P3, P4])
    yawC = th0 + turn_dir * np.pi / 2
    return arc3q_XY, line_XY, yawC, P4

def fly_one_ring(C, r, phi_start, turn_dir):
    """生成完整的一圈环形航线"""
    theta0 = phi_start
    theta1 = phi_start + turn_dir * 2 * np.pi
    ring_XY = sample_circle(C, r, theta0, theta1, turn_dir)
    return ring_XY, 2 * np.pi * r

def bridge_quarter_arc(C, r_inner, h, phi_k, turn_dir):
    """生成从内圈向外圈扩展的 1/4 桥接圆和切线"""
    n = np.array([math.cos(phi_k), math.sin(phi_k)])
    t = np.array([-math.sin(phi_k), math.cos(phi_k)])
    
    P4 = C + r_inner * n
    A0 = P4 + h * t
    Ob = C + h * t
    
    th0 = math.atan2(A0[1] - Ob[1], A0[0] - Ob[0])
    th1 = th0 + turn_dir * np.pi / 2
    arc_XY = sample_circle(Ob, r_inner, th0, th1, turn_dir)
    line_XY = np.vstack([P4, A0])
    
    phi_next = wrap_to_pi(phi_k + np.pi / 2)
    return line_XY, arc_XY, phi_next

# ================= 螺旋线优化主控函数 =================
def spiral_search_arc_exact(prob_grid, center_km, radius_km, X_km, Y_km, yaw_at_center, cfg):
    """
    单区域应召搜索规划算法
    返回最优化组合的螺旋线轨迹及覆盖率等信息
    """
    # 1. 提取配置 (字典解析)
    Vg, H, Rmin = cfg['uav']['Vg'], cfg['uav']['H'], cfg['uav']['Rmin']
    dcty, hsub, overlap_frac = cfg['sensor']['dcty'], cfg['sensor']['hsub'], cfg['sensor']['overlapFrac']
    vTgt, epsR, tau0 = cfg['target']['vTgt'], cfg['target']['epsR'], cfg['target']['tau0']
    Dmax, mode_hard = cfg['opt']['Dmax'], cfg['opt']['modeHard']
    gamma, deltaP_stop = cfg['opt']['gamma'], cfg['opt']['deltaP_stop']
    lambda_r0 = cfg['opt'].get('lambdaR0', 0.0)
    r0_interval, h_interval, N_grid = cfg['grid']['r0interval'], cfg['grid']['hinterval'], cfg['grid']['NGrid']
    turn_dir = cfg['traj']['turnDir']

    # 2. 传感器搜索宽度计算
    Wcty = 2 * math.sqrt(max(dcty**2 - (H + hsub)**2, 0))
    wR = Wcty / 2
    hMax = (1 - overlap_frac) * Wcty
    
    # 半径、间距和圈数的范围设定
    r0Min = max([Rmin, vTgt * tau0 + epsR, wR])
    r0Max = max(1.5 * r0Min, r0Min + 2000)
    r0_array = np.linspace(r0Min, r0Max, r0_interval)
    h_array = np.linspace(0.4 * Wcty, min(1.0 * Wcty, hMax), h_interval)

    center_m = np.array(center_km) * 1000.0
    theta0 = yaw_at_center

    # 3. 预处理概率场 (向量化加速环带概率计算)
    # 计算所有网格点到中心的距离(km)
    D_km = np.hypot(X_km - center_km[0], Y_km - center_km[1])
    # 目标区域总概率质量
    mask_region = D_km <= radius_km
    Ic_sel = np.sum(prob_grid[mask_region])
    Ic_sel = max(Ic_sel, 1e-9)

    # 定义环带概率计算函数
    def annulus_prob(r0, h, N, wR):
        if N <= 0: return 0.0
        r_in_km = max((r0 - wR) / 1000.0, 0)
        r_out_km = (r0 + (N - 1) * h + wR) / 1000.0
        mask_ring = (D_km >= r_in_km) & (D_km <= r_out_km) & mask_region 
        If = np.sum(prob_grid[mask_ring])
        return If / Ic_sel

    # 4. 暴力搜索最优 (r0, h, N) 组合
    # 引入首圈半径偏好：对较大搜索区，首圈不应总压在Rmin边界
    region_radius_m = radius_km * 1000.0
    r0_pref = min(max(0.5 * region_radius_m, r0Min), r0Max)
    best_J = float('inf')
    best_sol = None

    for r0 in r0_array:
        if r0 < Rmin - 1e-6: continue
        for h in h_array:
            if h > hMax + 1e-9: continue
            for N in N_grid:
                phi0 = wrap_to_pi(theta0 + np.pi)
                
                # --- 生成首圈 ---
                arc3, line3to4, _, P4 = enter_first_ring_3quarter(center_m, r0, turn_dir, phi0)
                ring1, len_r1 = fly_one_ring(center_m, r0, phi0, turn_dir)
                
                # 记录累计路径的轨迹段、长度和切入点
                all_segs = [arc3, line3to4, ring1]
                lens = [path_len(arc3), path_len(line3to4), len_r1]
                cut_ins = [P4]
                total_rings = 1
                
                # --- 迭代向外圈扩展 ---
                prev_P = annulus_prob(r0, h, total_rings, wR)
                
                while total_rings < N:
                    phi_k_prescr = wrap_to_pi(theta0 + np.pi + (total_rings - 1) * np.pi / 2)# 预设的切入角度 (每圈增加90度)
                    r_inner = r0 + (total_rings - 1) * h
                    if r_inner < Rmin - 1e-6: break
                    
                    # 桥接并生成外圈
                    lineXY, arcXY, phi_next = bridge_quarter_arc(center_m, r_inner, h, phi_k_prescr, turn_dir)
                    r_outer = r_inner + h
                    ringNext, lenNext = fly_one_ring(center_m, r_outer, phi_next, turn_dir)
                    
                    dL = path_len(lineXY) + path_len(arcXY) + lenNext # 本次扩展增加的路径长度
                    
                    all_segs.extend([lineXY, arcXY, ringNext])
                    lens.extend([path_len(lineXY), path_len(arcXY), lenNext])
                    total_rings += 1
                    cut_ins.append(arcXY[-1])
                    
                    cur_len = sum(lens)
                    
                    # 停止条件判定
                    # 1. 硬约束模式：如果当前路径长度超过 Dmax，立即回退到上一个可行解
                    if mode_hard and cur_len > Dmax + 1e-6:
                        all_segs, lens, cut_ins = all_segs[:-3], lens[:-3], cut_ins[:-1]
                        total_rings -= 1
                        break
                        
                    Pnow = annulus_prob(r0, h, total_rings, wR) # 当前环带覆盖概率
                    # 2. 软约束模式：如果增益 dJ 不再提升，回退到上一个解
                    if mode_hard:
                        if Pnow - prev_P < deltaP_stop: # 如果概率增益小于阈值，认为没有显著提升，回退
                            all_segs, lens, cut_ins = all_segs[:-3], lens[:-3], cut_ins[:-1]
                            total_rings -= 1
                            break
                    else:
                        # 计算增益 dJ = gamma * (路径长度增量 / Dmax) - (1 - gamma) * (覆盖率增量)，如果 dJ >= 0 则认为没有提升，回退
                        dJ = gamma * (dL / max(Dmax, 1)) - (1 - gamma) * (Pnow - prev_P)
                        if dJ >= 0:
                            all_segs, lens, cut_ins = all_segs[:-3], lens[:-3], cut_ins[:-1]
                            total_rings -= 1
                            break
                    prev_P = Pnow

                if len(all_segs) == 0: continue
                full_XY = np.vstack(all_segs)
                total_len = sum(lens)
                
                if mode_hard and total_len > Dmax + 1e-6: continue
                
                # 计算最终代价 J=路径长度与覆盖率的加权和
                Pdet = annulus_prob(r0, h, total_rings, wR)
                r0_penalty = lambda_r0 * abs(r0 - r0_pref) / max(region_radius_m, 1.0)
                J = gamma * (total_len / max(Dmax, 1)) + (1 - gamma) * (1 - Pdet) + r0_penalty
                
                if J < best_J:
                    best_J = J
                    idxStart = len(arc3) + len(line3to4)
                    
                    # 记录终点和航向
                    pEnd = full_XY[-1]
                    if len(full_XY) >= 2:
                        v = full_XY[-1] - full_XY[-2]
                    else:
                        v = np.array([1.0, 0.0])
                    yawEnd = math.atan2(v[1], v[0])
                    
                    best_sol = {
                        'path': full_XY,
                        'totalLen': total_len,
                        'totalTime': total_len / Vg,
                        'r0': r0, 'h': h, 'N': total_rings,
                        'Wcty': Wcty, 'Pdet': Pdet,
                        'center': center_m,
                        'cutIns': np.array(cut_ins),
                        'idxStart': idxStart,
                        'pEnd': pEnd,
                        'yawEnd': yawEnd
                    }

    if best_sol is None:
        raise ValueError("未能找到可行解，请检查输入参数、放宽 Dmax 或减小 Rmin。")
        
    return best_sol

# ================= 独立测试代码 =================
if __name__ == "__main__":
    # 1. 构建配置字典 (复刻 MATLAB cfg)
    cfg = {
        'uav': {'Vg': 50.0, 'H': 100.0, 'Rmin': 2000.0},
        'sensor': {'dcty': 600.0, 'hsub': 20.0, 'overlapFrac': 0.1, 'width_model': 'sqrt'},
        'target': {'vTgt': 5.0, 'epsR': 200.0, 'tau0': 0.0},
        'opt': {'Dmax': 150000.0, 'modeHard': True, 'gamma': 0.01, 'deltaP_stop': 0.005},
        'grid': {'r0interval': 10, 'hinterval': 20, 'NGrid': list(range(1, 7))},
        'traj': {'turnDir': 1} # 1 逆时针，-1 顺时针
    }
    
    # 2. 伪造一个 15km x 15km 的测试网格概率场
    x = np.linspace(0, 15, 150)
    y = np.linspace(0, 15, 150)
    X_km, Y_km = np.meshgrid(x, y)
    center_km = (7.5, 7.5)
    
    # 构建高斯分布模拟热点概率
    prob_grid = np.exp(-((X_km - center_km[0])**2 + (Y_km - center_km[1])**2) / (2 * 2.0**2))
    prob_grid /= np.sum(prob_grid)
    
    # 3. 运行核心螺旋规划算法
    print("正在计算最佳螺旋搜索航线...")
    yaw_entry = np.radians(45) # 假设无人机以 45 度角到达区域中心
    solution = spiral_search_arc_exact(
        prob_grid=prob_grid, 
        center_km=center_km, 
        radius_km=5.0, 
        X_km=X_km, Y_km=Y_km, 
        yaw_at_center=yaw_entry, 
        cfg=cfg
    )
    
    # 4. 可视化输出结果
    path = solution['path']
    center_m = solution['center']
    cut_ins = solution['cutIns']
    
    plt.figure(figsize=(10, 10))
    # 绘制连续轨迹
    plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=1.5, label=f"Spiral Path (Len: {solution['totalLen']:.0f}m)")
    # 绘制中心点
    plt.plot(center_m[0], center_m[1], 'r*', markersize=12, label="Hotspot Center")
    # 绘制切入点
    if len(cut_ins) > 0:
        plt.plot(cut_ins[:, 0], cut_ins[:, 1], 'ro', markerfacecolor='w', markersize=6, label="Cut-in Points")
    
    # 标记起始方向
    entry_start = center_m - np.array([math.cos(yaw_entry), math.sin(yaw_entry)]) * 800
    plt.arrow(entry_start[0], entry_start[1], 600*math.cos(yaw_entry), 600*math.sin(yaw_entry), 
              head_width=150, head_length=200, fc='g', ec='g', label='Entry Heading')

    plt.title(f"Optimized Dense Spiral (Rings: {solution['N']}, Coverage: {solution['Pdet']*100:.1f}%)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # save_name = 'spiral_search_python.png'
    # plt.savefig(save_name, dpi=300)
    # print(f"✅ 测试完成，螺旋轨迹已保存为：{save_name}")
    print(f"📊 规划结果: {solution['N']} 圈，搜索覆盖率: {solution['Pdet']*100:.2f}%，总航程: {solution['totalLen']:.1f} 米")
