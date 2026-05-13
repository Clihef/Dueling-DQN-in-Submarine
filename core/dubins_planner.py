import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ================= 定义路径片段类型 =================
L_SEG = 1  # 左转圆弧
S_SEG = 2  # 直线
R_SEG = 3  # 右转圆弧

# 6种可能的 Dubins 曲线类型
DIRDATA = [
    [L_SEG, S_SEG, L_SEG],  # 0: LSL
    [L_SEG, S_SEG, R_SEG],  # 1: LSR
    [R_SEG, S_SEG, L_SEG],  # 2: RSL
    [R_SEG, S_SEG, R_SEG],  # 3: RSR
    [R_SEG, L_SEG, R_SEG],  # 4: RLR
    [L_SEG, R_SEG, L_SEG]   # 5: LRL
]

# ================= 核心求解算子 =================
# 基于 Shkel and Lumelsky (2001) 分类法的 6 种曲线求解器
def dubins_LSL(alpha, beta, d):
    tmp0 = d + math.sin(alpha) - math.sin(beta)
    p_squared = 2 + (d * d) - (2 * math.cos(alpha - beta)) + (2 * d * (math.sin(alpha) - math.sin(beta)))
    if p_squared < 0: return None
    tmp1 = math.atan2((math.cos(beta) - math.cos(alpha)), tmp0)
    t = (-alpha + tmp1) % (2 * math.pi)
    p = math.sqrt(p_squared)
    q = (beta - tmp1) % (2 * math.pi)
    return [t, p, q]

def dubins_LSR(alpha, beta, d):
    p_squared = -2 + (d * d) + (2 * math.cos(alpha - beta)) + (2 * d * (math.sin(alpha) + math.sin(beta)))
    if p_squared < 0: return None
    p = math.sqrt(p_squared)
    tmp2 = math.atan2((-math.cos(alpha) - math.cos(beta)), (d + math.sin(alpha) + math.sin(beta))) - math.atan2(-2.0, p)
    t = (-alpha + tmp2) % (2 * math.pi)
    q = (-beta % (2 * math.pi) + tmp2) % (2 * math.pi)
    return [t, p, q]

def dubins_RSL(alpha, beta, d):
    p_squared = (d * d) - 2 + (2 * math.cos(alpha - beta)) - (2 * d * (math.sin(alpha) + math.sin(beta)))
    if p_squared < 0: return None
    p = math.sqrt(p_squared)
    tmp2 = math.atan2((math.cos(alpha) + math.cos(beta)), (d - math.sin(alpha) - math.sin(beta))) - math.atan2(2.0, p)
    t = (alpha - tmp2) % (2 * math.pi)
    q = (beta - tmp2) % (2 * math.pi)
    return [t, p, q]

def dubins_RSR(alpha, beta, d):
    tmp0 = d - math.sin(alpha) + math.sin(beta)
    p_squared = 2 + (d * d) - (2 * math.cos(alpha - beta)) + (2 * d * (math.sin(beta) - math.sin(alpha)))
    if p_squared < 0: return None
    tmp1 = math.atan2((math.cos(alpha) - math.cos(beta)), tmp0)
    t = (alpha - tmp1) % (2 * math.pi)
    p = math.sqrt(p_squared)
    q = (-beta + tmp1) % (2 * math.pi)
    return [t, p, q]

def dubins_RLR(alpha, beta, d):
    tmp_rlr = (6.0 - d * d + 2 * math.cos(alpha - beta) + 2 * d * (math.sin(alpha) - math.sin(beta))) / 8.0
    if abs(tmp_rlr) > 1.0: return None
    p = (2 * math.pi - math.acos(tmp_rlr)) % (2 * math.pi)
    t = (alpha - math.atan2(math.cos(alpha) - math.cos(beta), d - math.sin(alpha) + math.sin(beta)) + (p / 2) % (2 * math.pi)) % (2 * math.pi)
    q = (alpha - beta - t + p % (2 * math.pi)) % (2 * math.pi)
    return [t, p, q]

def dubins_LRL(alpha, beta, d):
    tmp_lrl = (6.0 - d * d + 2 * math.cos(alpha - beta) + 2 * d * (-math.sin(alpha) + math.sin(beta))) / 8.0
    if abs(tmp_lrl) > 1.0: return None
    p = (2 * math.pi - math.acos(tmp_lrl)) % (2 * math.pi)
    t = (-alpha - math.atan2(math.cos(alpha) - math.cos(beta), d + math.sin(alpha) - math.sin(beta)) + p / 2) % (2 * math.pi)
    q = (beta % (2 * math.pi) - alpha - t + p % (2 * math.pi)) % (2 * math.pi)
    return [t, p, q]

# ================= Dubins 核心逻辑 =================
def dubins_core(p1, p2, r):
    """
    计算最短的 Dubins 曲线参数
    返回字典 param 包含最优路径类型、三段归一化长度等
    """
    param = {
        'p_init': p1,
        'seg_param': [0.0, 0.0, 0.0],
        'r': r,
        'type': -1,
        'flag': 0
    }
    
    if r <= 0:
        param['flag'] = -1
        return param
        
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    D = math.sqrt(dx**2 + dy**2)
    d = D / r  # 距离归一化
    
    theta = math.atan2(dy, dx) % (2 * math.pi)
    alpha = (p1[2] - theta) % (2 * math.pi)
    beta = (p2[2] - theta) % (2 * math.pi)
    
    best_word = -1
    best_cost = -1
    
    # 将6种求解器放入列表遍历
    solvers = [dubins_LSL, dubins_LSR, dubins_RSL, dubins_RSR, dubins_RLR, dubins_LRL]
    
    for i, solver in enumerate(solvers):
        res = solver(alpha, beta, d)
        if res is not None:
            cost = sum(res)
            if best_cost == -1 or cost < best_cost:
                best_word = i
                best_cost = cost
                param['seg_param'] = res
                param['type'] = i
                
    if best_word == -1:
        param['flag'] = -2  # NO PATH
        
    return param

def dubins_segment(seg_param, seg_init, seg_type):
    """基础运动学计算：根据圆弧/直线类型推演下一步坐标"""
    seg_end = [0.0, 0.0, 0.0]
    if seg_type == L_SEG:
        seg_end[0] = seg_init[0] + math.sin(seg_init[2] + seg_param) - math.sin(seg_init[2])
        seg_end[1] = seg_init[1] - math.cos(seg_init[2] + seg_param) + math.cos(seg_init[2])
        seg_end[2] = seg_init[2] + seg_param
    elif seg_type == R_SEG:
        seg_end[0] = seg_init[0] - math.sin(seg_init[2] - seg_param) + math.sin(seg_init[2])
        seg_end[1] = seg_init[1] + math.cos(seg_init[2] - seg_param) - math.cos(seg_init[2])
        seg_end[2] = seg_init[2] - seg_param
    elif seg_type == S_SEG:
        seg_end[0] = seg_init[0] + math.cos(seg_init[2]) * seg_param
        seg_end[1] = seg_init[1] + math.sin(seg_init[2]) * seg_param
        seg_end[2] = seg_init[2]
    return seg_end

def dubins_path_sample(param, t):
    """提取特定距离 t 处的坐标"""
    if t < 0 or t >= sum(param['seg_param']) * param['r'] or param['flag'] < 0:
        return None
        
    tprime = t / param['r']  # 归一化距离
    p_init = [0.0, 0.0, param['p_init'][2]]
    
    types = DIRDATA[param['type']]
    p1 = param['seg_param'][0]
    p2 = param['seg_param'][1]
    
    mid_pt1 = dubins_segment(p1, p_init, types[0])
    mid_pt2 = dubins_segment(p2, mid_pt1, types[1])
    
    if tprime < p1:
        end_pt = dubins_segment(tprime, p_init, types[0])
    elif tprime < (p1 + p2):
        end_pt = dubins_segment(tprime - p1, mid_pt1, types[1])
    else:
        end_pt = dubins_segment(tprime - p1 - p2, mid_pt2, types[2])
        
    # 逆变换：缩放回真实半径，平移回原起点
    end_pt[0] = end_pt[0] * param['r'] + param['p_init'][0]
    end_pt[1] = end_pt[1] * param['r'] + param['p_init'][1]
    end_pt[2] = end_pt[2] % (2 * math.pi)
    return end_pt

def dubins_curve(p1, p2, r, stepsize=0.0):
    """
    生成两点之间的 Dubins 航迹
    输入: p1/p2 (x, y, theta), r (转弯半径), stepsize (采样步长)
    返回: 离散航迹点列表 [x, y, theta], 航迹总长度
    """
    param = dubins_core(p1, p2, r)
    if param['flag'] < 0:
        return None, float('inf')
        
    total_length = sum(param['seg_param']) * param['r']
    
    # 如果没指定步长，自动用总长除以 1000 作为步长
    if stepsize <= 0:
        stepsize = total_length / 1000.0
        
    path = []
    x = 0.0
    while x <= total_length:
        pt = dubins_path_sample(param, x)
        if pt:
            path.append(pt)
        x += stepsize
        
    # 补齐终点
    path.append(p2)
    return np.array(path), total_length

# ================= 独立测试代码 =================
if __name__ == "__main__":
    # 测试场景：从 (0,0, 0度) 飞向 (1000, 1000, 180度)
    start_pose = [0.0, 0.0, math.radians(0)]
    end_pose = [15000.0, 15000.0, math.radians(120)]
    R_min = 2000.0  # 最小转弯半径
    
    print("正在计算最优 Dubins 转场路径...")
    path, length = dubins_curve(start_pose, end_pose, R_min, stepsize=5.0)
    
    if path is None:
        print("❌ 无法计算有效的 Dubins 路径")
        exit()
    
    # 绘制结果
    plt.figure(figsize=(8, 8))
    plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label=f'Dubins Path (L={length:.1f}m)')
    plt.plot(start_pose[0], start_pose[1], 'go', markersize=8, label='Start')
    plt.plot(end_pose[0], end_pose[1], 'ro', markersize=8, label='End')
    
    # 标出起终点的方向向量
    dx_s, dy_s = 150 * math.cos(start_pose[2]), 150 * math.sin(start_pose[2])
    plt.arrow(start_pose[0], start_pose[1], dx_s, dy_s, head_width=600, head_length=800, fc='g', ec='g')
    
    dx_e, dy_e = 150 * math.cos(end_pose[2]), 150 * math.sin(end_pose[2])
    plt.arrow(end_pose[0], end_pose[1], dx_e, dy_e, head_width=600, head_length=800, fc='r', ec='r')
    
    plt.title('Dubins Transfer Curve Calculation')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.axis('equal')
    plt.show()
    # save_path = 'dubins_test.png'
    # plt.savefig(save_path, dpi=300)
    # print(f"✅ Dubins 测试路径已保存为：{save_path}")