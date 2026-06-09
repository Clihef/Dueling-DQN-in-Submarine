import math
import numpy as np
import matplotlib
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
# ================= 6种类型示意图（完美位姿精调版） =================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import math
    import numpy as np
    
    def force_dubins_type(p1, p2, r, target_type):
        """强制计算特定类型的 Dubins 曲线"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        D = math.sqrt(dx**2 + dy**2)
        d = D / r
        
        theta = math.atan2(dy, dx) % (2 * math.pi)
        alpha = (p1[2] - theta) % (2 * math.pi)
        beta = (p2[2] - theta) % (2 * math.pi)
        
        solvers = [dubins_LSL, dubins_LSR, dubins_RSL, dubins_RSR, dubins_RLR, dubins_LRL]
        res = solvers[target_type](alpha, beta, d)
        
        if res is None:
            return None, float('inf'), None
            
        param = {
            'p_init': p1,
            'seg_param': res,
            'r': r,
            'type': target_type,
            'flag': 0
        }
        
        total_length = sum(res) * r
        stepsize = total_length / 500.0
        
        path = []
        x = 0.0
        while x <= total_length:
            pt = dubins_path_sample(param, x)
            if pt: path.append(pt)
            x += stepsize
        path.append(p2)
        return np.array(path), total_length, param

    # 准备画布: 2行3列
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    
    type_names = ["LSL (0)", "LSR (1)", "RSL (2)", "RSR (3)", "RLR (4)", "LRL (5)"]
    R_min = 1.0 
    
    # --- 核心修改：为每种类型单独定制最完美的起终点位姿 [x, y, theta] ---
    # 目的：避免绕大圈，让首尾转弯圆弧的角度都保持较小，曲线更自然
    custom_poses = [
        # 0: LSL - 起点偏右下，左转向左上直线，再左转朝上
        ([0.0, 0.0, math.radians(-30)], [4.0, 3.0, math.radians(150)]),
        
        # 1: LSR - 起点朝右下，左转拉平，直线，右转朝右下
        ([0.0, 0.0, math.radians(-60)], [5.0, 0.0, math.radians(-90)]),
        
        # 2: RSL - 起点朝右上，右转拉平，直线，左转朝右上
        ([0.0, 0.0, math.radians(30)],  [5.0, 0.0, math.radians(60)]),
        
        # 3: RSR - 与LSL镜像对称
        ([0.0, 0.0, math.radians(30)],  [4.0, -3.0, math.radians(-90)]),
        
        # 4: RLR - 三圆相切，必须距离近
        ([0.0, 0.0, math.radians(45)],  [5.0, 0.0, math.radians(-45)]),
        
        # 5: LRL - 三圆相切，与RLR镜像
        ([0.0, 0.0, math.radians(-45)], [5.0, -1.0, math.radians(45)])
    ]
    
    # 科研配色方案
    color_path = '#1F618D'   
    color_start = '#229954'  
    color_end = '#CB4335'    
    color_circle = '#D5D8DC' 
    color_ref = '#7F8C8D'    
    
    for i in range(6):
        row = i // 3
        col = i % 3
        ax = axs[row, col]
        
        start_p, end_p = custom_poses[i]
        path, length, param = force_dubins_type(start_p, end_p, R_min, i)
        
        if path is not None and param is not None:
            # 计算圆心
            types = DIRDATA[param['type']]
            p_init_norm = [0.0, 0.0, param['p_init'][2]]
            seg_p1 = param['seg_param'][0]
            seg_p2 = param['seg_param'][1]
            
            pose1_norm = dubins_segment(seg_p1, p_init_norm, types[0])
            pose2_norm = dubins_segment(seg_p2, pose1_norm, types[1])
            
            def get_real_pose(norm_pose, param):
                return [
                    norm_pose[0] * param['r'] + param['p_init'][0],
                    norm_pose[1] * param['r'] + param['p_init'][1],
                    norm_pose[2] % (2 * math.pi)
                ]
                
            pose0 = param['p_init']
            pose1 = get_real_pose(pose1_norm, param)
            pose2 = get_real_pose(pose2_norm, param)
            
            def get_center(pose, turn_type, r):
                if turn_type == L_SEG:
                    return (pose[0] - r * math.sin(pose[2]), pose[1] + r * math.cos(pose[2]))
                elif turn_type == R_SEG:
                    return (pose[0] + r * math.sin(pose[2]), pose[1] - r * math.cos(pose[2]))
                return None

            centers = []
            if types[0] in [L_SEG, R_SEG]: centers.append(get_center(pose0, types[0], R_min))
            if types[1] in [L_SEG, R_SEG]: centers.append(get_center(pose1, types[1], R_min))
            if types[2] in [L_SEG, R_SEG]: centers.append(get_center(pose2, types[2], R_min))
            
            # 绘制虚线圆
            for c in centers:
                if c:
                    circle = patches.Circle(c, R_min, color=color_circle, fill=False, linestyle='--', linewidth=1.5, zorder=1)
                    ax.add_patch(circle)

            # 绘制实际轨迹
            ax.plot(path[:, 0], path[:, 1], color=color_path, linestyle='-', linewidth=3.5, zorder=3)
            
            # 绘制起终点方向箭头、参考线和详细标注
            arrow_len = 1.0
            ref_len = 1.3
            
            points_info = [
                (start_p, 'start', r'$\psi_s$', color_start),
                (end_p, 'end', r'$\psi_e$', color_end)
            ]
            
            for p, label_text, angle_text, color in points_info:
                # 垂直正北基准线
                ax.plot([p[0], p[0]], [p[1] - ref_len*0.2, p[1] + ref_len], color=color_ref, linestyle='-', linewidth=1, alpha=0.8, zorder=2)
                
                # 飞机朝向矢量箭头
                ax.annotate('', xy=(p[0] + arrow_len*math.cos(p[2]), p[1] + arrow_len*math.sin(p[2])), 
                            xytext=(p[0], p[1]), 
                            arrowprops=dict(arrowstyle="->", color=color, lw=2.5), zorder=4)
                
                # 绘制角度圆弧
                theta_deg = math.degrees(p[2]) % 360
                t1, t2 = sorted([90, theta_deg])
                # 处理跨越 360度/0度 的显示逻辑
                if t2 - t1 > 180:
                    t1, t2 = t2, t1 + 360
                    
                arc = patches.Arc((p[0], p[1]), 0.8, 0.8, theta1=t1, theta2=t2, color=color_ref, linewidth=1.2, zorder=2)
                ax.add_patch(arc)
                
                # 标注角度符号
                mid_angle = math.radians((t1 + t2) / 2)
                ax.text(p[0] + 0.6 * math.cos(mid_angle), p[1] + 0.6 * math.sin(mid_angle), 
                        angle_text, fontsize=14, color='#333333', ha='center', va='center', zorder=5)
                
                # 标注 start/end
                ax.text(p[0], p[1] - 0.5, label_text, fontsize=12, color=color, fontweight='bold', ha='center', va='center', zorder=5)
                ax.plot(p[0], p[1], 'o', color=color, markersize=7, zorder=5)

            ax.set_title(type_names[i], pad=4, color='#333333', fontsize=12, fontweight='bold')
            ax.axis('equal')
            ax.axis('off')
            ax.margins(0.06) # 增加边距，防止字被切掉
        else:
            ax.set_title(type_names[i], pad=4, color='#333333', fontsize=12, fontweight='bold')
            ax.axis('off')

    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
    plt.show()
