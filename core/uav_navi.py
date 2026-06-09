import numpy as np
import scipy.signal as signal
import matplotlib
import matplotlib.pyplot as plt

def uav_navi_traverse(position_target, total_time, time_step, velocity_UAV, erro_radius=150.0):
    """
    无人机自主导航航路点遍历程序,计算无人机从起点到目标点的导航路径
        
    参数:
    position_target: numpy array (N, 2)，目标航路点序列 [x, y]
    total_time: float, 总仿真时间 (s)
    time_step: float, 仿真步长 (s)
    velocity_UAV: float, 无人机巡航速度 (m/s)
    erro_radius: float, 允许的误差半径 (m)

    返回:
    position_UAV: numpy array (M, 2)，无人机轨迹坐标
    """
    
    # ==========================================================
    # 1. 初始条件:飞机气动参数及结构参数
    # ==========================================================
    zT = 0.0;
    m = 260.0; g = 9.8; G = m * g
    cA = 0.54148; b = 7.5; Sw = 3.965; alphaf=0.0
    
    Ix = 78.119; Iy = 204.428; Iz = 264.343
    Ixy=-0.; Ixz=-16.002; Iyz=0.;
    
    RHO = 1.22505
    V0 = 180.0 / 3.6  # 此处原代码固定了动压计算基准速度
    Q0 = V0 * V0 * RHO * Sw / 2.0
    
    #  静导数
    #  纵向静导数
    #  升力
    alpha0=-5.01; CLM=0.; CLa=0.1057; CL0=-CLa*alpha0; CLde=0.0057*57.3; CLaa=-0.4235;
    CL=G/Q0; CLa=57.3*CLa

    #  阻力
    CD0=0.0574; CDM=0.; A=0.0233; CD=CD0+A*CL*CL; CDa=0.00574*57.3

    # 横侧向静导数
    CY0 = 0;
    CYb = -0.01284 * 57.3
    CYr = 0.227433
    CYp = -0.051765
    CYdr = 0.004387 * 57.3
    CYda = -0.001329 * 57.3
    
    # 纵向动导数
    M0 = 0.0
    Cm0 = 0.2997
    CmM = 0.0
    Cma = -0.0306 * 57.3
    Cmaa = -3.1637
    Cmq = -12.1963

    # 横向动导数
    Cl0 = 0;
    Clb = -0.00199 * 57.3
    Clp = -0.619656
    Clr = 0.271793
    
    # 侧向动导数
    Cn0 = 0;
    Cnb = 0.00134 * 57.3
    Cnp = 0.119757
    Cnr = -0.09947
    
    #  纵向操纵导数
    Cmde=-0.0221*57.3
    
    # 横向与侧向操纵导数
    Clda = -0.002623 * 57.3
    Cldr = -0.000295 * 57.3
    Cnda = -0.000009 * 57.3
    Cndr = -0.001254 * 57.3
    
    # ================= 发动机拉力系数 =================
    TdT = 0.0
    T0 = 0.0
    Tv = 0.0
    
    Xv = (Q0 * (2 * CD + M0 * CDM) - V0 * Tv) / (m * V0)
    Xa = (Q0 * CDa - G) / (m * V0)
    Xth = g / V0
    XdT = -TdT / (m * V0)
    Zv = Q0 * (2 * CL + M0 * CLM) / (m * V0)
    Za = Q0 * CLa / (m * V0)
    Zaa = Q0 * CLaa / (m * V0)
    Zde = Q0 * CLde / (m * V0)
    Mv = -(Q0 * cA * (2 * Cm0 + M0 * CmM) + V0 * Tv * zT) / Iy
    Ma = -Q0 * cA * Cma / Iy
    Maa = -Q0 * cA * cA * Cmaa / (Iy * V0)
    Mq = -Q0 * cA * cA * Cmq / (Iy * V0)
    Mde = -Q0 * cA * Cmde / Iy
    MdT = -TdT * zT / Iy
    
    # ================= 纵向方程 (虽然当前轨迹不用，但依要求保留) =================
    AV = np.array([
        [-Xv, -Xa, 0, -Xth],
        [-Zv, -Za, 1, 0],
        [-(Mv - Maa * Zv), -(Ma - Maa * Za), -(Mq + Maa), 0],
        [0, 0, 1, 0]
    ])
    BV = np.array([
        [0, -XdT],
        [-Zde, 0],
        [-(Mde - Maa * Zde), -MdT],
        [0, 0]
    ])
    CV = np.eye(4)
    DV = np.zeros((4, 2))

    # 横侧向系统参数
    Yb = -Q0 * CYb / (m * V0)
    Yp = -Q0 * CYp / (m * V0)
    Yr = -Q0 * CYr / (m * V0)
    Yph = -g / V0
    Ydr = -Q0 * CYdr / (m * V0)
    Yda = -Q0 * CYda / (m * V0)    
    Lb = -Q0 * b * Clb / Ix
    Lp = -Q0 * b * b * Clp / (2 * Ix * V0)
    Lr = -Q0 * b * b * Clr / (2 * Ix * V0)
    Ldr = -Q0 * b * Cldr / Ix
    Lda = -Q0 * b * Clda / Ix    
    Nb = -Q0 * b * Cnb / Iz
    Np = -Q0 * b * b * Cnp / (2 * Iz * V0)
    Nr = -Q0 * b * b * Cnr / (2 * Iz * V0)
    Ndr = -Q0 * b * Cndr / Iz
    Nda = -Q0 * b * Cnda / Iz

    # 舵回路增益
    Kp = 0.2
    Kr = 0.6
    Kps = 0.2
    Kph = 1.0
    Kphr = 0.3

    # ==========================================================
    # 2. 状态空间方程构建 (连续转离散)
    # ==========================================================
    AS = np.array([
        [-Yb, -(Yp+Kp*Yda), -(1+Yr+Kr*Ydr), -(Yph+Kph*Yda), -(Kps*Yda+Kps*Kphr*Ydr)],
        [-Lb, -(Lp+Kp*Lda), -(Lr+Kr*Ldr), -Kph*Lda, -(Kps*Lda+Kps*Kphr*Ldr)],
        [-Nb, -(Np+Kp*Nda), -(Nr+Kr*Ndr), -Kph*Nda, -(Kps*Nda+Kps*Kphr*Ndr)],
        [0,   1,            0,             0,          0],
        [0,   0,            1,             0,          0]
    ])

    BS = np.array([
        [(Kps*Yda+Kphr*Kps*Ydr)],
        [(Kps*Lda+Kphr*Kps*Ldr)],
        [(Kps*Nda+Kphr*Kps*Ndr)],
        [0],
        [0]
    ])

    CS = np.array([[0, 0, 0, 0, 1]])
    DS = np.array([[0]])

    # 连续系统离散化 (等价于 c2d(sys_ss, sample_time))
    sys_d = signal.cont2discrete((AS, BS, CS, DS), time_step, method='zoh')
    Ad = sys_d[0]
    Bd = sys_d[1]

    # ==========================================================
    # 3. 导航控制初始化
    # ==========================================================
    p_1 = np.array(position_target)
    num_waypoints = len(p_1)
    
    kp_1 = 40.0; ki_1 = 0.0; kd_1 = 30.0
    # erro_radius = 60.0
    air_speed = velocity_UAV
    
    # 初始起点与第一个目标点 (注意：Python 索引从0开始)
    k_1 = 1  
    start_x = p_1[0, 0]
    start_y = p_1[0, 1]
    dest_x = p_1[k_1, 0]
    dest_y = p_1[k_1, 1]
    
    # 计算初始期望航向
    psi0_1 = np.arctan2(dest_y - start_y, dest_x - start_x) % (2 * np.pi)

    # 状态初始化
    x_vector2_1 = np.zeros((5, 1))
    x_vector2_1[4, 0] = psi0_1  # 核心修复：让飞机初始偏航角直接对准第一段航路
    
    # 记录轨迹的列表
    x_traj = [start_x]
    y_traj = [start_y]
    dist_error_list = [0.0]  # 记录横向误差用于计算积分和微分
    
    closest_dist_to_dest = float('inf')
    NN = int(total_time / time_step)

    sum_err = 0.0                  # 积分累加器，替代 sum()
    # ==========================================================
    # 4. 仿真主循环
    # ==========================================================
    for i in range(1, NN):
        # 航迹推算 (当前航向角为状态向量的第5个元素，即 x_vector2_1[4,0])
        current_heading = x_vector2_1[4, 0]
        curr_x = x_traj[-1] + air_speed * time_step * np.cos(current_heading)
        curr_y = y_traj[-1] + air_speed * time_step * np.sin(current_heading)
        
        # 利用向量点乘判断是否已经越过目标点所在的法平面 (防止飞过头后无限折返或飞离)
        vec_sd = np.array([dest_x - start_x, dest_y - start_y], dtype=float)
        vec_sc = np.array([curr_x - start_x, curr_y - start_y], dtype=float)
        seg_len2 = float(np.dot(vec_sd, vec_sd))
        seg_len = np.sqrt(seg_len2)
        along_progress = np.dot(vec_sc, vec_sd) / seg_len2 if seg_len2 > 1e-9 else 1.0
        dist_to_dest = np.hypot(curr_x - dest_x, curr_y - dest_y)
        cross_track = (
            abs(vec_sd[0] * vec_sc[1] - vec_sd[1] * vec_sc[0]) / seg_len
            if seg_len > 1e-9 else 0.0
        )
        closest_dist_to_dest = min(closest_dist_to_dest, dist_to_dest)
        reached_waypoint = dist_to_dest < erro_radius
        pass_radius = max(2.0 * erro_radius, 0.25 * seg_len)
        passed_waypoint = (
            along_progress >= 1.0
            and (cross_track <= pass_radius or closest_dist_to_dest <= 3.0 * erro_radius)
        )

        # 检查是否到达当前目标点
        if reached_waypoint or passed_waypoint:
            x_traj.append(curr_x)
            y_traj.append(curr_y)
            k_1 += 1
            
            # 如果到达了最后一个点，跳出循环
            if k_1 >= num_waypoints:
                break

            start_x = curr_x
            start_y = curr_y

            dest_x = p_1[k_1, 0]
            dest_y = p_1[k_1, 1]
            
            # 更新基准航线角
            psi0_1 = np.arctan2(dest_y - start_y, dest_x - start_x)

            sum_err = 0.0            # 清空积分器，防止将上一段航线的累积误差带入新航线引发超调
            dist_error_list = [0.0]  # 重置误差列表，保持微分项的正确性

        # 计算偏航距离误差 (点到直线的距离)
            closest_dist_to_dest = float('inf')
            continue

        num1 = abs((dest_y - start_y)*curr_x - (dest_x - start_x)*curr_y + start_y*dest_x - dest_y*start_x)
        den1 = np.sqrt((dest_y - start_y)**2 + (dest_x - start_x)**2)
        dist_err = num1 / den1 if den1 != 0 else 0.0
        
        # 判断在直线的左侧还是右侧
        f_1 = (curr_x - start_x)*(dest_y - start_y) - (dest_x - start_x)*(curr_y - start_y)
        if f_1 < 0:
            dist_err = -dist_err
            
        # PID 导航控制器设计
        # 注意：此处用实际物理步长 time_step 替代了原代码硬编码的 0.02
        sum_err += dist_err * time_step
        diff_err = (dist_err - dist_error_list[-1]) / time_step
        dist_error_list.append(dist_err)

        # PID 计算
        increase1 = (kp_1/57.3)*dist_err + (ki_1/57.3)*sum_err + (kd_1/57.3)*diff_err
        
        # 限幅
        increase1 = np.clip(increase1, -np.pi/2, np.pi/2)
        expect_heading = psi0_1 + increase1
        
        # 核心修复：计算真正的最短夹角误差，消除多圈螺旋积累的绕圈阶跃
        err_angle = np.arctan2(np.sin(expect_heading - current_heading), np.cos(expect_heading - current_heading))
        # 用当前真实状态加上最短误差，反推给状态空间模型，保证输入 u 与状态 x5 始终平滑连续
        expect_heading = current_heading + err_angle

        # 状态方程更新
        x_vector1_1 = x_vector2_1
        x_vector2_1 = Ad @ x_vector1_1 + Bd * expect_heading
        
        # 记录位置
        x_traj.append(curr_x)
        y_traj.append(curr_y)

    position_UAV = np.column_stack((x_traj, y_traj))
    return position_UAV

# ==========================================================
# 测试与绘图模块
# ==========================================================
if __name__ == "__main__":
    # 定义目标航路点 (以 250x250米 区域测试)
    target_waypoints = np.array([
        [0, 0],
        [1000, 1000],
        [2000, 500],
        [3000, 0],
        [4500, 500],
        [5000, 1000]
    ])
    
    # 仿真参数
    TOTAL_TIME = 150.0   # 仿真总时长 300s
    DT = 0.02            # 控制与仿真步长 0.02s
    UAV_V = 180.0 / 3.6  # 速度 50 m/s
    
    print("开始进行无人机气动仿真...")
    traj = uav_navi_traverse(target_waypoints, TOTAL_TIME, DT, UAV_V)
    print("仿真完成！正在绘制航迹...")
    
    # 还原并优化原代码中被注释的绘图逻辑
    plt.figure(figsize=(10, 8), facecolor='white')
    
    # 画出航路点与期望直线段
    plt.plot(target_waypoints[:, 0], target_waypoints[:, 1], 'ro-', 
             markersize=8, markerfacecolor='g', label='Waypoints & Ref Path', linestyle='-.', linewidth=2)
    
    # 画出无人机实际飞行的气动平滑轨迹
    plt.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2.5, label='UAV Actual Trajectory')
    
    plt.title('UAV Navigation Dynamics Model', fontsize=14)
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.axis('equal')
    plt.show()

    # save_file = 'uav_trajectory_test.png'
    # plt.savefig(save_file, dpi=300)
    # print(f"✅ 轨迹图已保存至: {save_file}")
