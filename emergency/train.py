"""
DQN突发事件临机决策 — 训练脚本 (多步MDP)
使用PyTorch + 优先经验回放训练Dueling DQN
5个离散动作: 0=放弃, 1-4=指派给UAV1-4
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from core import heatmap as hm
from core import ga_allocator as ga
from emergency.agent import (
    EmergencyDQN, PrioritizedReplayBuffer,
    get_valid_actions, select_action, NUM_UAVS, NUM_ACTIONS
)
from emergency.simulator import EmergencySimulator
from emergency.utils import (
    EmergencyFSM, build_state_vector, get_affected_targets,
    apply_decision, compute_route_distance_km,
    STATE_DIM, SPAN_KM, MAX_RANGE_KM, MAX_FLIGHT_TIME_S, GLOBAL_C_DROP,
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

ORACLE_COMPUTE_INTERVAL = 50  # 每N回合计算一次GA oracle (降低频率加速训练)


# ==================== 检查点存取 ====================

def _make_ckpt_path(model_save_path, episode):
    model_name = os.path.splitext(os.path.basename(model_save_path))[0]
    return os.path.join('checkpoints', f'{model_name}_ckpt_ep{episode:06d}.pt')


def find_latest_ckpt(model_save_path):
    os.makedirs('checkpoints', exist_ok=True)
    model_name = os.path.splitext(os.path.basename(model_save_path))[0]
    pattern = f'{model_name}_ckpt_ep'
    candidates = [f for f in os.listdir('checkpoints')
                  if f.startswith(pattern) and f.endswith('.pt')
                  and '_buffer' not in f]
    if not candidates:
        return None
    candidates.sort(key=lambda x: int(x.replace(pattern, '').replace('.pt', '')))
    return os.path.join('checkpoints', candidates[-1])


def save_checkpoint(ckpt_path, online_net, target_net, optimizer, replay_buffer,
                    epsilon, step_count, start_episode, rewards, losses):
    torch.save({
        'online_net': online_net.state_dict(),
        'target_net': target_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epsilon': epsilon,
        'step_count': step_count,
        'start_episode': start_episode,
        'rewards': rewards,
        'losses': losses,
    }, ckpt_path)
    buf_path = ckpt_path.replace('.pt', '_buffer.npz')
    buf = replay_buffer
    np.savez_compressed(buf_path,
                        state=buf.data['state'][:buf.size],
                        action=buf.data['action'][:buf.size],
                        reward=buf.data['reward'][:buf.size],
                        next_state=buf.data['next_state'][:buf.size],
                        done=buf.data['done'][:buf.size],
                        size=buf.size,
                        next_idx=buf.next_idx,
                        max_priority=buf.max_priority,
                        priority_sum=np.array(buf.priority_sum),
                        priority_min=np.array(buf.priority_min))


def load_checkpoint(ckpt_path, device, buffer_capacity, state_dim=STATE_DIM_V2, alpha=0.6):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    online_net = EmergencyDQN(state_dim=state_dim).to(device)
    online_net.load_state_dict(ckpt['online_net'])

    target_net = EmergencyDQN(state_dim=state_dim).to(device)
    target_net.load_state_dict(ckpt['target_net'])
    target_net.eval()

    optimizer = torch.optim.Adam(online_net.parameters(), lr=1e-4)
    optimizer.load_state_dict(ckpt['optimizer'])

    replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity, alpha=alpha,
                                             state_dim=state_dim)
    buf_path = ckpt_path.replace('.pt', '_buffer.npz')
    if os.path.exists(buf_path):
        buf_data = np.load(buf_path)
        saved_size = int(buf_data['size'])
        replay_buffer.data['state'][:saved_size] = buf_data['state']
        replay_buffer.data['action'][:saved_size] = buf_data['action']
        replay_buffer.data['reward'][:saved_size] = buf_data['reward']
        replay_buffer.data['next_state'][:saved_size] = buf_data['next_state']
        replay_buffer.data['done'][:saved_size] = buf_data['done']
        replay_buffer.size = saved_size
        replay_buffer.next_idx = int(buf_data['next_idx'])
        replay_buffer.max_priority = float(buf_data['max_priority'])
        replay_buffer.priority_sum = list(buf_data['priority_sum'])
        replay_buffer.priority_min = list(buf_data['priority_min'])
        print(f"  回放缓存已恢复: {saved_size} 条样本")
    else:
        print(f"  警告: 未找到回放缓存文件 {buf_path}, 使用空缓存")

    return (online_net, target_net, optimizer, replay_buffer,
            ckpt['epsilon'], ckpt['step_count'], ckpt['start_episode'],
            list(ckpt['rewards']), list(ckpt['losses']))


# ==================== 训练主函数 ====================

def train(num_episodes=500000, batch_size=128, lr=1e-4, gamma=0.95,
          target_update_freq=1000, buffer_capacity=500000,
          model_save_path='models/emergency_dqn_model.pt',
          log_interval=100, save_interval=5000,
          resume_path=None, oracle_interval=ORACLE_COMPUTE_INTERVAL):
    """训练DQN智能体 (多步MDP)

    Args:
        num_episodes: 训练总回合数
        batch_size: 小批量大小
        lr: 学习率
        gamma: 折扣因子 (有限视野MDP用0.95)
        target_update_freq: target网络更新间隔
        buffer_capacity: 回放缓存容量
        model_save_path: 模型保存路径
        log_interval: 日志输出间隔
        save_interval: 模型保存间隔
        resume_path: 断点续训检查点路径
        oracle_interval: GA oracle计算间隔
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"动作空间: 5维 (0=放弃, 1-4=指派给UAV1-4)")
    print(f"状态空间: {STATE_DIM}维 (逐目标聚焦)")

    # ---- 1. 初始化环境 ----
    print("初始化仿真环境...")
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
    print("仿真环境初始化完成")

    # ---- 2. 运行初始GA获取基线 ----
    print("运行初始GA获取基线分配...")
    ga_targets = [{'id': h['id'], 'pos': h['center_km'],
                   'weight': h['weight'], 'radius_km': h['radius_km']}
                  for h in UNIFIED_HOTSPOTS]
    ga.init_ga_env(ga_targets, UAV_BASE_KM, NUM_UAVS,
                   prob_grid=prob_grid, X_km=X_km, Y_km=Y_km,
                   spiral_cfg=spiral_cfg, weights=(0.5, 0.3, 0.4), d_max=MAX_FLIGHT_TIME_S,  c_drop=GLOBAL_C_DROP)

    best_chrom, best_cost, history, norm_factors = ga.run_ga(
        pop_size=200, generations=300, patience=100
    )
    _, raw_routes, baseline_times = ga.evaluate_chromosome(best_chrom, norm_factors)
    
    # 🌟 新增：截断分离真实UAV航线和垃圾桶
    baseline_routes = raw_routes[:NUM_UAVS]
    dropped_initial = raw_routes[NUM_UAVS] if len(raw_routes) > NUM_UAVS else []
    
    print(f"GA基线分配: {baseline_routes}")
    if dropped_initial:
        print(f"⚠️ 初始基线规划中因超限被放弃的目标: {dropped_initial}")

    # ---- 3. 生成参考路径 ----
    from core.spiral_search import spiral_search_arc_exact
    from core.dubins_planner import dubins_curve
    import math

    ref_paths = []
    route_boundaries_list = []  # 每个UAV每个目标的累积路径距离
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

    print("参考路径生成完成")

    # ---- 4. 初始化或恢复训练状态 ----
    start_episode = 0
    episode_rewards = []
    episode_losses = []

    loaded_from_ckpt = False
    if resume_path is None:
        resume_path = 'auto'

    if resume_path == 'auto':
        found = find_latest_ckpt(model_save_path)
        if found:
            print(f"自动找到最新检查点: {found}")
            resume_path = found
        else:
            print("未找到已有检查点，从头训练")
            resume_path = None
    elif resume_path == 'none':
        resume_path = None

    if resume_path and os.path.exists(resume_path):
        print(f"从检查点恢复训练: {resume_path}")
        (online_net, target_net, optimizer, replay_buffer,
         epsilon, step_count, start_episode,
         episode_rewards, episode_losses) = load_checkpoint(
            resume_path, device, buffer_capacity, state_dim=STATE_DIM_V2)
        print(f"  已恢复: {start_episode} 回合, epsilon={epsilon:.4f}, "
              f"step_count={step_count}")
        loaded_from_ckpt = True

    if not loaded_from_ckpt:
        online_net = EmergencyDQN(state_dim=STATE_DIM_V2, num_actions=N_MAX).to(device)
        target_net = EmergencyDQN(state_dim=STATE_DIM_V2, num_actions=N_MAX).to(device)
        target_net.load_state_dict(online_net.state_dict())
        target_net.eval()

        optimizer = torch.optim.Adam(online_net.parameters(), lr=lr)
        replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity, alpha=0.6, state_dim=STATE_DIM_V2
        )
        epsilon = 1.0
        step_count = 0

    epsilon_min = 0.05
    # 指数衰减：0.9998^15000 ≈ 0.05，断点续训兼容（基于当前epsilon值衰减，不依赖全局回合数）
    epsilon_decay = 0.9999
    beta_start = 0.4
    beta_end = 1.0

    remaining_episodes = num_episodes - start_episode
    if remaining_episodes <= 0:
        print(f"训练已完成: 当前已训练 {start_episode} >= {num_episodes} 回合")
        return online_net, episode_rewards, episode_losses

    print(f"训练: {start_episode + 1} -> {num_episodes} 回合 "
          f"(新增 {remaining_episodes}), batch={batch_size}, lr={lr}")
    print(f"epsilon: {epsilon:.4f} -> {epsilon_min:.3f}, gamma={gamma}")

    best_avg_reward = -float('inf')
    oracle_cost_ema = None  # GA oracle代价的指数移动平均

    # CSV增量日志 (每回合追加，崩溃不丢失)
    csv_log_path = model_save_path.replace('.pt', '_log.csv')
    if not loaded_from_ckpt or not os.path.exists(csv_log_path):
        os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)
        with open(csv_log_path, 'w') as f:
            f.write('episode,reward,loss,epsilon,steps,J_max,J_sum,buffer_size,'
                    'range_violations,max_range_ratio,abandoned,'
                    'abandon_avg_w,assign_avg_w,coverage\n')
    episode_J_max = []
    episode_J_sum = []
    current_loss = 0.0

    # id→hotspot 查找表 (用于最近邻重排序)
    id_to_hotspot_lookup = {h['id']: h for h in UNIFIED_HOTSPOTS}

    pbar = tqdm(range(remaining_episodes), desc="训练", unit="ep",
                ncols=100,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    pbar.set_postfix({'reward': '---', 'loss': '---', 'eps': f'{epsilon:.3f}',
                      'buf': f'{replay_buffer.size}'})

    try:
        for ep in pbar:
            current_episode = start_episode + ep + 1

            # ========== 生成突发事件 ==========
            emergency = sim.generate_random_emergency(baseline_routes, ref_paths)

            uav_positions, _, completed, remaining, consumed_ranges, active_target_ids, progress_kms = \
                sim.simulate_until_emergency(ref_paths, emergency,
                                             baseline_routes, route_boundaries_list)

            # S1: 已搜索目标不再位移
            if emergency['type'] == 'S1':
                emergency['shifts'] = [s for s in emergency.get('shifts', [])
                                       if s['id'] not in completed]
                emergency['affected_targets'] = [tid for tid in emergency.get('affected_targets', [])
                                                  if tid not in completed]

            modified_hotspots, active_uavs, _ = sim.apply_emergency(emergency)

            # 更新剩余目标集 (S2添加新目标)
            if emergency['type'] == 'S2':
                for nt in emergency.get('new_targets', []):
                    remaining.add(nt['id'])

            # 获取受影响目标列表 (按权重降序)，排除已完成目标
            affected = get_affected_targets(emergency, baseline_routes, modified_hotspots)
            affected = [t for t in affected if t['id'] not in completed]

            if not affected:
                continue  # 无未完成的受影响目标，跳过

            # ========== 初始化：从基线路由中移除已完成目标 ==========
            current_routes = [[tid for tid in r if tid not in completed]
                              for r in baseline_routes]

            # S4: 故障UAV的路由清空，其目标进入affected列表
            if emergency['type'] == 'S4':
                failed = emergency['failed_uav']
                current_routes[failed] = []

            # 用于记录本episode的分配决策
            fsm = EmergencyFSM()
            fsm.trigger_emergency(emergency)
            if not fsm.is_emergency():
                continue  # FSM 拒绝触发，跳过本回合

            # ========== V2 轮询MDP决策循环 ==========
            episode_cost = None
            state = sim.reset_v2(uav_positions, consumed_ranges, modified_hotspots,
                                 active_uavs=active_uavs, completed=completed,
                                 active_target_ids=active_target_ids, progress_kms=progress_kms,
                                 emergency=emergency)
            done = False
            episode_step_reward = 0.0

            while not done:
                valid_mask = get_valid_actions_v2(sim.current_uav_idx, sim.uav_states, sim.targets,
                                                  locked_target_idx=sim.locked_target_idxs[sim.current_uav_idx],
                                                  emergency=emergency)
                if not valid_mask.any():
                    break
                with torch.no_grad():
                    q_vals = online_net(torch.FloatTensor(state).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
                action = select_action(q_vals, valid_mask, epsilon)
                next_state, reward, done, _ = sim.step_v2(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_step_reward += reward

            episode_rewards.append(episode_step_reward)

            # ========== 训练一步 ==========
            if replay_buffer.size >= batch_size:
                # PER beta 绝对退火：前300000回合从0.4→1.0，兼容断点续训
                beta_anneal_episodes = 300000.0
                progress = min(1.0, current_episode / beta_anneal_episodes)
                beta = beta_start + (beta_end - beta_start) * progress
                samples = replay_buffer.sample(batch_size, beta)

                states_batch = torch.FloatTensor(samples['state']).to(device)
                actions_batch = torch.LongTensor(samples['action']).to(device)
                rewards_batch = torch.FloatTensor(samples['reward']).to(device)
                dones_batch = torch.FloatTensor(samples['done']).to(device)
                weights_batch = torch.FloatTensor(samples['weights']).to(device)

                q_values = online_net(states_batch)
                q_action = q_values.gather(1, actions_batch.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_states_batch = torch.FloatTensor(samples['next_state']).to(device)
                    next_q = target_net(next_states_batch)
                    max_next_q = next_q.max(dim=1)[0]
                    target_q = rewards_batch + gamma * max_next_q * (1 - dones_batch)

                td_errors = target_q - q_action
                loss = (weights_batch * nn.functional.smooth_l1_loss(
                    q_action, target_q, reduction='none'
                )).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=1.0)
                optimizer.step()

                replay_buffer.update_priorities(
                    samples['indexes'],
                    np.abs(td_errors.detach().cpu().numpy()) + 1e-6
                )

                episode_losses.append(loss.item())
                current_loss = loss.item()
                step_count += 1

            # ========== 更新target网络 ==========
            if step_count > 0 and step_count % target_update_freq == 0:
                target_net.load_state_dict(online_net.state_dict())

            # ========== epsilon衰减 ==========
            # 指数衰减：基于当前值衰减，断点续训兼容
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            # ========== CSV增量日志 ==========
            episode_J_max.append(0.0)
            episode_J_sum.append(0.0)

            n_targets = len(sim.targets)
            n_visited = sum(1 for t in sim.targets if t['mask'])
            coverage = n_visited / max(n_targets, 1)
            unvisited_w = sum(t['weight'] for t in sim.targets if not t['mask'])

            with open(csv_log_path, 'a') as f:
                f.write(f'{current_episode},{episode_rewards[-1]:.4f},{current_loss:.6f},'
                        f'{epsilon:.4f},{n_targets},'
                        f'0.0,0.0,'
                        f'{replay_buffer.size},0,0.0000,'
                        f'{n_targets - n_visited},{unvisited_w:.3f},{coverage:.3f},{coverage:.3f}\n')

            # ========== 进度条更新 ==========
            if len(episode_rewards) >= 10:
                recent_reward = np.mean(episode_rewards[-10:])
                recent_loss = np.mean(episode_losses[-10:]) if episode_losses else 0.0
            elif episode_rewards:
                recent_reward = np.mean(episode_rewards)
                recent_loss = np.mean(episode_losses) if episode_losses else 0.0
            else:
                recent_reward = 0.0
                recent_loss = 0.0
            pbar.set_postfix({
                'reward': f'{recent_reward:+.3f}',
                'loss': f'{recent_loss:.4f}',
                'eps': f'{epsilon:.3f}',
                'buf': f'{replay_buffer.size}'
            })

            # ========== 日志输出 ==========
            if current_episode % log_interval == 0:
                avg_reward = np.mean(episode_rewards[-log_interval:])
                avg_loss = np.mean(episode_losses[-log_interval:]) if episode_losses else 0.0
                tqdm.write(f"Ep {current_episode}/{num_episodes} | "
                           f"avg_reward: {avg_reward:+.4f} | loss: {avg_loss:.6f} | "
                           f"eps: {epsilon:.3f} | targets: {n_targets}")

                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    torch.save(online_net.state_dict(),
                               model_save_path.replace('.pt', '_best.pt'))
                    tqdm.write(f"  -> 保存最佳模型 (avg_reward={best_avg_reward:+.4f})")

            # ========== 定期保存检查点 + 模型 ==========
            if current_episode % save_interval == 0:
                ckpt_path = _make_ckpt_path(model_save_path, current_episode)
                save_checkpoint(ckpt_path, online_net, target_net, optimizer,
                                replay_buffer, epsilon, step_count,
                                current_episode, episode_rewards, episode_losses)
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save(online_net.state_dict(), model_save_path)
                tqdm.write(f"  -> 检查点+模型已保存: ep{current_episode}")

    except KeyboardInterrupt:
        save_ep = current_episode if 'current_episode' in dir() else start_episode
        ckpt_path = _make_ckpt_path(model_save_path, save_ep)
        tqdm.write(f"\n训练被中断 (Ctrl+C), 正在保存检查点...")
        save_checkpoint(ckpt_path, online_net, target_net, optimizer,
                        replay_buffer, epsilon, step_count,
                        save_ep, episode_rewards, episode_losses)
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(online_net.state_dict(), model_save_path)
        tqdm.write(f"  检查点已保存: {ckpt_path}")
        tqdm.write(f"  模型已保存: {model_save_path}")
        tqdm.write(f"  下次运行自动恢复，或指定: --resume {ckpt_path}")
        return online_net, episode_rewards, episode_losses

    pbar.close()

    # ---- 最终保存 ----
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    final_ckpt = _make_ckpt_path(model_save_path, num_episodes)
    save_checkpoint(final_ckpt, online_net, target_net, optimizer,
                    replay_buffer, epsilon, step_count,
                    num_episodes, episode_rewards, episode_losses)
    torch.save(online_net.state_dict(), model_save_path)
    tqdm.write(f"\n训练完成! 模型: {model_save_path}, 检查点: {final_ckpt}")

    os.makedirs('models', exist_ok=True)
    np.savez(os.path.join('models', 'training_history.npz'),
             rewards=episode_rewards,
             losses=episode_losses,
             J_max=episode_J_max,
             J_sum=episode_J_sum)
    print(f"训练历史已保存: models/training_history.npz")
    print(f"CSV日志已保存: {csv_log_path}")

    return online_net, episode_rewards, episode_losses


# ==================== CLI入口 ====================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQN突发事件临机决策训练 (5动作, 多步MDP)')
    parser.add_argument('--episodes', type=int, default=500000,
                        help='训练回合数 (默认: 500000)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='小批量大小 (默认: 64)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率 (默认: 1e-4)')
    parser.add_argument('--model-path', type=str, default='models/emergency_dqn_model.pt',
                        help='模型保存路径')
    parser.add_argument('--buffer-capacity', type=int, default=500000,
                        help='回放缓存容量 (默认: 500000)')
    parser.add_argument('--target-update', type=int, default=1000,
                        help='Target网络更新间隔 (默认: 1000)')
    parser.add_argument('--resume', type=str, default='auto',
                        help='恢复训练: auto(自动最新), none(从头), 或指定ckpt路径')
    parser.add_argument('--save-interval', type=int, default=5000,
                        help='检查点保存间隔 (默认: 5000)')
    parser.add_argument('--oracle-interval', type=int, default=ORACLE_COMPUTE_INTERVAL,
                        help=f'GA oracle计算间隔 (默认: {ORACLE_COMPUTE_INTERVAL}, 0=关闭)')
    args = parser.parse_args()

    model, rewards, losses = train(
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        lr=args.lr,
        model_save_path=args.model_path,
        buffer_capacity=args.buffer_capacity,
        target_update_freq=args.target_update,
        resume_path=args.resume,
        save_interval=args.save_interval,
        oracle_interval=args.oracle_interval,
    )
