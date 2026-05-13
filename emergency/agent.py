"""
基于Dueling DQN的突发事件临机决策智能体
5个离散动作: 0=放弃, 1-4=指派给UAV1-4
参考: Reference/rl/dqn/model.py (PyTorch Dueling DQN)
      Reference/rl/dqn/replay_buffer.py (Prioritized Experience Replay)
"""
import random
import numpy as np
import torch
import torch.nn as nn


# ==================== 常量 ====================

NUM_UAVS = 4
NUM_ACTIONS = 5  # 0=放弃, 1-4=指派给UAV1-4


# ==================== Dueling DQN 网络 ====================

class EmergencyDQN(nn.Module):
    """Dueling DQN: 输入38维状态向量，输出5个Q值"""

    def __init__(self, state_dim=96, num_actions=20, hidden_dim=128):
        super().__init__()
        self.num_actions = num_actions

        # 共享特征提取器
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 状态价值头 V(s)
        self.state_value = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # 动作优势头 A(s, a)
        self.action_value = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state):
        """
        Args:
            state: (batch, state_dim) 状态向量
        Returns:
            q_values: (batch, 5) 5个动作的Q值 [abandon, UAV1, UAV2, UAV3, UAV4]
        """
        features = self.feature(state)
        v = self.state_value(features)
        a = self.action_value(features)
        q = v + (a - a.mean(dim=-1, keepdim=True))
        return q


# ==================== 动作选择 ====================

def get_valid_actions(active_uavs_mask, _emergency_type=None):
    """返回当前有效的动作索引列表

    Args:
        active_uavs_mask: bool[4], True=活跃, False=故障/不可用
        emergency_type: str, 事件类型（保留兼容，S4屏蔽故障UAV）

    Returns:
        valid_actions: list of int, 有效动作索引，0=放弃, 1-4=指派给UAV1-4
    """
    valid = [0]  # 所有场景下放弃均可用（航程约束下需战略放弃）
    # UAV1-4对应动作索引1-4
    for k, active in enumerate(active_uavs_mask):
        if active:
            valid.append(k + 1)  # UAV1→1, UAV2→2, ...
    return valid


def select_action(q_values, valid_mask, epsilon=0.0):
    """ε-greedy动作选择，带动作屏蔽

    Args:
        q_values: (num_actions,) numpy array, Q值
        valid_mask: (num_actions,) bool array, 合法动作掩码
        epsilon: 探索概率

    Returns:
        action: int, 动作索引
    """
    if np.random.random() < epsilon:
        return int(np.random.choice(np.where(valid_mask)[0]))
    masked_q = np.where(valid_mask, q_values, -np.inf)
    masked_q += np.random.uniform(-1e-6, 1e-6, size=masked_q.shape)
    return int(np.argmax(masked_q))


# ==================== 优先经验回放缓存 ====================

class PrioritizedReplayBuffer:
    """基于二叉线段树的优先经验回放 (PER)"""

    def __init__(self, capacity, alpha=0.6, state_dim=96):
        self.capacity = capacity
        self.alpha = alpha

        # 二叉线段树 (1-indexed)
        self.priority_sum = [0.0] * (2 * capacity)
        self.priority_min = [float('inf')] * (2 * capacity)
        self.max_priority = 1.0

        # 数据存储
        self.data = {
            'state': np.zeros((capacity, state_dim), dtype=np.float32),
            'action': np.zeros(capacity, dtype=np.int32),
            'reward': np.zeros(capacity, dtype=np.float32),
            'next_state': np.zeros((capacity, state_dim), dtype=np.float32),
            'done': np.zeros(capacity, dtype=np.bool_),
        }
        self.next_idx = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        idx = self.next_idx
        self.data['state'][idx] = state
        self.data['action'][idx] = action
        self.data['reward'][idx] = reward
        self.data['next_state'][idx] = next_state
        self.data['done'][idx] = done

        self.next_idx = (idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

        priority_alpha = self.max_priority ** self.alpha
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)

    def _set_priority_min(self, idx, priority_alpha):
        idx += self.capacity
        self.priority_min[idx] = priority_alpha
        while idx >= 2:
            idx //= 2
            self.priority_min[idx] = min(self.priority_min[2 * idx],
                                         self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority):
        idx += self.capacity
        self.priority_sum[idx] = priority
        while idx >= 2:
            idx //= 2
            self.priority_sum[idx] = (self.priority_sum[2 * idx] +
                                      self.priority_sum[2 * idx + 1])

    def _sum(self):
        return self.priority_sum[1]

    def _min(self):
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):
        idx = 1
        while idx < self.capacity:
            if self.priority_sum[idx * 2] > prefix_sum:
                idx = 2 * idx
            else:
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1
        return idx - self.capacity

    def sample(self, batch_size, beta):
        samples = {
            'weights': np.zeros(batch_size, dtype=np.float32),
            'indexes': np.zeros(batch_size, dtype=np.int32),
        }

        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            samples['indexes'][i] = idx

        prob_min = self._min() / max(self._sum(), 1e-9)
        max_weight = (prob_min * self.size) ** (-beta) if self.size > 0 else 1.0

        for i in range(batch_size):
            idx = samples['indexes'][i]
            prob = self.priority_sum[idx + self.capacity] / max(self._sum(), 1e-9)
            weight = (prob * self.size) ** (-beta) if self.size > 0 else 1.0
            samples['weights'][i] = weight / max(max_weight, 1e-9)

        for k, v in self.data.items():
            samples[k] = v[samples['indexes']]

        return samples

    def update_priorities(self, indexes, priorities):
        for idx, priority in zip(indexes, priorities):
            self.max_priority = max(self.max_priority, float(priority))
            priority_alpha = float(priority) ** self.alpha
            self._set_priority_min(int(idx), priority_alpha)
            self._set_priority_sum(int(idx), priority_alpha)

    def is_full(self):
        return self.capacity == self.size
