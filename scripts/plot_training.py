"""
训练指标可视化 — 学术风格科研绘图
用法: python -m emergency.plot_training [--csv outputs/models/emergency_dqn_model_log.csv]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# ==================== 学术配色 ====================
# Paul Tol "Bright" qualitative palette (色盲友好)
COLORS = {
    'blue':   '#4477AA',
    'cyan':   '#66CCEE',
    'green':  '#228833',
    'yellow': '#CCBB44',
    'red':    '#EE6677',
    'purple': '#AA3377',
    'grey':   '#BBBBBB',
    'orange': '#EE7733',
}
SMOOTH_WINDOW = 50  # 移动平均窗口


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # 过滤 loss=0 的行 (训练尚未开始)
    return df


def smooth(series, window=SMOOTH_WINDOW):
    """移动平均平滑"""
    if len(series) < window:
        return series
    return uniform_filter1d(series.values.astype(float), size=window)


def setup_style():
    """学术图表样式"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_all(csv_path, output_dir='outputs'):
    setup_style()
    df = load_data(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # 只保留 loss>0 的数据用于loss曲线
    df_train = df[df['loss'] > 0].copy()

    max_ep = df['episode'].max()
    total_eps = len(df)

    # ==================== Figure 1: Reward + Loss + Epsilon + Steps ====================
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ((ax_r, ax_l), (ax_e, ax_s)) = axes

    # --- Reward ---
    ax_r.plot(df['episode'], df['reward'], color=COLORS['grey'], alpha=0.18, lw=0.5)
    ax_r.plot(df['episode'], smooth(df['reward']), color=COLORS['blue'], lw=1.5,
              label=f'Moving Avg ({SMOOTH_WINDOW} ep)')
    ax_r.axhline(y=0, color=COLORS['red'], lw=0.8, linestyle='--', alpha=0.6)
    ax_r.set_ylabel('Reward')
    ax_r.set_title('(a) Episode Reward')
    ax_r.legend(loc='lower right')

    # --- Loss ---
    if len(df_train) > 0:
        ax_l.plot(df_train['episode'], df_train['loss'], color=COLORS['grey'],
                  alpha=0.15, lw=0.5)
        ax_l.plot(df_train['episode'], smooth(df_train['loss']),
                  color=COLORS['red'], lw=1.5,
                  label=f'Moving Avg ({SMOOTH_WINDOW} ep)')
    ax_l.set_ylabel('TD Loss')
    ax_l.set_xlabel('Episode')
    ax_l.set_title('(b) Training Loss')
    ax_l.set_yscale('log')
    ax_l.legend(loc='upper right')

    # --- Epsilon ---
    ax_e.plot(df['episode'], df['epsilon'], color=COLORS['green'], lw=1.2)
    ax_e.set_ylabel('Epsilon')
    ax_e.set_xlabel('Episode')
    ax_e.set_title('(c) Exploration Rate ($\\varepsilon$)')
    ax_e.set_ylim(0, 1.05)

    # --- 替换为 Coverage Ratio (目标覆盖率) ---
    if 'coverage' in df.columns:
        ax_s.plot(df['episode'], df['coverage'], color=COLORS['grey'], alpha=0.15, lw=0.5)
        ax_s.plot(df['episode'], smooth(df['coverage']), color=COLORS['blue'], lw=1.5)
        ax_s.set_ylabel('Coverage Ratio')
        ax_s.set_xlabel('Episode')
        ax_s.set_title('(d) Target Coverage Rate')
        ax_s.set_ylim(0, 1.05)

    fig.suptitle(f'DQN Training Metrics ($n={max_ep}$ episodes)', fontsize=14,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'training_metrics.png')
    fig.savefig(path)
    print(f'[OK] {path}')
    plt.close(fig)

    # ==================== Figure 2: J_max + J_sum ====================
    fig, (ax_jm, ax_js) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_jm.plot(df['episode'], df['J_max'], color=COLORS['grey'], alpha=0.15, lw=0.5)
    ax_jm.plot(df['episode'], smooth(df['J_max']), color=COLORS['orange'], lw=1.5)
    ax_jm.set_ylabel('J$_{max}$ (s)')
    ax_jm.set_xlabel('Episode')
    ax_jm.set_title('(a) Makespan (worst UAV completion time)')
    ax_jm.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    ax_js.plot(df['episode'], df['J_sum'], color=COLORS['grey'], alpha=0.15, lw=0.5)
    ax_js.plot(df['episode'], smooth(df['J_sum']), color=COLORS['cyan'], lw=1.5)
    ax_js.set_ylabel('J$_{sum}$ (s)')
    ax_js.set_xlabel('Episode')
    ax_js.set_title('(b) Total Flight Time')
    ax_js.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    fig.suptitle(f'Mission Cost Evolution ($n={max_ep}$ episodes)', fontsize=14,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'training_cost.png')
    fig.savefig(path)
    print(f'[OK] {path}')
    plt.close(fig)

    # ==================== Figure 3: Cumulative Reward + Reward Histogram ====================
    fig, (ax_c, ax_h) = plt.subplots(1, 2, figsize=(12, 4.5))

    cumsum = df['reward'].cumsum()
    ax_c.plot(df['episode'], cumsum, color=COLORS['blue'], lw=1.2)
    ax_c.axhline(y=0, color=COLORS['red'], lw=0.8, linestyle='--', alpha=0.6)
    ax_c.set_ylabel('Cumulative Reward')
    ax_c.set_xlabel('Episode')
    ax_c.set_title('(a) Cumulative Reward')

    # 分段直方图：前半 vs 后半
    mid = total_eps // 2
    ax_h.hist(df['reward'].iloc[:mid], bins=40, alpha=0.5, color=COLORS['blue'],
              label=f'Eps 1-{mid}', density=True)
    ax_h.hist(df['reward'].iloc[mid:], bins=40, alpha=0.5, color=COLORS['red'],
              label=f'Eps {mid+1}-{total_eps}', density=True)
    ax_h.set_xlabel('Reward')
    ax_h.set_ylabel('Density')
    ax_h.set_title('(b) Reward Distribution')
    ax_h.legend()

    fig.suptitle(f'Training Progress Overview ($n={max_ep}$ episodes)', fontsize=14,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'training_overview.png')
    fig.savefig(path)
    print(f'[OK] {path}')
    plt.close(fig)

    # ==================== Figure 4: 航程约束 + 覆盖/放弃 ====================
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ((ax_rv, ax_rr), (ax_cv, ax_ab)) = axes

    # --- Range Violations ---
    ax_rv.plot(df['episode'], df['range_violations'], color=COLORS['grey'], alpha=0.12, lw=0.5)
    ax_rv.plot(df['episode'], smooth(df['range_violations']), color=COLORS['red'], lw=1.5)
    ax_rv.set_ylabel('Range Violations (count)')
    ax_rv.set_title('(a) UAVs Exceeding MAX_RANGE')
    ax_rv.set_yticks([0, 1, 2, 3, 4])
    ax_rv.axhline(y=0, color=COLORS['green'], lw=0.8, linestyle='--', alpha=0.5)

    # --- Max Range Ratio ---
    if 'max_range_ratio' in df.columns:
        ax_rr.plot(df['episode'], df['max_range_ratio'], color=COLORS['grey'], alpha=0.15, lw=0.5)
        ax_rr.plot(df['episode'], smooth(df['max_range_ratio']), color=COLORS['orange'], lw=1.5)
        ax_rr.axhline(y=1.0, color=COLORS['red'], lw=1.0, linestyle='--', alpha=0.7, label='MAX=1.0')
        ax_rr.set_ylabel('Max Range Ratio')
        ax_rr.set_title('(b) Max Range / MAX_RANGE')
        ax_rr.legend(loc='upper right')

    # --- Coverage ---
    if 'coverage' in df.columns:
        ax_cv.plot(df['episode'], df['coverage'], color=COLORS['grey'], alpha=0.12, lw=0.5)
        ax_cv.plot(df['episode'], smooth(df['coverage']), color=COLORS['blue'], lw=1.5)
        ax_cv.set_ylabel('Coverage Ratio')
        ax_cv.set_xlabel('Episode')
        ax_cv.set_title('(c) Target Coverage (assigned / affected)')
        ax_cv.set_ylim(0, 1.05)

    # --- Abandon Rate + Weights ---
    ax_ab2 = ax_ab.twinx()
    if 'abandon_avg_w' in df.columns and 'assign_avg_w' in df.columns:
        ax_ab.plot(df['episode'], smooth(df['abandon_avg_w']), color=COLORS['red'], lw=1.2,
                   alpha=0.8, label='Abandoned avg w')
        ax_ab.plot(df['episode'], smooth(df['assign_avg_w']), color=COLORS['green'], lw=1.2,
                   alpha=0.8, label='Assigned avg w')
        ax_ab.set_ylabel('Avg Weight', color=COLORS['purple'])
    if 'abandoned' in df.columns:
        ax_ab2.plot(df['episode'], smooth(df['abandoned']), color=COLORS['orange'], lw=1.0,
                    alpha=0.5, label='# Abandoned')
        ax_ab2.set_ylabel('# Abandoned', color=COLORS['orange'])
    ax_ab.set_xlabel('Episode')
    ax_ab.set_title('(d) Abandon Analysis')
    lines1, labels1 = ax_ab.get_legend_handles_labels()
    lines2, labels2 = ax_ab2.get_legend_handles_labels()
    ax_ab.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7)

    fig.suptitle(f'Constraint & Coverage Metrics ($n={max_ep}$ episodes)', fontsize=14,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'training_constraints.png')
    fig.savefig(path)
    print(f'[OK] {path}')
    plt.close(fig)

    # ==================== 统计摘要 ====================
    recent_n = max(100, int(max_ep * 0.2))
    early_n = max(100, int(max_ep * 0.2))
    recent = df.iloc[-recent_n:]
    early = df.iloc[:early_n]
    print(f'\n统计摘要 (共 {max_ep} 回合):')
    print(f'  Reward: 前{early_n}回合均值={early["reward"].mean():+.4f}  '
          f'→ 后{recent_n}回合均值={recent["reward"].mean():+.4f}')
    if len(df_train) > 0:
        recent_loss = df_train.iloc[-min(500, len(df_train)):]
        print(f'  Loss:   最终均值={recent_loss["loss"].mean():.6f}')
    print(f'  J_max:  前{early_n}回合均值={early["J_max"].mean():.0f}s  '
          f'→ 后{recent_n}回合均值={recent["J_max"].mean():.0f}s')
    print(f'  Steps:  均值={df["steps"].mean():.1f}')
    print(f'  Range Violations: 前{early_n}回合均值={early["range_violations"].mean():.1f}  '
          f'→ 后{recent_n}回合均值={recent["range_violations"].mean():.1f}')
    if 'coverage' in df.columns:
        print(f'  Coverage: 前{early_n}回合均值={early["coverage"].mean():.2f}  '
              f'→ 后{recent_n}回合均值={recent["coverage"].mean():.2f}')
    if 'abandoned' in df.columns:
        print(f'  Abandoned/ep: 前{early_n}回合均值={early["abandoned"].mean():.1f}  '
              f'→ 后{recent_n}回合均值={recent["abandoned"].mean():.1f}')
    if 'abandon_avg_w' in df.columns:
        print(f'  Abandon avg w: 后{recent_n}回合均值={recent["abandon_avg_w"].mean():.3f}'
              f' vs Assign avg w: {recent["assign_avg_w"].mean():.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练指标可视化')
    parser.add_argument('--csv', type=str, default='outputs/models/emergency_dqn_model_log.csv',
                        help='CSV日志路径')
    parser.add_argument('--output', type=str, default='outputs',
                        help='输出目录')
    args = parser.parse_args()
    plot_all(args.csv, args.output)
