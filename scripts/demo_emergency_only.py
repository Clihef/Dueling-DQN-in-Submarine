"""
Emergency scenario schematic plots without route trajectories.

This script draws only the emergency scene for S1-S4:
targets, shifts, new targets, no-fly zone, and failed UAV markers.
It does not render any actual flight trajectory.
"""

import argparse
import math
import os
import random
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import heatmap as hm
from emergency.agent import NUM_UAVS
from emergency.simulator import EmergencySimulator
from scripts.train import UNIFIED_HOTSPOTS, UAV_BASE_KM, dx_km, spiral_cfg


SPAN_KM = 100.0
OUTPUT_DIR = os.path.join('outputs', 'eval')
EVENTS = ['S1', 'S2', 'S3', 'S4']

EVENT_COLORS = {
    'S1': '#8C6D1F',
    'S2': '#D95F02',
    'S3': '#D73027',
    'S4': '#4575B4',
}

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'axes.linewidth': 1.0,
})


def make_prob_grid():
    x_km = np.arange(0, SPAN_KM + dx_km / 2, dx_km)
    y_km = np.arange(0, SPAN_KM + dx_km / 2, dx_km)
    X_km, Y_km = np.meshgrid(x_km, y_km)
    hm_hotspots = [
        {'center': h['center_km'], 'sigma': h['sigma'], 'weight': h['weight']}
        for h in UNIFIED_HOTSPOTS
    ]
    prob_grid = hm.generate_controlled_prob_field(
        X_km, Y_km, hm_hotspots, alpha=0.15
    )
    return prob_grid, X_km, Y_km


def build_simple_routes():
    """A compact baseline route set used only to generate S3/S4 emergencies."""
    sorted_ids = sorted(int(h['id']) for h in UNIFIED_HOTSPOTS)
    chunks = [
        sorted_ids[0:4],
        sorted_ids[4:8],
        sorted_ids[8:12],
        sorted_ids[12:15],
    ]
    while len(chunks) < NUM_UAVS:
        chunks.append([])
    return chunks[:NUM_UAVS]


def visual_nfz_radius(emergency):
    return float(emergency['no_fly_radius']) * 1.55 + 1.0


def nfz_overlaps_hotspots(emergency):
    nfz_x, nfz_y = emergency['no_fly_center']
    nfz_r = visual_nfz_radius(emergency)
    for tgt in UNIFIED_HOTSPOTS:
        x, y = tgt['center_km']
        dist = math.hypot(nfz_x - x, nfz_y - y)
        if dist < nfz_r + float(tgt['radius_km']) + 0.5:
            return True
    return False


def generate_s1_two(sim, max_attempts=200):
    emergency = None
    for _ in range(max_attempts):
        emergency = sim._generate_s1()
        if len(emergency.get('shifts', [])) == 2:
            return emergency
    return emergency


def generate_s2_three(sim, max_attempts=200):
    emergency = None
    for _ in range(max_attempts):
        emergency = sim._generate_s2()
        if len(emergency.get('new_targets', [])) == 3:
            return emergency
    return emergency


def generate_s3_clear(sim, routes, max_attempts=300):
    emergency = None
    for _ in range(max_attempts):
        emergency = sim._generate_s3(routes)
        if not nfz_overlaps_hotspots(emergency):
            return emergency
    return emergency


def setup_axes(ax):
    ax.set_xlim(0, SPAN_KM)
    ax.set_ylim(0, SPAN_KM)
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('white')
    ax.grid(True, axis='y', linestyle='--', color='#B8B8B8', alpha=0.25, linewidth=0.8)
    ax.grid(False, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('x / km')
    ax.set_ylabel('y / km')


def draw_target(ax, target, facecolor='white', edgecolor='#333333', alpha=1.0,
                linewidth=1.2, linestyle='-'):
    x, y = target['center_km']
    radius = target['radius_km']
    patch = Circle(
        (x, y), radius,
        facecolor=facecolor, edgecolor=edgecolor,
        alpha=alpha, linewidth=linewidth, linestyle=linestyle,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x, y, f"T{int(target['id'])}",
        ha='center', va='center', fontsize=9,
        color='#1F1F1F', weight='bold', zorder=4,
    )


def draw_base(ax):
    ax.scatter(
        [UAV_BASE_KM[0]], [UAV_BASE_KM[1]],
        marker='^', s=90, color='#111111', zorder=6,
        edgecolors='white', linewidths=0.7,
    )
    ax.text(
        UAV_BASE_KM[0] + 1.5, UAV_BASE_KM[1] + 1.5,
        'Base', fontsize=9, color='#111111', zorder=6,
    )


def add_legend(ax, handles):
    ax.legend(
        handles=handles,
        loc='upper right',
        frameon=True,
        framealpha=0.92,
        edgecolor='#D0D0D0',
        facecolor='white',
    )


def draw_route_order_from_base(ax, route, color, label=None, split_at=None):
    """Draw a schematic visit-order polyline starting from the base."""
    if not route:
        return

    coords = [UAV_BASE_KM]
    coords.extend(
        UNIFIED_HOTSPOTS[int(tid)]['center_km']
        for tid in route
        if 0 <= int(tid) < len(UNIFIED_HOTSPOTS)
    )
    if len(coords) < 2:
        return

    if split_at is None or split_at >= len(route):
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        ax.plot(
            xs, ys,
            linestyle='--', linewidth=1.6, color=color,
            marker='o', markersize=4.5, alpha=0.88,
            zorder=1,
        )
        for idx, (x, y) in enumerate(coords[1:], start=1):
            ax.text(
                x + 0.6, y + 0.6, str(idx),
                fontsize=7, color=color, weight='bold',
                zorder=2,
            )
    else:
        split_at = max(1, min(int(split_at), len(route) - 1))
        visited_coords = coords[:split_at + 1]
        future_coords = coords[split_at:]

        ax.plot(
            [p[0] for p in visited_coords],
            [p[1] for p in visited_coords],
            linestyle='-', linewidth=1.8, color=color,
            marker='o', markersize=4.5, alpha=0.92,
            zorder=1,
        )
        ax.plot(
            [p[0] for p in future_coords],
            [p[1] for p in future_coords],
            linestyle='--', linewidth=1.6, color='#777777',
            marker='o', markersize=4.0, alpha=0.7,
            zorder=1,
        )
        for idx, (x, y) in enumerate(visited_coords[1:], start=1):
            ax.text(
                x + 0.55, y + 0.55, f'V{idx}',
                fontsize=7, color=color, weight='bold',
                zorder=2,
            )
        for idx, (x, y) in enumerate(future_coords[1:], start=1):
            ax.text(
                x + 0.55, y + 0.55, f'F{idx}',
                fontsize=7, color='#666666', weight='bold',
                zorder=2,
            )

    if label:
        ax.text(
            coords[0][0] + 1.0, coords[0][1] + 1.0, label,
            fontsize=8.5, color=color, ha='left', va='bottom', zorder=2,
        )


def draw_original_targets(ax):
    for tgt in UNIFIED_HOTSPOTS:
        draw_target(
            ax, tgt, facecolor='#F5F5F5', edgecolor='#555555',
            alpha=0.85, linewidth=1.0,
        )


def draw_s1(ax, sim, emergency):
    shifted = {int(item['id']): item for item in emergency.get('shifts', [])}
    original = {int(h['id']): h for h in sim.original_hotspots}

    for tgt in UNIFIED_HOTSPOTS:
        tid = int(tgt['id'])
        if tid not in shifted:
            draw_target(ax, tgt, facecolor='#F5F5F5', edgecolor='#555555', alpha=0.85)
            continue

        old_tgt = original[tid]
        new_x = old_tgt['center_km'][0] + shifted[tid]['dx']
        new_y = old_tgt['center_km'][1] + shifted[tid]['dy']
        moved_tgt = {
            'id': tid,
            'center_km': (new_x, new_y),
            'radius_km': old_tgt['radius_km'],
        }

        draw_target(
            ax, old_tgt, facecolor='none', edgecolor=EVENT_COLORS['S1'],
            alpha=1.0, linewidth=1.4, linestyle='--',
        )
        draw_target(
            ax, moved_tgt, facecolor='#FFF6D8', edgecolor=EVENT_COLORS['S1'],
            alpha=0.95, linewidth=1.6,
        )
        arrow = FancyArrowPatch(
            posA=old_tgt['center_km'],
            posB=moved_tgt['center_km'],
            arrowstyle='->',
            mutation_scale=12,
            linewidth=1.2,
            color=EVENT_COLORS['S1'],
            zorder=5,
        )
        ax.add_patch(arrow)
        ax.text(
            old_tgt['center_km'][0],
            old_tgt['center_km'][1] + old_tgt['radius_km'] + 1.5,
            f"Old T{tid}",
            fontsize=8.5, color=EVENT_COLORS['S1'],
            ha='center', va='bottom', zorder=6,
        )
        ax.text(
            moved_tgt['center_km'][0],
            moved_tgt['center_km'][1] - moved_tgt['radius_km'] - 1.4,
            'Shifted',
            fontsize=8.5, color=EVENT_COLORS['S1'],
            ha='center', va='top', zorder=6,
        )

    ax.text(
        0.02, 0.98,
        f"S1: {len(emergency.get('affected_targets', []))} targets displaced",
        transform=ax.transAxes, ha='left', va='top',
        fontsize=10, weight='bold', color=EVENT_COLORS['S1'],
        bbox=dict(facecolor='white', edgecolor='#D0D0D0', alpha=0.92, boxstyle='round,pad=0.35'),
        zorder=7,
    )

    add_legend(ax, [
        Patch(facecolor='#F5F5F5', edgecolor='#555555', label='Original target'),
        Patch(facecolor='#FFF6D8', edgecolor=EVENT_COLORS['S1'], label='Shifted target'),
        Line2D([0], [0], color=EVENT_COLORS['S1'], linestyle='-', marker='o', label='Visit order'),
        Line2D([0], [0], marker='^', color='none', markerfacecolor='#111111',
               markeredgecolor='white', markersize=8, label='Base'),
    ])


def draw_s2(ax, emergency):
    for tgt in UNIFIED_HOTSPOTS:
        draw_target(ax, tgt, facecolor='#F5F5F5', edgecolor='#555555', alpha=0.85)

    new_targets = emergency.get('new_targets', [])
    for nt in new_targets:
        draw_target(
            ax, nt, facecolor='#FFE7D0', edgecolor=EVENT_COLORS['S2'],
            alpha=0.98, linewidth=1.8,
        )
        x, y = nt['center_km']
        ax.scatter(
            [x], [y], marker='*', s=120, color=EVENT_COLORS['S2'],
            edgecolors='white', linewidths=0.6, zorder=6,
        )
        ax.text(
            x, y + nt['radius_km'] + 1.3, f"New T{int(nt['id'])}",
            fontsize=8.5, color=EVENT_COLORS['S2'],
            ha='center', va='bottom', zorder=6,
        )

    ax.text(
        0.02, 0.98,
        f"S2: {len(new_targets)} new targets inserted",
        transform=ax.transAxes, ha='left', va='top',
        fontsize=10, weight='bold', color=EVENT_COLORS['S2'],
        bbox=dict(facecolor='white', edgecolor='#D0D0D0', alpha=0.92, boxstyle='round,pad=0.35'),
        zorder=7,
    )

    add_legend(ax, [
        Patch(facecolor='#F5F5F5', edgecolor='#555555', label='Existing target'),
        Patch(facecolor='#FFE7D0', edgecolor=EVENT_COLORS['S2'], label='New target'),
        Line2D([0], [0], marker='^', color='none', markerfacecolor='#111111',
               markeredgecolor='white', markersize=8, label='Base'),
    ])


def draw_s3(ax, emergency, simple_routes):
    for tgt in UNIFIED_HOTSPOTS:
        draw_target(ax, tgt, facecolor='#F5F5F5', edgecolor='#555555', alpha=0.85)

    affected_uav = int(emergency.get('affected_uav', 0))
    if 0 <= affected_uav < len(simple_routes):
        draw_route_order_from_base(
            ax, simple_routes[affected_uav],
            color='#5A5A5A', label=f'UAV{affected_uav + 1} visit order'
        )

    nfz_x, nfz_y = emergency['no_fly_center']
    nfz_r = visual_nfz_radius(emergency)
    nfz = Circle(
        (nfz_x, nfz_y), nfz_r,
        facecolor='#FDE0DD', edgecolor=EVENT_COLORS['S3'],
        alpha=0.45, linewidth=2.0, linestyle='--', zorder=2,
    )
    ax.add_patch(nfz)
    ax.text(
        nfz_x, nfz_y, 'NFZ',
        ha='center', va='center', fontsize=11,
        color=EVENT_COLORS['S3'], weight='bold', zorder=6,
    )

    for tid in emergency.get('affected_targets', []):
        tgt = next((h for h in UNIFIED_HOTSPOTS if int(h['id']) == int(tid)), None)
        if tgt is None:
            continue
        draw_target(
            ax, tgt, facecolor='#FFF1EC', edgecolor=EVENT_COLORS['S3'],
            alpha=0.95, linewidth=1.7,
        )

    ax.text(
        0.02, 0.98,
        'S3: no-fly zone blocks a future segment',
        transform=ax.transAxes, ha='left', va='top',
        fontsize=10, weight='bold', color=EVENT_COLORS['S3'],
        bbox=dict(facecolor='white', edgecolor='#D0D0D0', alpha=0.92, boxstyle='round,pad=0.35'),
        zorder=7,
    )

    add_legend(ax, [
        Line2D([0], [0], color='#5A5A5A', linestyle='--', marker='o', label='Visit order'),
        Patch(facecolor='#FDE0DD', edgecolor=EVENT_COLORS['S3'], label='No-fly zone'),
        Patch(facecolor='#FFF1EC', edgecolor=EVENT_COLORS['S3'], label='Affected target'),
        Line2D([0], [0], marker='^', color='none', markerfacecolor='#111111',
               markeredgecolor='white', markersize=8, label='Base'),
    ])


def draw_s4(ax, emergency, simple_routes):
    for tgt in UNIFIED_HOTSPOTS:
        draw_target(ax, tgt, facecolor='#F5F5F5', edgecolor='#555555', alpha=0.85)

    failed_uav = int(emergency['failed_uav'])
    lost_targets = emergency.get('lost_targets', [])
    split_at = None
    route = None

    if 0 <= failed_uav < len(simple_routes):
        route = simple_routes[failed_uav]
        split_at = max(1, len(route) // 2)
        draw_route_order_from_base(
            ax, route,
            color=EVENT_COLORS['S4'],
            label=f'UAV{failed_uav + 1} planned order',
            split_at=split_at,
        )

        coords = [UAV_BASE_KM] + [
            UNIFIED_HOTSPOTS[int(tid)]['center_km']
            for tid in route
            if 0 <= int(tid) < len(UNIFIED_HOTSPOTS)
        ]
        if len(coords) >= split_at + 2:
            p0 = np.asarray(coords[split_at], dtype=float)
            p1 = np.asarray(coords[split_at + 1], dtype=float)
            fail_pt = 0.5 * (p0 + p1)
            ax.scatter(
                [fail_pt[0]], [fail_pt[1]],
                marker='x', s=140, color=EVENT_COLORS['S4'],
                linewidths=2.2, zorder=8,
            )
            ax.text(
                fail_pt[0] + 1.0, fail_pt[1] + 1.0,
                'Failure point', fontsize=8.5,
                color=EVENT_COLORS['S4'], weight='bold', zorder=8,
            )

        visited_ids = route[:split_at]
        future_ids = route[split_at:]
        for tid in visited_ids:
            tgt = next((h for h in UNIFIED_HOTSPOTS if int(h['id']) == int(tid)), None)
            if tgt is None:
                continue
            draw_target(
                ax, tgt, facecolor='#EAF2FB', edgecolor='#7AA6D8',
                alpha=0.98, linewidth=1.7,
            )
        for tid in future_ids:
            tgt = next((h for h in UNIFIED_HOTSPOTS if int(h['id']) == int(tid)), None)
            if tgt is None:
                continue
            draw_target(
                ax, tgt, facecolor='none', edgecolor=EVENT_COLORS['S4'],
                alpha=1.0, linewidth=1.8, linestyle='--',
            )

    ax.scatter(
        [UAV_BASE_KM[0] + 4.5], [UAV_BASE_KM[1] + 4.5],
        marker='x', s=120, color=EVENT_COLORS['S4'],
        linewidths=2.0, zorder=7,
    )
    ax.text(
        UAV_BASE_KM[0] + 6.0, UAV_BASE_KM[1] + 4.8,
        f"UAV{failed_uav + 1} failure", fontsize=9.5,
        color=EVENT_COLORS['S4'], weight='bold', zorder=7,
    )

    ax.text(
        0.02, 0.98,
        'S4: one UAV fails and its pending targets are released',
        transform=ax.transAxes, ha='left', va='top',
        fontsize=10, weight='bold', color=EVENT_COLORS['S4'],
        bbox=dict(facecolor='white', edgecolor='#D0D0D0', alpha=0.92, boxstyle='round,pad=0.35'),
        zorder=7,
    )

    add_legend(ax, [
        Patch(facecolor='#EAF2FB', edgecolor='#7AA6D8', label='Visited target'),
        Patch(facecolor='none', edgecolor=EVENT_COLORS['S4'], label='Pending target'),
        Line2D([0], [0], color=EVENT_COLORS['S4'], linestyle='-', marker='o', label='Planned order'),
        Line2D([0], [0], marker='x', color=EVENT_COLORS['S4'], linestyle='None',
               markersize=8, label='Failure point'),
        Line2D([0], [0], marker='^', color='none', markerfacecolor='#111111',
               markeredgecolor='white', markersize=8, label='Base'),
    ])


def render_scene(sim, emergency, event_type, simple_routes, output_path):
    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    setup_axes(ax)
    draw_original_targets(ax)

    if event_type == 'S1':
        draw_s1(ax, sim, emergency)
    elif event_type == 'S2':
        draw_s2(ax, emergency)
    elif event_type == 'S3':
        draw_s3(ax, emergency, simple_routes)
    elif event_type == 'S4':
        draw_s4(ax, emergency, simple_routes)
    else:
        raise ValueError(f'Unknown event type: {event_type}')

    draw_base(ax)
    ax.set_title(f'Emergency Scenario {event_type} - Schematic View', fontsize=15, weight='bold', pad=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description='Render emergency scenario schematics only.')
    parser.add_argument('--type', choices=['S1', 'S2', 'S3', 'S4', 'all'], default='all')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    prob_grid, X_km, Y_km = make_prob_grid()
    sim = EmergencySimulator(UNIFIED_HOTSPOTS, NUM_UAVS, UAV_BASE_KM, spiral_cfg, prob_grid, X_km, Y_km)
    simple_routes = build_simple_routes()

    os.makedirs(args.output_dir, exist_ok=True)
    event_types = EVENTS if args.type == 'all' else [args.type]
    for event_type in event_types:
        if event_type == 'S1':
            emergency = generate_s1_two(sim)
        elif event_type == 'S2':
            emergency = generate_s2_three(sim)
        elif event_type == 'S3':
            emergency = generate_s3_clear(sim, simple_routes)
        elif event_type == 'S4':
            emergency = sim._generate_s4(simple_routes)
        else:
            raise ValueError(f'Unknown event type: {event_type}')

        output_path = os.path.join(args.output_dir, f'emergency_only_{event_type}.png')
        render_scene(sim, emergency, event_type, simple_routes, output_path)
        print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()
