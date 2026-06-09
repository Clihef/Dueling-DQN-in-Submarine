"""
应急重新规划评估。

在同一套平衡的
S1/S2/S3/S4 场景集上，将 DQN 与 GA 重新规划和规则基线进行比较。指标通过共享的模拟器
成本函数计算，因此训练/演示/评估使用相同的物理核算。
"""
import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from core import ga_allocator as ga
from core import heatmap as hm
from core.dubins_planner import dubins_curve
from core.spiral_search import spiral_search_arc_exact
from emergency.agent import EmergencyDQN, NUM_UAVS
from emergency.simulator import EmergencySimulator
from emergency.utils import (
    GLOBAL_C_DROP, MAX_FLIGHT_TIME_S, MAX_RANGE_KM, N_MAX, SPAN_KM,
    STATE_DIM_V2, UAV_VELOCITY_KM_S, UAV_V_M_S,
    compute_detour_distance_km, get_valid_actions_v2,
)
from scripts.train import UNIFIED_HOTSPOTS, UAV_BASE_KM, dx_km, spiral_cfg


METHODS = ['DQN', 'GA_Replan', 'Rigid_Rule', 'Distance_Greedy', 'Weight_Greedy']
EVENT_TYPES = ['S1', 'S2', 'S3', 'S4']
METHOD_LABELS = {
    'DQN': 'DQN',
    'GA_Replan': 'GA Replan',
    'Rigid_Rule': 'Rigid Rule',
    'Distance_Greedy': 'Distance Greedy',
    'Weight_Greedy': 'Weight Greedy',
}
METHOD_COLORS = {
    'DQN': '#E64B35',              # 朱红色 (强调色，突出你的核心算法)
    'GA_Replan': '#4DBBD5',        # 明亮的天青蓝 (代表强大的计算力/基准)
    'Rigid_Rule': '#7E6148',       # 泥土褐 (代表僵化、保守)
    'Distance_Greedy': '#00A087',  # 翡翠绿 
    'Weight_Greedy': '#3C5488',    # 深沉的藏青紫
}
GRID_COLOR = '#A8A8A8'
FUNCTION_FIELDS = [
    'range_ok', 'nfz_ok', 'position_response_ok',
    'new_target_response_ok', 'failure_response_ok', 'all_function_ok',
]
PERFORMANCE_FIELDS = [
    'decision_time_ms', 'ga_optimize_time_ms', 'ga_repair_time_ms',
    'ga_fill_time_ms', 'J_max_s', 'J_sum_s', 'W_arrival', 'weighted_arrival_sum',
    'total_range_km', 'future_range_km', 'max_uav_range_km',
    'coverage', 'weighted_coverage', 'abandoned_count', 'abandoned_weight',
    'range_violation_count', 'nfz_violation_count',
]


def resolve_project_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def target_weight_map(hotspots):
    return {h['id']: h.get('weight', 0.0) for h in hotspots}


def target_lookup(hotspots):
    return {h['id']: h for h in hotspots}


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


def run_initial_ga(prob_grid, X_km, Y_km):
    ga_targets = [
        {
            'id': h['id'], 'pos': h['center_km'],
            'weight': h['weight'], 'radius_km': h['radius_km'],
        }
        for h in UNIFIED_HOTSPOTS
    ]
    ga.init_ga_env(
        ga_targets, UAV_BASE_KM, NUM_UAVS,
        prob_grid=prob_grid, X_km=X_km, Y_km=Y_km,
        spiral_cfg=spiral_cfg, weights=(0.5, 0.3, 0.4),
        d_max=MAX_FLIGHT_TIME_S, c_drop=GLOBAL_C_DROP,
    )
    best_chrom, _, _, norm_factors = ga.run_ga(
        pop_size=200, generations=300, patience=100
    )
    _, raw_routes, _ = ga.evaluate_chromosome(best_chrom, norm_factors)
    return raw_routes[:NUM_UAVS]


def build_reference_paths(prob_grid, X_km, Y_km, baseline_routes):
    ref_paths = []
    route_boundaries = []
    for route in baseline_routes:
        if not route:
            ref_paths.append(np.empty((0, 2)))
            route_boundaries.append([])
            continue

        uav_path = []
        boundaries = []
        cumulative_dist = 0.0
        curr_pose = [UAV_BASE_KM[0] * 1000.0, UAV_BASE_KM[1] * 1000.0, 0.0]

        for tgt_idx in route:
            tgt = UNIFIED_HOTSPOTS[tgt_idx]
            target_m = [tgt['center_km'][0] * 1000.0,
                        tgt['center_km'][1] * 1000.0]
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
                spiral_sol['path'][1][0] - spiral_sol['path'][0][0],
            )
            d_path, d_len = dubins_curve(
                curr_pose, [s_start_pos[0], s_start_pos[1], s_start_yaw],
                r=spiral_cfg['uav']['Rmin'], stepsize=200.0,
            )
            if d_path is not None:
                uav_path.extend(d_path[:, 0:2].tolist())
                cumulative_dist += d_len
            uav_path.extend(s_path.tolist())
            cumulative_dist += spiral_sol['totalLen']
            boundaries.append(cumulative_dist)
            curr_pose = [
                spiral_sol['pEnd'][0], spiral_sol['pEnd'][1],
                spiral_sol['yawEnd'],
            ]

        ref_paths.append(np.array(uav_path))
        route_boundaries.append(boundaries)
    return ref_paths, route_boundaries


def build_eval_context(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    prob_grid, X_km, Y_km = make_prob_grid()
    sim = EmergencySimulator(
        UNIFIED_HOTSPOTS, NUM_UAVS, UAV_BASE_KM, spiral_cfg,
        prob_grid, X_km, Y_km,
    )
    baseline_routes = run_initial_ga(prob_grid, X_km, Y_km)
    ref_paths, route_boundaries = build_reference_paths(
        prob_grid, X_km, Y_km, baseline_routes
    )
    return {
        'sim': sim,
        'prob_grid': prob_grid,
        'X_km': X_km,
        'Y_km': Y_km,
        'baseline_routes': baseline_routes,
        'ref_paths': ref_paths,
        'route_boundaries': route_boundaries,
    }


def generate_raw_emergency(sim, event_type, baseline_routes):
    if event_type == 'S1':
        return sim._generate_s1()
    if event_type == 'S2':
        return sim._generate_s2()
    if event_type == 'S3':
        return sim._generate_s3(baseline_routes)
    if event_type == 'S4':
        return sim._generate_s4(baseline_routes)
    raise ValueError(f'Unknown event type: {event_type}')


def build_scenario(context, scenario_id, requested_type):
    sim = context['sim']
    baseline_routes = context['baseline_routes']
    ref_paths = context['ref_paths']
    route_boundaries = context['route_boundaries']

    emergency = generate_raw_emergency(sim, requested_type, baseline_routes)
    event_type = emergency['type']
    uav_positions, _, completed, remaining, consumed_ranges, active_target_ids, progress_kms = \
        sim.simulate_until_emergency(
            ref_paths, emergency, baseline_routes, route_boundaries
        )

    if event_type == 'S1':
        active_ids = {tid for tid in active_target_ids if tid is not None}
        shift_eligible = set(remaining) - active_ids
        if not shift_eligible:
            shift_eligible = set(remaining)
        emergency = sim._generate_s1(
            eligible_ids=shift_eligible,
            trigger_time_frac=emergency.get('trigger_time_frac', 0.5),
        )

    modified_hotspots, active_uavs, _ = sim.apply_emergency(emergency)
    remaining_ids = set(remaining)
    if event_type == 'S2':
        for nt in emergency.get('new_targets', []):
            remaining_ids.add(nt['id'])

    return {
        'scenario_id': scenario_id,
        'requested_type': requested_type,
        'event_type': event_type,
        'emergency': copy.deepcopy(emergency),
        'uav_positions': np.array(uav_positions, dtype=float),
        'completed': set(completed),
        'remaining_ids': remaining_ids,
        'consumed_ranges': list(consumed_ranges),
        'active_target_ids': list(active_target_ids),
        'progress_kms': list(progress_kms),
        'modified_hotspots': copy.deepcopy(modified_hotspots),
        'active_uavs': list(active_uavs),
    }


def generate_eval_scenarios(context, scenarios_per_type, seed):
    random.seed(seed)
    np.random.seed(seed)
    scenarios = []
    scenario_id = 0
    for event_type in EVENT_TYPES:
        for _ in range(scenarios_per_type):
            scenarios.append(build_scenario(context, scenario_id, event_type))
            scenario_id += 1
    return scenarios


def normalize_routes(routes):
    normalized = [[] for _ in range(NUM_UAVS)]
    for k, route in enumerate(routes[:NUM_UAVS]):
        normalized[k] = [int(tid) for tid in route]
    return normalized


def current_route_position(route, uav_idx, scenario, h_by_id):
    if route:
        return h_by_id[route[-1]]['center_km']
    return tuple(scenario['uav_positions'][uav_idx])


def leg_distance_km(uav_idx, target_id, routes, scenario, h_by_id):
    curr_pos = current_route_position(routes[uav_idx], uav_idx, scenario, h_by_id)
    target = h_by_id[target_id]
    tgt_pos = target['center_km']
    dist = np.hypot(tgt_pos[0] - curr_pos[0], tgt_pos[1] - curr_pos[1])
    if scenario['event_type'] == 'S3':
        nf_cx, nf_cy = scenario['emergency']['no_fly_center']
        nf_r = scenario['emergency']['no_fly_radius']
        dist += compute_detour_distance_km(curr_pos, tgt_pos, nf_cx, nf_cy, nf_r)
    spiral = target.get('spiral_dist')
    if spiral is None:
        spiral = target.get('spiral_time', 500) * UAV_V_M_S
    return dist + spiral / 1000.0


def route_cost(context, scenario, routes):
    return context['sim'].compute_cost_for_assignment(
        normalize_routes(routes),
        copy.deepcopy(scenario['modified_hotspots']),
        scenario['active_uavs'],
        scenario['emergency'],
        scenario['consumed_ranges'],
        scenario['uav_positions'],
        baseline_routes=context['baseline_routes'],
        completed=scenario['completed'],
        active_target_ids=scenario['active_target_ids'],
        progress_kms=scenario['progress_kms'],
    )


def routes_feasible(context, scenario, routes):
    return route_cost(context, scenario, routes).get('range_violations', 1) == 0


def repair_routes_to_range_feasible(context, scenario, routes):
    routes = normalize_routes(routes)
    removed = []
    guard = 0
    while guard < N_MAX + NUM_UAVS:
        cost = route_cost(context, scenario, routes)
        total_ranges = np.asarray(cost.get('total_ranges', np.zeros(NUM_UAVS)), dtype=float)
        if int(cost.get('range_violations', 0)) == 0:
            return routes, removed

        candidates = []
        for uav_idx, route in enumerate(routes):
            if uav_idx >= total_ranges.size or total_ranges[uav_idx] <= MAX_RANGE_KM:
                continue
            for pos, target_id in enumerate(route):
                tentative = copy.deepcopy(routes)
                tentative[uav_idx].pop(pos)
                tentative_cost = route_cost(context, scenario, tentative)
                tentative_ranges = np.asarray(
                    tentative_cost.get('total_ranges', np.zeros(NUM_UAVS)), dtype=float
                )
                max_range = float(np.max(tentative_ranges)) if tentative_ranges.size else 0.0
                target = next(
                    (h for h in scenario['modified_hotspots'] if h['id'] == target_id),
                    {'weight': 0.0},
                )
                candidates.append((
                    int(tentative_cost.get('range_violations', 0)),
                    max_range,
                    float(target.get('weight', 0.0)),
                    uav_idx,
                    pos,
                    target_id,
                    tentative,
                ))

        if not candidates:
            return routes, removed

        _, _, _, _, _, target_id, routes = min(candidates, key=lambda item: item[:3])
        removed.append(int(target_id))
        guard += 1

    return routes, removed


def fill_feasible_abandoned_targets(context, scenario, routes):
    routes = normalize_routes(routes)
    weights = target_weight_map(scenario['modified_hotspots'])
    inserted = []

    while True:
        assigned = {tid for route in routes for tid in route}
        abandoned = [
            tid for tid in scenario['remaining_ids']
            if tid not in assigned
        ]
        if not abandoned:
            return routes, inserted

        best_insert = None
        for target_id in sorted(abandoned, key=lambda tid: weights.get(tid, 0.0), reverse=True):
            for uav_idx, active in enumerate(scenario['active_uavs']):
                if not active:
                    continue
                for pos in range(len(routes[uav_idx]) + 1):
                    tentative = copy.deepcopy(routes)
                    tentative[uav_idx].insert(pos, int(target_id))
                    tentative_cost = route_cost(context, scenario, tentative)
                    if int(tentative_cost.get('range_violations', 0)) != 0:
                        continue
                    candidate = (
                        -float(weights.get(target_id, 0.0)),
                        float(tentative_cost.get('J_max', 0.0)),
                        float(tentative_cost.get('J_sum', 0.0)),
                        target_id,
                        tentative,
                    )
                    if best_insert is None or candidate < best_insert:
                        best_insert = candidate

        if best_insert is None:
            return routes, inserted

        _, _, _, target_id, routes = best_insert
        inserted.append(int(target_id))


def dqn_policy(context, scenario, dqn, device):
    sim = context['sim']
    if scenario['event_type'] == 'S3':
        current_routes = [
            [tid for tid in route if tid not in scenario['completed']]
            for route in context['baseline_routes']
        ]
        local_routes, local_feasible, _ = sim.s3_local_detour_routes(
            current_routes,
            scenario['active_uavs'],
            scenario['consumed_ranges'],
            uav_positions=scenario['uav_positions'],
            emergency=scenario['emergency'],
            baseline_routes=context['baseline_routes'],
            completed=scenario['completed'],
            active_target_ids=scenario['active_target_ids'],
            progress_kms=scenario['progress_kms'],
        )
        if local_feasible:
            return normalize_routes(local_routes), {'used_local_detour': True}

    state = sim.reset_v2(
        scenario['uav_positions'],
        scenario['consumed_ranges'],
        copy.deepcopy(scenario['modified_hotspots']),
        active_uavs=scenario['active_uavs'],
        completed=scenario['completed'],
        active_target_ids=scenario['active_target_ids'],
        progress_kms=scenario['progress_kms'],
        emergency=scenario['emergency'],
    )
    done = False
    routes = [[] for _ in range(NUM_UAVS)]
    guard = 0

    while not done and guard < N_MAX + NUM_UAVS + 4:
        all_locked = [idx for idx in sim.locked_target_idxs if idx is not None]
        valid_mask = get_valid_actions_v2(
            sim.current_uav_idx,
            sim.uav_states,
            sim.targets,
            locked_target_idx=sim.locked_target_idxs[sim.current_uav_idx],
            emergency=scenario['emergency'],
            all_locked_idxs=all_locked,
        )
        if not valid_mask.any():
            break
        with torch.no_grad():
            q_values = dqn(
                torch.FloatTensor(state).unsqueeze(0).to(device)
            ).squeeze(0).cpu().numpy()
        action = int(np.argmax(np.where(valid_mask, q_values, -np.inf)))
        routes[sim.current_uav_idx].append(int(sim.targets[action]['id']))
        state, _, done, _ = sim.step_v2(action)
        guard += 1

    return routes, {'used_local_detour': False}


def ga_replan_policy(context, scenario, ga_pop, ga_generations, ga_patience):
    remaining_ids = set(scenario['remaining_ids'])
    n_active = sum(scenario['active_uavs'])
    timing_meta = {
        'ga_optimize_time_ms': 0.0,
        'ga_repair_time_ms': 0.0,
        'ga_fill_time_ms': 0.0,
    }
    if not remaining_ids or n_active == 0:
        return [[] for _ in range(NUM_UAVS)], timing_meta

    ga_targets = []
    for h in scenario['modified_hotspots']:
        if h['id'] not in remaining_ids:
            continue
        spiral_dist = h.get('spiral_dist')
        if spiral_dist is None:
            spiral_dist = h.get('spiral_time', 500) * UAV_V_M_S
        ga_targets.append({
            'id': h['id'],
            'pos': h['center_km'],
            'weight': h['weight'],
            'radius_km': h['radius_km'],
            'spiral_dist': spiral_dist,
            'spiral_time': h.get('spiral_time', spiral_dist / UAV_V_M_S),
        })

    if not ga_targets:
        return [[] for _ in range(NUM_UAVS)], timing_meta

    active_indices = [i for i, active in enumerate(scenario['active_uavs']) if active]
    active_start_positions = [
        tuple(scenario['uav_positions'][k]) for k in active_indices
    ]
    full_d_max_s = MAX_RANGE_KM / UAV_VELOCITY_KM_S
    per_uav_d_max = [
        full_d_max_s - scenario['consumed_ranges'][k] / UAV_VELOCITY_KM_S
        for k in active_indices
    ]

    try:
        optimize_start = time.perf_counter()
        ga.init_ga_env(
            ga_targets, active_start_positions, n_active,
            weights=(0.5, 0.3, 0.4),
            d_max=per_uav_d_max if per_uav_d_max else full_d_max_s,
            c_drop=GLOBAL_C_DROP,
        )
        best_chrom, _, _, _ = ga.run_ga(
            pop_size=ga_pop, generations=ga_generations, patience=ga_patience
        )
        _, _, _, raw_routes, _, _, _ = ga.calculate_raw_costs(best_chrom)
        timing_meta['ga_optimize_time_ms'] = (
            time.perf_counter() - optimize_start
        ) * 1000.0
    except Exception as exc:
        print(f'[WARNING] GA_Replan failed in scenario {scenario["scenario_id"]}: {exc}')
        timing_meta['failed'] = True
        return [[] for _ in range(NUM_UAVS)], timing_meta

    routes = [[] for _ in range(NUM_UAVS)]
    for local_uav_idx, route in enumerate(raw_routes[:n_active]):
        real_uav_idx = active_indices[local_uav_idx]
        routes[real_uav_idx] = [int(ga_targets[tgt_idx]['id']) for tgt_idx in route]

    repair_start = time.perf_counter()
    repaired_routes, repair_removed = repair_routes_to_range_feasible(context, scenario, routes)
    timing_meta['ga_repair_time_ms'] = (time.perf_counter() - repair_start) * 1000.0

    fill_start = time.perf_counter()
    filled_routes, fill_inserted = fill_feasible_abandoned_targets(
        context, scenario, repaired_routes
    )
    timing_meta['ga_fill_time_ms'] = (time.perf_counter() - fill_start) * 1000.0
    timing_meta.update({
        'failed': False,
        'repair_removed_ids': repair_removed,
        'fill_inserted_ids': fill_inserted,
    })
    return filled_routes, timing_meta


def rigid_rule_policy(context, scenario):
    routes = [
        [tid for tid in route if tid not in scenario['completed']]
        for route in context['baseline_routes']
    ]
    if scenario['event_type'] == 'S4':
        failed_uav = scenario['emergency']['failed_uav']
        routes[failed_uav] = []
    return normalize_routes(routes), {'responds_position': False}


def distance_greedy_policy(context, scenario):
    routes = [[] for _ in range(NUM_UAVS)]
    unassigned = set(scenario['remaining_ids'])
    h_by_id = target_lookup(scenario['modified_hotspots'])

    while unassigned:
        best = None
        for target_id in sorted(unassigned):
            if target_id not in h_by_id:
                continue
            for uav_idx, active in enumerate(scenario['active_uavs']):
                if not active:
                    continue
                tentative = copy.deepcopy(routes)
                tentative[uav_idx].append(target_id)
                if not routes_feasible(context, scenario, tentative):
                    continue
                score = leg_distance_km(uav_idx, target_id, routes, scenario, h_by_id)
                if best is None or score < best[0]:
                    best = (score, uav_idx, target_id)
        if best is None:
            break
        _, uav_idx, target_id = best
        routes[uav_idx].append(target_id)
        unassigned.remove(target_id)

    return routes, {'responds_position': True}


def weight_greedy_policy(context, scenario):
    routes = [[] for _ in range(NUM_UAVS)]
    h_by_id = target_lookup(scenario['modified_hotspots'])
    weights = target_weight_map(scenario['modified_hotspots'])
    ordered_targets = sorted(
        scenario['remaining_ids'],
        key=lambda tid: weights.get(tid, 0.0),
        reverse=True,
    )

    for target_id in ordered_targets:
        if target_id not in h_by_id:
            continue
        candidates = []
        for uav_idx, active in enumerate(scenario['active_uavs']):
            if not active:
                continue
            tentative = copy.deepcopy(routes)
            tentative[uav_idx].append(target_id)
            if routes_feasible(context, scenario, tentative):
                candidates.append((
                    leg_distance_km(uav_idx, target_id, routes, scenario, h_by_id),
                    uav_idx,
                ))
        if candidates:
            _, best_uav = min(candidates)
            routes[best_uav].append(target_id)

    return routes, {'responds_position': True}


def run_policy(context, scenario, method, dqn, device, args):
    start = time.perf_counter()
    if method == 'DQN':
        routes, meta = dqn_policy(context, scenario, dqn, device)
    elif method == 'GA_Replan':
        routes, meta = ga_replan_policy(
            context, scenario, args.ga_pop, args.ga_generations, args.ga_patience
        )
    elif method == 'Rigid_Rule':
        routes, meta = rigid_rule_policy(context, scenario)
    elif method == 'Distance_Greedy':
        routes, meta = distance_greedy_policy(context, scenario)
    elif method == 'Weight_Greedy':
        routes, meta = weight_greedy_policy(context, scenario)
    else:
        raise ValueError(f'Unknown method: {method}')
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    meta['decision_time_ms'] = elapsed_ms # type: ignore
    return normalize_routes(routes), meta


def evaluate_method_result(context, scenario, method, routes, meta):
    routes = normalize_routes(routes)
    cost = route_cost(context, scenario, routes)
    remaining_ids = set(scenario['remaining_ids'])
    weights = target_weight_map(scenario['modified_hotspots'])

    assigned_ids = set()
    failed_uav = scenario['emergency'].get('failed_uav') \
        if scenario['event_type'] == 'S4' else None
    for uav_idx, route in enumerate(routes):
        if failed_uav is not None and uav_idx == failed_uav:
            continue
        if scenario['active_uavs'][uav_idx]:
            assigned_ids.update(route)

    covered_ids = assigned_ids & remaining_ids
    if scenario['event_type'] == 'S1' and not meta.get('responds_position', True):
        shifted_ids = set(scenario['emergency'].get('affected_targets', []))
        covered_ids -= shifted_ids

    abandoned_ids = remaining_ids - covered_ids
    total_weight = sum(weights.get(tid, 0.0) for tid in remaining_ids)
    covered_weight = sum(weights.get(tid, 0.0) for tid in covered_ids)

    raw_nfz_violations = int(cost.get('nfz_violations', 0))
    if scenario['event_type'] == 'S3':
        # raw_nfz_violations counts straight-line intersections with the no-fly
        # circle. Replanning methods use the shared S3 detour model, so an
        # intersection means "detour needed", not "entered the no-fly zone".
        # Rigid_Rule keeps the original path and does not react to S3, so its
        # straight-line intersection remains a functional NFZ failure.
        effective_nfz_violations = (
            raw_nfz_violations
            if method == 'Rigid_Rule' and not meta.get('used_local_detour')
            else 0
        )
    else:
        effective_nfz_violations = raw_nfz_violations
    range_violation_count = int(cost.get('range_violations', 0))

    affected_s1 = set(scenario['emergency'].get('affected_targets', [])) \
        if scenario['event_type'] == 'S1' else set()
    new_targets = {t['id'] for t in scenario['emergency'].get('new_targets', [])} \
        if scenario['event_type'] == 'S2' else set()

    position_response_ok = True
    if scenario['event_type'] == 'S1' and affected_s1:
        position_response_ok = meta.get('responds_position', True) and affected_s1 <= covered_ids

    new_target_response_ok = True
    if scenario['event_type'] == 'S2' and new_targets:
        new_target_response_ok = bool(new_targets & covered_ids)

    failure_response_ok = True
    if scenario['event_type'] == 'S4':
        failed_pending_ids = set()
        if failed_uav is not None and failed_uav < len(context['baseline_routes']):
            failed_pending_ids = set(context['baseline_routes'][failed_uav]) & remaining_ids
        failure_response_ok = (
            bool(failed_pending_ids & assigned_ids)
            if failed_pending_ids else True
        )

    range_ok = range_violation_count == 0
    nfz_ok = effective_nfz_violations == 0
    all_function_ok = all([
        range_ok, nfz_ok, position_response_ok,
        new_target_response_ok, failure_response_ok,
    ])

    route_dists = np.asarray(cost.get('route_dists_km', np.zeros(NUM_UAVS)), dtype=float)
    total_ranges = np.asarray(cost.get('total_ranges', np.zeros(NUM_UAVS)), dtype=float)
    if total_ranges.size == 0:
        total_ranges = np.asarray(scenario['consumed_ranges'], dtype=float) + route_dists

    row = {
        'scenario_id': scenario['scenario_id'],
        'event_type': scenario['event_type'],
        'method': method,
        'decision_time_ms': meta.get('decision_time_ms', 0.0),
        'ga_optimize_time_ms': meta.get('ga_optimize_time_ms', 0.0),
        'ga_repair_time_ms': meta.get('ga_repair_time_ms', 0.0),
        'ga_fill_time_ms': meta.get('ga_fill_time_ms', 0.0),
        'range_ok': range_ok,
        'nfz_ok': nfz_ok,
        'position_response_ok': position_response_ok,
        'new_target_response_ok': new_target_response_ok,
        'failure_response_ok': failure_response_ok,
        'all_function_ok': all_function_ok,
        'J_max_s': float(cost.get('J_max', 0.0)),
        'J_sum_s': float(cost.get('J_sum', 0.0)),
        'W_arrival': float(cost.get('weighted_arrival_sum', 0.0)),
        'weighted_arrival_sum': float(cost.get('weighted_arrival_sum', 0.0)),
        'total_range_km': float(np.sum(total_ranges)),
        'future_range_km': float(np.sum(route_dists)),
        'max_uav_range_km': float(np.max(total_ranges)) if total_ranges.size else 0.0,
        'coverage': len(covered_ids) / max(len(remaining_ids), 1),
        'weighted_coverage': covered_weight / max(total_weight, 1e-9),
        'abandoned_count': len(abandoned_ids),
        'abandoned_weight': float(sum(weights.get(tid, 0.0) for tid in abandoned_ids)),
        'range_violation_count': range_violation_count,
        'nfz_violation_count': effective_nfz_violations,
        'raw_nfz_intersections': raw_nfz_violations,
        'used_local_detour': bool(meta.get('used_local_detour', False)),
        'ga_repair_removed_count': len(meta.get('repair_removed_ids', [])),
        'ga_fill_inserted_count': len(meta.get('fill_inserted_ids', [])),
        'ga_repair_removed_ids_json': json.dumps(
            [int(tid) for tid in meta.get('repair_removed_ids', [])]
        ),
        'ga_fill_inserted_ids_json': json.dumps(
            [int(tid) for tid in meta.get('fill_inserted_ids', [])]
        ),
        'routes_json': json.dumps(routes, ensure_ascii=False),
        'abandoned_ids_json': json.dumps(sorted(int(tid) for tid in abandoned_ids)),
    }
    return row


def load_dqn(model_path, device):
    dqn = EmergencyDQN(state_dim=STATE_DIM_V2, num_actions=N_MAX).to(device)
    if os.path.exists(model_path):
        dqn.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        print(f'DQN model loaded: {model_path}')
    else:
        print(f'[WARNING] Model not found: {model_path}; using untrained DQN.')
    dqn.eval()
    return dqn


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {'mean': 0.0, 'median': 0.0, 'p25': 0.0, 'p75': 0.0, 'std': 0.0}
    return {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
        'std': float(np.std(arr)),
    }


def summarize(rows, group_fields):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped[key].append(row)

    summary = []
    for key, items in sorted(grouped.items()):
        out = {field: key[idx] for idx, field in enumerate(group_fields)}
        for field in FUNCTION_FIELDS:
            out[f'{field}_rate'] = 100.0 * np.mean([float(item[field]) for item in items])
        for field in PERFORMANCE_FIELDS:
            field_stats = stats([float(item[field]) for item in items])
            out[f'{field}_mean'] = field_stats['mean']
            out[f'{field}_median'] = field_stats['median']
            out[f'{field}_p25'] = field_stats['p25']
            out[f'{field}_p75'] = field_stats['p75']
            out[f'{field}_std'] = field_stats['std']
        out['n'] = len(items)
        summary.append(out)
    return summary


def print_core_summary(summary):
    print('\nEvaluation summary')
    print('-' * 146)
    header = (
        f'{"Method":<16} {"AllOK%":>8} {"RangeOK%":>9} {"NFZOK%":>8} '
        f'{"Cov":>7} {"WCov":>7} {"Jmax(s)":>10} {"Jsum(s)":>10} '
        f'{"WArr":>12} {"MaxKm":>9} {"Time(ms)":>10}'
    )
    print(header)
    print('-' * 146)
    for row in summary:
        print(
            f'{row["method"]:<16} '
            f'{row["all_function_ok_rate"]:>8.1f} '
            f'{row["range_ok_rate"]:>9.1f} '
            f'{row["nfz_ok_rate"]:>8.1f} '
            f'{row["coverage_mean"]:>7.3f} '
            f'{row["weighted_coverage_mean"]:>7.3f} '
            f'{row["J_max_s_mean"]:>10.1f} '
            f'{row["J_sum_s_mean"]:>10.1f} '
            f'{row["W_arrival_mean"]:>12.2e} '
            f'{row["max_uav_range_km_mean"]:>9.1f} '
            f'{row["decision_time_ms_mean"]:>10.1f}'
        )
    print('-' * 146)


def apply_scientific_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.titlesize': 15,
        'axes.titleweight': 'bold',
        'axes.labelsize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.linewidth': 1.0,
        'savefig.bbox': 'tight',
    })


def style_cartesian_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', color=GRID_COLOR, linestyle='--', linewidth=0.8, alpha=0.3)
    ax.grid(False, axis='x')
    ax.set_facecolor('white')


def method_color(method):
    return METHOD_COLORS[method]


def method_label(method):
    return str(METHOD_LABELS.get(method, method))


def summary_by_method(summary):
    return {row['method']: row for row in summary}


def quality_adjusted_makespan_score(row):
    return (
        float(row['weighted_coverage_mean'])
        * float(row['all_function_ok_rate']) / 100.0
        / max(float(row['J_max_s_mean']), 1e-9)
    )


def balanced_coverage_score(row):
    return (
        0.5 * float(row['coverage_mean'])
        + 0.5 * float(row['weighted_coverage_mean'])
    )


def balanced_coverage_time_score(row):
    return balanced_coverage_score(row) / max(float(row['J_max_s_mean']), 1e-9)


def mission_quality_score(row):
    """Composite quality score for Pareto Y axis, independent of decision time."""
    safety_score = (
        float(row['range_ok_rate']) + float(row['nfz_ok_rate'])
    ) / 200.0
    function_score = float(row['all_function_ok_rate']) / 100.0
    return (
        0.40 * float(row['coverage_mean'])
        + 0.30 * float(row['weighted_coverage_mean'])
        + 0.15 * safety_score
        + 0.15 * function_score
    )


def plot_function_success(rows, output_dir):
    apply_scientific_style()
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['event_type'], row['method'])].append(row['all_function_ok'])

    x = np.arange(len(EVENT_TYPES))
    width = 0.14
    fig, ax = plt.subplots(figsize=(11, 5))
    for idx, method in enumerate(METHODS):
        rates = [
            100.0 * np.mean(grouped.get((event_type, method), [False]))
            for event_type in EVENT_TYPES
        ]
        ax.bar(
            x + (idx - 2) * width, rates, width,
            color=method_color(method), edgecolor='white', linewidth=0.7,
            label=method_label(method),
        )
    ax.set_title('Functional Success by Emergency Type')
    ax.set_ylabel('Function Success Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(EVENT_TYPES)
    ax.set_ylim(0, 105)
    style_cartesian_axis(ax)
    ax.legend(ncols=3, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'function_success_by_type.png'), dpi=200)
    plt.close(fig)


def normalize_higher_better(values):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    v_min = np.min(finite)
    v_max = np.max(finite)
    if abs(v_max - v_min) < 1e-12:
        return np.ones_like(arr)
    return (arr - v_min) / (v_max - v_min)


def normalize_lower_better(values):
    return 1.0 - normalize_higher_better(values)


def task_efficiency_scores(summary_rows):
    """Effective emergency mission output per combined execution/computation cost."""
    j_max = np.asarray([float(row['J_max_s_mean']) for row in summary_rows], dtype=float)
    j_sum = np.asarray([float(row['J_sum_s_mean']) for row in summary_rows], dtype=float)
    w_arrival = np.asarray([float(row['W_arrival_mean']) for row in summary_rows], dtype=float)
    total_range = np.asarray([float(row['total_range_km_mean']) for row in summary_rows], dtype=float)
    decision_time = np.log1p(np.asarray([
        float(row['decision_time_ms_mean']) for row in summary_rows
    ], dtype=float))
    effective_output = np.asarray([
        float(row['coverage_mean'])
        * float(row['weighted_coverage_mean'])
        * float(row['all_function_ok_rate']) / 100.0
        * (float(row['range_ok_rate']) + float(row['nfz_ok_rate'])) / 200.0
        for row in summary_rows
    ], dtype=float)

    combined_cost = (
        0.25 * j_max / max(float(np.max(j_max)), 1e-9)
        + 0.15 * j_sum / max(float(np.max(j_sum)), 1e-9)
        + 0.25 * w_arrival / max(float(np.max(w_arrival)), 1e-9)
        + 0.20 * total_range / max(float(np.max(total_range)), 1e-9)
        + 0.15 * decision_time / max(float(np.max(decision_time)), 1e-9)
    )
    return normalize_higher_better(effective_output / np.maximum(combined_cost, 1e-9))


def plot_radar_overall_metrics(summary, output_dir):
    apply_scientific_style()
    by_method = summary_by_method(summary)
    available_methods = [method for method in METHODS if method in by_method]
    method_rows = [by_method[m] for m in available_methods]
    efficiency_scores = task_efficiency_scores(method_rows)

    raw = {
        'Success': [by_method[m]['all_function_ok_rate'] / 100.0 for m in available_methods],
        'Coverage': [by_method[m]['coverage_mean'] for m in available_methods],
        'Safety': [
            (by_method[m]['range_ok_rate'] + by_method[m]['nfz_ok_rate']) / 200.0
            for m in available_methods
        ],
        'Real-time\nEfficiency': normalize_lower_better([
            by_method[m]['decision_time_ms_mean'] for m in available_methods
        ]),
        'Task\nEfficiency': efficiency_scores,
    }

    labels = list(raw.keys())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw={'polar': True})
    ax.set_theta_offset(np.pi / 2) # type: ignore
    ax.set_theta_direction(-1) # type: ignore

    for idx, method in enumerate(available_methods):
        values = [float(raw[label][idx]) for label in labels]
        values += values[:1]
        ax.plot(
            angles, values, color=method_color(method), linewidth=2.2,
            label=method_label(method),
        )
        ax.fill(angles, values, color=method_color(method), alpha=0.20)

    ax.set_title('Overall Policy Capability Radar', pad=22)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(color=GRID_COLOR, linestyle='--', linewidth=0.8, alpha=0.3)
    ax.spines['polar'].set_color('#4A4A4A')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncols=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'radar_overall_metrics.png'), dpi=250)
    plt.close(fig)


def plot_half_violin_swarm(rows, output_dir):
    apply_scientific_style()
    metrics = [
        ('decision_time_ms', 'Decision Time (ms)', False),
        ('J_max_s', 'J_max Cost (s)', False),
        ('J_sum_s', 'J_sum Cost (s)', False),
        ('W_arrival', 'W_arrival (log10)', True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.4))
    axes = np.asarray(axes).ravel()
    rng = np.random.default_rng(202405)

    for ax, (field, title, log10_values) in zip(axes, metrics):
        positions = np.arange(len(METHODS), dtype=float)
        data = []
        for method in METHODS:
            values = np.asarray(
                [float(row[field]) for row in rows if row['method'] == method],
                dtype=float,
            )
            if log10_values:
                values = values[np.isfinite(values) & (values > 0.0)]
                values = np.log10(values) if values.size else values
            data.append(values)

        for pos, method, values in zip(positions, METHODS, data):
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            violin = ax.violinplot(
                values, positions=[pos], widths=0.72, showmeans=False,
                showmedians=False, showextrema=False,
            )
            for body in violin['bodies']:
                body.set_facecolor(method_color(method))
                body.set_edgecolor(method_color(method))
                body.set_alpha(0.28)
                body.set_linewidth(1.2)
                verts = body.get_paths()[0].vertices
                verts[:, 0] = np.minimum(verts[:, 0], pos)

            jitter = rng.uniform(0.06, 0.34, size=values.size)
            ax.scatter(
                np.full(values.size, pos) + jitter, values,
                s=22, color=method_color(method), alpha=0.72,
                edgecolor='white', linewidth=0.35, zorder=3,
            )
            q1, med, q3 = np.percentile(values, [25, 50, 75])
            ax.plot([pos - 0.34, pos + 0.34], [med, med],
                    color='#222222', linewidth=1.5, zorder=4)
            ax.plot([pos + 0.40, pos + 0.40], [q1, q3],
                    color='#222222', linewidth=1.2, zorder=4)

        ax.set_title(title)
        ax.set_ylabel(title)
        ax.set_xticks(positions)
        ax.set_xticklabels([method_label(m) for m in METHODS], rotation=24, ha='right')
        style_cartesian_axis(ax)

    fig.suptitle('Distribution of Real-time and Mission Cost Metrics', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'violin_swarm_performance.png'), dpi=250)
    plt.close(fig)


def pareto_front(points):
    sorted_points = sorted(points, key=lambda p: (p['x'], -p['y']))
    front = []
    best_y = -float('inf')
    for point in sorted_points:
        if point['y'] > best_y + 1e-12:
            front.append(point)
            best_y = point['y']
    return front


def plot_pareto_front_bubble(summary, output_dir):
    apply_scientific_style()
    quality_scores = [mission_quality_score(row) for row in summary]
    normalized_quality = normalize_higher_better(quality_scores)
    points = []
    for row, quality_score, y_value in zip(summary, quality_scores, normalized_quality):
        method = row['method']
        points.append({
            'method': method,
            'x': max(float(row['decision_time_ms_mean']), 1e-3),
            'y': float(y_value),
            'quality_score': float(quality_score),
            'coverage': float(row['weighted_coverage_mean']),
            'range': float(row['total_range_km_mean']),
        })

    ranges = np.asarray([p['range'] for p in points], dtype=float)
    if np.ptp(ranges) < 1e-9:
        sizes = np.full(len(points), 520.0)
    else:
        sizes = 280.0 + 680.0 * (ranges - ranges.min()) / np.ptp(ranges)

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    for point, size in zip(points, sizes):
        method = point['method']
        ax.scatter(
            point['x'], point['y'], s=size,
            color=method_color(method), alpha=0.72,
            edgecolor='white', linewidth=1.1, label=method_label(method), zorder=3,
        )
        ax.annotate(
            method_label(method), (point['x'], point['y']),
            xytext=(6, 5), textcoords='offset points', fontsize=10,
            color=method_color(method), weight='bold',
        )

    front = pareto_front(points)
    if len(front) >= 2:
        ax.plot(
            [p['x'] for p in front], [p['y'] for p in front],
            color='#333333', linestyle='--', linewidth=1.4,
            label='Pareto Front', zorder=2,
        )

    ax.set_xscale('log')
    ax.set_title('Pareto Front: Real-time Decision vs. Mission Quality')
    ax.set_xlabel('Decision Time (ms, log scale)')
    ax.set_ylabel('Composite Mission Quality (normalized)')
    ax.set_ylim(-0.04, 1.06)
    style_cartesian_axis(ax)
    ax.legend(frameon=False, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'pareto_front_bubble.png'), dpi=250)
    plt.close(fig)


def plot_range_vs_coverage(rows, output_dir):
    apply_scientific_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in METHODS:
        xs = [float(row['max_uav_range_km']) for row in rows if row['method'] == method]
        ys = [float(row['coverage']) for row in rows if row['method'] == method]
        ax.scatter(
            xs, ys, s=24, color=method_color(method), alpha=0.62,
            edgecolor='white', linewidth=0.35, label=method_label(method),
        )
    ax.axvline(MAX_RANGE_KM, color='#222222', linestyle='--', linewidth=1.2)
    ax.set_title('Coverage Under Range Constraint')
    ax.set_xlabel('Max UAV Range (km)')
    ax.set_ylabel('Coverage')
    style_cartesian_axis(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'range_vs_coverage.png'), dpi=200)
    plt.close(fig)


def plot_summary_table(summary, output_dir):
    apply_scientific_style()
    columns = [
        'Method', 'AllOK%', 'RangeOK%', 'NFZOK%', 'Cov', 'W-Cov',
        'Jmax(s)', 'Jsum(s)', 'WArr', 'Time(ms)',
    ]
    table_rows = []
    for row in summary:
        table_rows.append([
            row['method'],
            f'{row["all_function_ok_rate"]:.1f}',
            f'{row["range_ok_rate"]:.1f}',
            f'{row["nfz_ok_rate"]:.1f}',
            f'{row["coverage_mean"]:.3f}',
            f'{row["weighted_coverage_mean"]:.3f}',
            f'{row["J_max_s_mean"]:.1f}',
            f'{row["J_sum_s_mean"]:.1f}',
            f'{row["W_arrival_mean"]:.2e}',
            f'{row["decision_time_ms_mean"]:.1f}',
        ])
    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.axis('off')
    table = ax.table(cellText=table_rows, colLabels=columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (row_idx, _), cell in table.get_celld().items():
        cell.set_edgecolor('#D0D0D0')
        if row_idx == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#F2F2F2')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'summary_table.png'), dpi=200)
    plt.close(fig)


def write_outputs(rows, output_dir, no_plots):
    os.makedirs(output_dir, exist_ok=True)

    detail_fields = [
        'scenario_id', 'event_type', 'method',
        *FUNCTION_FIELDS,
        *PERFORMANCE_FIELDS,
        'raw_nfz_intersections', 'used_local_detour',
        'ga_repair_removed_count', 'ga_fill_inserted_count',
        'ga_repair_removed_ids_json', 'ga_fill_inserted_ids_json',
        'routes_json', 'abandoned_ids_json',
    ]
    write_csv(os.path.join(output_dir, 'eval_detail.csv'), rows, detail_fields)

    summary = summarize(rows, ['method'])
    summary_by_type = summarize(rows, ['method', 'event_type'])
    write_csv(os.path.join(output_dir, 'eval_summary.csv'), summary, list(summary[0].keys()))
    write_csv(
        os.path.join(output_dir, 'eval_summary_by_type.csv'),
        summary_by_type,
        list(summary_by_type[0].keys()),
    )

    with open(os.path.join(output_dir, 'eval_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(
            {'summary': summary, 'summary_by_type': summary_by_type, 'detail': rows},
            f, ensure_ascii=False, indent=2, default=json_default,
        )

    if not no_plots:
        plot_radar_overall_metrics(summary, output_dir)
        plot_half_violin_swarm(rows, output_dir)
        plot_pareto_front_bubble(summary, output_dir)
        plot_function_success(rows, output_dir)
        plot_range_vs_coverage(rows, output_dir)
        plot_summary_table(summary, output_dir)

    return summary, summary_by_type


def evaluate_all_methods(args):
    output_dir = resolve_project_path(args.output_dir)
    model_path = resolve_project_path(args.model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Device: {device}')
    print(f'Seed: {args.seed}')
    print('Building evaluation context...')
    context = build_eval_context(args.seed)
    print(f'Baseline routes: {context["baseline_routes"]}')

    dqn = load_dqn(model_path, device)
    scenarios = generate_eval_scenarios(context, args.scenarios_per_type, args.seed)
    print(f'Evaluating {len(scenarios)} scenarios ({args.scenarios_per_type} per type)...')

    rows = []
    total_runs = len(scenarios) * len(METHODS)
    run_idx = 0
    for scenario in scenarios:
        for method in METHODS:
            routes, meta = run_policy(context, scenario, method, dqn, device, args)
            rows.append(evaluate_method_result(context, scenario, method, routes, meta))
            run_idx += 1
        if (scenario['scenario_id'] + 1) % max(args.scenarios_per_type, 1) == 0:
            print(f'  completed scenario {scenario["scenario_id"] + 1}/{len(scenarios)} '
                  f'({run_idx}/{total_runs} method runs)')

    summary, _ = write_outputs(rows, output_dir, args.no_plots)
    print_core_summary(summary)
    print(f'Outputs written to: {output_dir}')
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate emergency replanning policies on balanced event scenarios.'
    )
    parser.add_argument(
        '--model-path', type=str,
        default='outputs/models/emergency_dqn_model.pt',
    )
    parser.add_argument('--scenarios-per-type', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='outputs/eval')
    parser.add_argument('--ga-pop', type=int, default=200)
    parser.add_argument('--ga-generations', type=int, default=300)
    parser.add_argument('--ga-patience', type=int, default=100)
    parser.add_argument('--no-plots', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    evaluate_all_methods(parse_args())
