import copy
import math
import random

import numpy as np

from emergency.simulator import EmergencySimulator
from scripts import demo_emergency_only as schematic


def _bare_simulator(hotspots):
    sim = object.__new__(EmergencySimulator)
    sim.original_hotspots = copy.deepcopy(hotspots)
    return sim


def _regions_do_not_overlap(regions):
    for i, a in enumerate(regions):
        for b in regions[i + 1:]:
            dist = math.hypot(a[1] - b[1], a[2] - b[2])
            assert dist >= a[3] + b[3]


def test_s1_schematic_generates_two_nonoverlapping_shifted_targets():
    random.seed(42)
    np.random.seed(42)
    sim = _bare_simulator(schematic.UNIFIED_HOTSPOTS)

    emergency = schematic.generate_s1_two(sim)

    assert emergency["type"] == "S1"
    assert len(emergency["shifts"]) == 2

    shifted = {int(s["id"]): s for s in emergency["shifts"]}
    final_regions = []
    for target in schematic.UNIFIED_HOTSPOTS:
        tid = int(target["id"])
        x, y = target["center_km"]
        if tid in shifted:
            x += shifted[tid]["dx"]
            y += shifted[tid]["dy"]
        final_regions.append((tid, x, y, target["radius_km"]))
    _regions_do_not_overlap(final_regions)


def test_s2_schematic_generates_three_nonoverlapping_new_targets():
    random.seed(43)
    np.random.seed(43)
    sim = _bare_simulator(schematic.UNIFIED_HOTSPOTS)

    emergency = schematic.generate_s2_three(sim)

    assert emergency["type"] == "S2"
    assert len(emergency["new_targets"]) == 3

    regions = [
        (int(t["id"]), t["center_km"][0], t["center_km"][1], t["radius_km"])
        for t in schematic.UNIFIED_HOTSPOTS + emergency["new_targets"]
    ]
    _regions_do_not_overlap(regions)


def test_s3_schematic_nfz_does_not_overlap_hotspots():
    random.seed(44)
    np.random.seed(44)
    sim = _bare_simulator(schematic.UNIFIED_HOTSPOTS)

    emergency = schematic.generate_s3_clear(sim, schematic.build_simple_routes())

    assert emergency["type"] == "S3"
    assert not schematic.nfz_overlaps_hotspots(emergency)


def test_compute_cost_for_assignment_returns_core_metrics():
    hotspots = [
        {
            "id": 0,
            "center_km": (10.0, 0.0),
            "radius_km": 1.0,
            "weight": 1.0,
            "spiral_dist": 1000.0,
            "spiral_time": 20.0,
        },
        {
            "id": 1,
            "center_km": (20.0, 0.0),
            "radius_km": 1.0,
            "weight": 0.5,
            "spiral_dist": 1000.0,
            "spiral_time": 20.0,
        },
    ]
    sim = EmergencySimulator(
        hotspots,
        num_uavs=4,
        uav_base_km=(0.0, 0.0),
        spiral_cfg={},
        prob_grid=None,
        X_km=None,
        Y_km=None,
    )

    cost = sim.compute_cost_for_assignment(
        [[0], [1], [], []],
        hotspots,
        active_uavs=[True, True, True, True],
        consumed_ranges=[0.0, 0.0, 0.0, 0.0],
        uav_positions=np.array([(0.0, 0.0)] * 4, dtype=float),
    )

    for key in [
        "J_max",
        "J_sum",
        "weighted_arrival_sum",
        "route_dists_km",
        "total_ranges",
        "range_violations",
        "nfz_violations",
    ]:
        assert key in cost

    assert cost["range_violations"] == 0
    assert float(np.sum(cost["route_dists_km"])) > 0
