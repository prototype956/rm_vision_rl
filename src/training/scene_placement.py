"""Deterministic placement proposals, never a substitute for native admission.

The rectangles below deliberately over-cover the RMUL map's raised structures.
They only order candidates; support, collision, optics and continuous visibility
are still decided by the unchanged native reset. No controller is queried here.
"""
import copy
from functools import lru_cache
import math

import numpy as np

SAMPLER = "rmul_floor_clearance_v1"
POOL_SIZE = 32768
# Bevy world X/Z, from rm_simulator_2027/src/field.rs. Include bridges/openings
# in each deck footprint, as those are unsuitable for these ground-level tasks.
FLOOR = np.array([[-6., -4.], [6., 4.]])
BLOCKS = np.array([
    [[-3.5, -4.], [-1.5, 1.]],
    [[1.5, -1.], [3.5, 4.]],
    [[-6., -4.], [-5.25, -3.55]],
    [[5.25, 3.55], [6., 4.]],
])
BODY_MARGIN = .42
SIGHT_MARGIN = .20


def identity():
    from src.training.curriculum_v2 import digest
    return digest(dict(sampler=SAMPLER, pool_size=POOL_SIZE, floor=FLOOR.tolist(),
                       blocks=BLOCKS.tolist(), body_margin=BODY_MARGIN,
                       sight_margin=SIGHT_MARGIN))


def _halton(count, base):
    indices = np.arange(1, count + 1)
    values = np.zeros(count)
    factor = 1.
    while np.any(indices):
        factor /= base
        values += factor * (indices % base)
        indices //= base
    return values


def _intersects(start, end, lower, upper):
    """Vectorized closed 2-D segment/AABB intersection, including parallel rays."""
    delta = end - start
    parallel = np.abs(delta) < 1e-12
    outside = np.any(parallel & ((start < lower) | (start > upper)), axis=-1)
    divisor = np.where(parallel, 1., delta)
    first, last = (lower - start) / divisor, (upper - start) / divisor
    entry = np.max(np.where(parallel, -np.inf, np.minimum(first, last)), axis=-1)
    leave = np.min(np.where(parallel, np.inf, np.maximum(first, last)), axis=-1)
    return ~outside & (np.maximum(entry, 0.) <= np.minimum(leave, 1.))


def _reach(robot):
    motion = robot["motion"]
    if motion["kind"] not in ("translation", "combined") or motion["speed_m_s"] == 0:
        return np.zeros(2)
    yaw = motion.get("translation_heading_rad", robot.get("yaw_rad", 0.))
    return motion["amplitude_m"] * np.array([math.cos(yaw), -math.sin(yaw)])


def _sight_overlaps(points, block):
    """2-D convex sight strip vs expanded rectangle, used only for ordering."""
    lower, upper = block[0] - SIGHT_MARGIN, block[1] + SIGHT_MARGIN
    overlap = np.all((points.max(axis=1) >= lower) &
                     (points.min(axis=1) <= upper), axis=1)
    center, half = (lower + upper) / 2, (upper - lower) / 2
    # All possible edges include every convex-hull edge, even for degenerate
    # triangles/segments when one robot is static. Extra axes are safe as well.
    for a in range(points.shape[1]):
        for b in range(a):
            delta = points[:, a] - points[:, b]
            normal = np.column_stack((-delta[:, 1], delta[:, 0]))
            projected = np.sum(points * normal[:, None, :], axis=2)
            middle = np.sum(normal * center, axis=1)
            radius = np.sum(np.abs(normal) * half, axis=1)
            overlap &= (projected.max(axis=1) >= middle - radius) & (
                projected.min(axis=1) <= middle + radius)
    return overlap


@lru_cache(maxsize=2)
def _pool(seed, scenario_json):
    import json
    reference = json.loads(scenario_json)
    duel = reference["duel"]
    midpoint = (np.array(duel["controlled"]["position_xz_m"]) +
                np.array(duel["target"]["position_xz_m"])) / 2
    # Separate deterministic stream: never consume training/sensor RNG state.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x504F5345]))
    values = np.column_stack([_halton(POOL_SIZE, b) for b in (2, 3, 5, 7)])
    values = (values + rng.random(4)) % 1.
    yaw = values[:, 0] * math.tau
    cosine, sine = np.cos(yaw), np.sin(yaw)

    def rotate(vector):
        return np.column_stack((cosine * vector[0] + sine * vector[1],
                                -sine * vector[0] + cosine * vector[1]))

    centers, reaches = [], []
    for robot in duel.values():
        centers.append(rotate(np.array(robot["position_xz_m"]) - midpoint))
        reaches.append(rotate(_reach(robot)))
    endpoints = np.stack([p + sign * reach for p, reach in zip(centers, reaches)
                          for sign in (-1, 1)], axis=1)
    lower = FLOOR[0] + BODY_MARGIN - endpoints.min(axis=1)
    upper = FLOOR[1] - BODY_MARGIN - endpoints.max(axis=1)
    fits = np.all(lower <= upper, axis=1)
    shifts = lower + values[:, 1:3] * np.maximum(upper - lower, 0.)
    paths = [(p - r + shifts, p + r + shifts) for p, r in zip(centers, reaches)]
    body_clear, sight_clear = fits.copy(), fits.copy()
    sight_points = np.stack([v for path in paths for v in path], axis=1)
    for block in BLOCKS:
        for start, end in paths:
            body_clear &= ~_intersects(start, end, block[0] - BODY_MARGIN,
                                      block[1] + BODY_MARGIN)
        sight_clear &= ~_sight_overlaps(sight_points, block)
    # Retain broad fallback proposals: the hint can miss valid elevated optical
    # paths, and may never declare a task impossible. Halton order spreads poses
    # across bearings and positions rather than concentrating at a score maximum.
    tier = np.where(body_clear & sight_clear, 0, np.where(body_clear, 1, 2))
    order = np.argsort(tier, kind="stable")
    count = int(np.count_nonzero(tier == 0))
    return reference, yaw, centers, shifts, values[:, 3], order, count


def candidate_scene(descriptor, attempt):
    import json
    if not 0 <= attempt < 128:
        raise ValueError("placement candidate requires attempt in [0,128)")
    if descriptor.get("version") != 3 or descriptor.get("release_revision", 1) < 3:
        raise ValueError("clearance sampler requires revision 3")
    selection = descriptor.get("placement_search", {})
    if selection.get("sampler") != SAMPLER or selection.get("hints_sha256") != identity():
        raise ValueError("placement sampler identity mismatch")
    reference, yaws, centers, shifts, phases, order, count = _pool(
        descriptor["seed"], json.dumps(descriptor["scenario"], sort_keys=True))
    index = int(order[attempt])
    result = copy.deepcopy(descriptor)
    result.pop("admitted", None)
    result.pop("visibility_certificate", None)
    for name, center in zip(("controlled", "target"), centers):
        robot = result["scenario"]["duel"][name]
        source = reference["duel"][name]
        robot["position_xz_m"] = (center[index] + shifts[index]).tolist()
        robot["yaw_rad"] = math.remainder(source.get("yaw_rad", 0.) + yaws[index], math.tau)
        if "translation_heading_rad" in source["motion"]:
            robot["motion"]["translation_heading_rad"] = math.remainder(
                source["motion"]["translation_heading_rad"] + yaws[index], math.tau)
    # T1/T2 chassis facing stays independent of translation and sight bearing.
    # Other motions keep the rigidly transformed yaw because it sets their path.
    if descriptor["course"] in (7, 8):
        result["scenario"]["duel"]["target"]["yaw_rad"] = phases[index] * math.tau - math.pi
    result["attempt"] = attempt
    result["placement_search"] = dict(selection, coarse_pool_size=POOL_SIZE,
        preferred_candidates=count, proposal_index=index)
    return result
