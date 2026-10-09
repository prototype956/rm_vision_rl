"""Selection costs from pre-action vision estimates, never simulator truth.

All times are seconds. Costs apply to confirmed controller actions. The policy's
shared selection memory supplies the previous distinct slot for A->B->A costs.
"""
import math


def selection_costs(info, config):
    parts = dict(base_switch_penalty=0.0, return_penalty=0.0, facing_penalty=0.0)
    details = dict(facing_active=False, facing_best=0.0, facing_selected=0.0, facing_gap=0.0)
    ctx = info.get('switch_reward_context')
    if (ctx is None or ctx['track_generation'] != info.get('track_generation')
            or ctx.get('generation_changed', False) or not ctx['tracking_reliable']):
        return parts, details
    valid = ctx['candidate_valid']
    angles = ctx['facing_impact_rad']
    age = ctx['selected_slot_age_s']
    if (len(valid) != 4 or len(angles) != 4 or not math.isfinite(age) or age < 0
            or any(not math.isfinite(a) for a, v in zip(angles, valid) if v)):
        raise ValueError('invalid selection reward context')
    slot = info['selected_slot']
    if slot not in range(-1, 4):
        raise ValueError('invalid executed slot')
    if info.get('slot_switched') and ctx['previous_slot'] >= 0 and ctx['previous_valid']:
        parts['base_switch_penalty'] = config['switch_cost']
        if slot >= 0 and slot == ctx['return_slot'] and valid[slot]:
            parts['return_penalty'] = config['return_cost'] * max(
                0.0, 1.0 - age / config['return_window_s']) ** 2
    if any(valid):
        qualities = [max(0.0, math.cos(a)) ** 2 if v else 0.0 for a, v in zip(angles, valid)]
        best = max(qualities)
        selected = qualities[slot] if slot >= 0 else 0.0
        gap = max(0.0, best - selected - config['facing_margin'])
        parts['facing_penalty'] = config['facing_rate_per_s'] * .01 * gap
        details.update(facing_active=True, facing_best=best, facing_selected=selected, facing_gap=gap)
    return parts, details
