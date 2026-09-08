"""Summarize bounded Foxglove control captures without confusing idle and active cost."""
import argparse
from collections import Counter
import json
from pathlib import Path


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    at = (len(values) - 1) * fraction
    lower = int(at)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (at - lower)


def describe(rows):
    cost = [row['runtime']['control_compute_time_us'] for row in rows]
    return {
        'samples': len(rows),
        'states': dict(Counter(row['tracker_state'] for row in rows)),
        'reject_reasons': dict(Counter(row['fire']['reject_reason'] for row in rows)),
        'external_enabled': sum(row['talos']['external_control_enabled'] for row in rows),
        'published_valid': sum(row['command']['published_valid'] for row in rows),
        'local_shot_accepted': sum(row['fire']['shot_accepted'] for row in rows),
        'fire_high_samples': sum(row['fire']['pulse'] for row in rows),
        'compute_us': {f'p{p}': percentile(cost, p / 100) for p in (50, 95, 99)},
        'compute_max_us': max(cost, default=None),
        'compute_over_10ms': sum(value > 10000 for value in cost),
        'period_over_11ms': sum(row['runtime']['control_period_s'] > .011 for row in rows),
        'deadline_over_1ms': sum(row['runtime']['deadline_lateness_us'] > 1000 for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.capture).read_text().splitlines()]
    active = [row for row in rows if row['talos']['external_control_enabled'] and
              row['tracker_state'] == 'tracking' and row['command']['published_valid']]
    report = {'capture': args.capture, 'all': describe(rows), 'active_tracking': describe(active)}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
