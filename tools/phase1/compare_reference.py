"""Compare the frozen phase-1 numeric traces; log text is excluded explicitly."""
import argparse
import csv
import json
import math
import re
from pathlib import Path


def rows(path):
    with Path(path).open() as stream:
        return list(csv.reader(line for line in stream if re.match(r"^\d+,\d+,", line)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--session", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    expected, actual = rows(args.baseline), rows(args.candidate)
    integer_columns = set(range(11))
    if args.session:
        integer_columns.update(range(20, 24))
    failures = []
    compared = 0
    if len(expected) != 4800 or len(actual) != len(expected):
        failures.append({"row_counts": [len(expected), len(actual)], "expected": 4800})
    for index, (left, right) in enumerate(zip(expected, actual)):
        if len(left) != len(right):
            failures.append({"row": index, "column_counts": [len(left), len(right)]})
            continue
        for column, (a, b) in enumerate(zip(left, right)):
            compared += 1
            if column in integer_columns:
                equal = int(a) == int(b)
            else:
                x, y = float(a), float(b)
                equal = math.isfinite(x) and math.isfinite(y) and math.isclose(
                    x, y, rel_tol=1e-7, abs_tol=1e-9)
            if not equal:
                failures.append({"row": index, "column": column, "baseline": a, "candidate": b})
    report = {"passed": not failures, "rows": len(actual), "values": compared,
              "failure_count": len(failures), "first_failures": failures[:20],
              "absolute_tolerance": 1e-9, "relative_tolerance": 1e-7}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
