"""Compare pre-backend traces; fail on any difference, including whitespace/order."""
import argparse
import json
from pathlib import Path


def compare(left, right):
    a = [json.loads(line) for line in Path(left).read_text(encoding="utf-8").splitlines()]
    b = [json.loads(line) for line in Path(right).read_text(encoding="utf-8").splitlines()]
    if not a or len(a) != len(b):
        raise AssertionError("Empty trace or different number of cases")
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            fields = sorted(k for k in x.keys() | y.keys() if x.get(k) != y.get(k))
            raise AssertionError(f"Case {i} differs: {fields}")
    return len(a)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    args = parser.parse_args()
    print(f"Identical: {compare(args.left, args.right)} permutation cases")
