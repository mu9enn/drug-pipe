from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Print first N samples from a jsonl file")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--num", type=int, default=3)
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.num:
                break
            print(f"===== sample {i} =====")
            obj = json.loads(line)
            print(json.dumps(obj, ensure_ascii=False, indent=2)[:5000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
