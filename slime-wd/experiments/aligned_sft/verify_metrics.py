"""Fail promotion unless every expected optimizer step has finite measured metrics."""
import json
import math
import sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path(sys.argv[1])
expected = int(sys.argv[2])
events = EventAccumulator(str(root), size_guidance={'scalars': 0}).Reload()
summary = {}
for tag in ('train/loss', 'train/grad_norm'):
    values = events.Scalars(tag)
    assert {v.step for v in values} == set(range(expected)), (tag, len(values), expected)
    assert all(math.isfinite(v.value) and v.value > 0 for v in values), tag
    summary[tag] = {'count': len(values), 'min': min(v.value for v in values), 'max': max(v.value for v in values)}
(root / 'verified.json').write_text(json.dumps(summary, indent=2) + '\n')
