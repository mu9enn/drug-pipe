"""Build a reviewable MolClaw patch; vendor the exact shared contract, not a fork."""
import difflib
import hashlib
import json
from pathlib import Path

project = Path(__file__).resolve().parents[3]
here = Path(__file__).resolve().parent
benchmark = project / 'slime-wd/molbench'
commit = '180abb7679ffc5b1ca9974e7390e141d8098f642'
changes = []
for source in sorted((benchmark / 'aligned/data').rglob('*.csv')):
    relative = source.relative_to(benchmark / 'aligned')
    original = benchmark / 'upstream' / commit / relative
    changes.append((str(Path('molbench') / relative), original.read_bytes().decode('utf-8'), source.read_bytes().decode('utf-8')))
contract = project / 'data-pipe/pipeline/output_contracts.py'
changes.extend([
    ('molbench/eval/output_contracts.py', '', contract.read_text()),
    ('molbench/eval/score_json.py', '', (here / 'score_json.py').read_text()),
])
parts = []
for name, old, new in changes:
    for line in difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                    fromfile='a/' + name if old else '/dev/null', tofile='b/' + name):
        parts.append(line if line.endswith('\n') else line + '\n\\ No newline at end of file\n')
patch = ''.join(parts)
(here / 'molclaw-json-contract-v4.patch').write_text(patch)
(here / 'manifest.json').write_text(json.dumps({
    'upstream_commit': commit,
    'shared_contract_sha256': hashlib.sha256(contract.read_bytes()).hexdigest(),
    'patch_sha256': hashlib.sha256(patch.encode()).hexdigest(),
    'scoring_math': 'unchanged eval_runner.py',
}, indent=2) + '\n')
