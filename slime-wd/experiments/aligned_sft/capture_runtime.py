"""Record executable source, checkpoint bytes and installed runtime versions."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def capture(model, output):
    project = Path(__file__).resolve().parents[3]
    paths = [model / name for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json',
                                      'model.safetensors.index.json')]
    index = json.loads(paths[-1].read_text())
    paths += [model / name for name in sorted(set(index['weight_map'].values()))]
    versions = {}
    for name in ['torch', 'transformers', 'sglang', 'ray', 'megatron-core', 'transformer-engine', 'triton']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    sources = [project / 'data-pipe/pipeline/output_contracts.py',
               project / 'slime-wd/slime/slime/utils/mask_utils.py',
               project / 'slime-wd/slime/slime/rollout/sft_rollout.py',
               project / 'slime-wd/dsh-molbench/run_dsh_molbench.py',
               project / 'slime-wd/deepseek-harness/pnpm-lock.yaml']
    result = {'python': platform.python_version(), 'versions': versions, 'model': str(model),
              'model_config': json.loads((model / 'config.json').read_text()),
              'checkpoint_files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in paths},
              'sources': {str(p.relative_to(project)): sha(p) for p in sources},
              'gpu_inventory': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                                        '--format=csv,noheader'], text=True).splitlines()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        f.write(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    capture(args.model, args.output)
