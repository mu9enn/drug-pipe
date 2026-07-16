# Slime (Core)

This repository is a trimmed-down working copy of Slime for development. Non-core assets, docs, tests, and environment setup files were removed to keep the codebase focused.

## What is included

- Ray-based training pipeline with rollout/actor/critic components.
- Pluggable extensions under the plugins package.
- Lightweight launch scripts.

## Entry points

- [train.py](train.py): synchronous training loop.
- [train_async.py](train_async.py): asynchronous training loop.

Run the entry points with --help to see available options:

```bash
python train.py --help
python train_async.py --help
```

## Code layout

- [slime/](slime/): core library.
- [slime_plugins/](slime_plugins/): optional plugins and extensions.
- [scripts/](scripts/): lightweight run helpers.

## Notes

- CLI arguments are defined in [slime/utils/arguments.py](slime/utils/arguments.py).
- This repo intentionally excludes environment setup, docs, and tests.
