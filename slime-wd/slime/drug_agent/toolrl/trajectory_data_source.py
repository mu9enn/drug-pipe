from __future__ import annotations

import copy
import logging
import math
import os
from pathlib import Path

from drug_agent.toolrl.trajectory_batching import validate_trajectory_order


logger = logging.getLogger(__name__)


def _metadata(sample):
    return sample.metadata if isinstance(sample.metadata, dict) else {}


class TrajectoryBatchDataSource:
    """Read selected decisions continuously in canonical trajectory order."""

    def __init__(self, args):
        from slime.utils.data import Dataset
        from slime.utils.processing_utils import load_processor, load_tokenizer

        self.args = args
        if not args.rollout_global_dataset or args.prompt_data is None:
            raise ValueError("trajectory ToolRL sampling requires --prompt-data")
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        if args.dump_details is not None:
            tokenizer.save_pretrained(Path(args.dump_details) / "tokenizer")
            if processor:
                processor.save_pretrained(Path(args.dump_details) / "processor")
        self.dataset = Dataset(
            args.prompt_data,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.rollout_max_prompt_len,
            prompt_key=args.input_key,
            multimodal_keys=args.multimodal_keys,
            label_key=args.label_key,
            metadata_key=args.metadata_key,
            tool_key=args.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
            seed=args.rollout_seed,
        )
        if bool(args.rollout_shuffle):
            raise ValueError("v8 ToolRL canonical traversal requires omitting --rollout-shuffle")
        self.samples = validate_trajectory_order(list(self.dataset.samples), metadata_of=_metadata)
        if not self.samples:
            raise ValueError("trajectory ToolRL dataset contains no decisions")
        self.cursor = 0
        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0

    def get_samples(self, num_samples):
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        batch = []
        for _ in range(num_samples):
            batch.append(self.samples[self.cursor])
            self.cursor += 1
            if self.cursor == len(self.samples):
                self.cursor = 0
                self.epoch_id += 1

        samples = []
        for prompt_sample in batch:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples

    def add_samples(self, samples):
        if samples:
            raise RuntimeError(
                "fixed canonical traversal does not accept generated-sample buffering"
            )

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return
        import torch

        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "cursor": self.cursor,
                "epoch_id": self.epoch_id,
                "sample_group_index": self.sample_group_index,
                "sample_index": self.sample_index,
            },
            path,
        )

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset or self.args.load is None:
            return
        import torch

        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info("trajectory data-source state does not exist: %s", path)
            return
        state = torch.load(path)
        self.epoch_id = int(state.get("epoch_id", 0))
        self.cursor = int(state.get("cursor", 0))
        if not 0 <= self.cursor < len(self.samples):
            raise ValueError(f"saved ToolRL cursor is out of range: {self.cursor}")
        self.sample_group_index = int(state.get("sample_group_index", 0))
        self.sample_index = int(state.get("sample_index", 0))

    def __len__(self):
        # Slime derives num_rollouts_per_epoch with integer division and each
        # rollout must contain exactly rollout_batch_size groups. Expose the
        # padded training-side length so the final selected decisions are not
        # silently skipped; get_samples handles the boundary by continuing at
        # the deterministic start of the next epoch.
        size = int(self.args.rollout_batch_size)
        return math.ceil(len(self.samples) / size) * size
