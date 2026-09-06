from __future__ import annotations

import copy
import logging
import os
import random
from pathlib import Path

from drug_agent.toolrl.trajectory_batching import validate_packed_decision_batches


logger = logging.getLogger(__name__)


def _metadata(sample):
    return sample.metadata if isinstance(sample.metadata, dict) else {}


class TrajectoryBatchDataSource:
    """Return fixed-size rollout batches made only of complete trajectories."""

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
        self.shuffle_batches = bool(args.rollout_shuffle)
        self.batches = validate_packed_decision_batches(
            list(self.dataset.samples),
            metadata_of=_metadata,
            rollout_batch_size=args.rollout_batch_size,
        )
        if not self.batches:
            raise ValueError("trajectory ToolRL dataset contains no rollout batches")
        self.batch_offset = 0
        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self._set_epoch(0)

    def _set_epoch(self, epoch_id: int) -> None:
        self.epoch_id = epoch_id
        self.batch_order = list(range(len(self.batches)))
        if self.shuffle_batches:
            random.Random(self.args.rollout_seed + epoch_id).shuffle(self.batch_order)
        self.batch_offset = 0

    def get_samples(self, num_samples):
        if num_samples != self.args.rollout_batch_size:
            raise ValueError(
                "trajectory sampling requires over_sampling_batch_size == rollout_batch_size; "
                f"got {num_samples} and {self.args.rollout_batch_size}"
            )
        if self.batch_offset == len(self.batch_order):
            self._set_epoch(self.epoch_id + 1)
        batch = self.batches[self.batch_order[self.batch_offset]]
        self.batch_offset += 1

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
                "partial-rollout buffering is incompatible with trajectory-atomic sampling"
            )

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return
        import torch

        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "batch_offset": self.batch_offset,
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
        self._set_epoch(int(state.get("epoch_id", 0)))
        self.batch_offset = int(state.get("batch_offset", 0))
        self.sample_group_index = int(state.get("sample_group_index", 0))
        self.sample_index = int(state.get("sample_index", 0))

    def __len__(self):
        return len(self.batches) * self.args.rollout_batch_size
