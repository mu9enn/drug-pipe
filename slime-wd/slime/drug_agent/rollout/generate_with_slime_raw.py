from __future__ import annotations

from slime.utils.types import Sample

from drug_agent.protocol.slime_raw import PROFILE_NAME
from drug_agent.rollout.generate_with_drug_agent import generate_profile


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    """Run the strict, model-neutral prompt/tool/observation evaluation loop."""

    return await generate_profile(
        args,
        sample,
        sampling_params,
        evaluation=evaluation,
        rollout_profile=PROFILE_NAME,
    )
