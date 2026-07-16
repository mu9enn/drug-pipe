from __future__ import annotations

from typing import Protocol, Any


class PairwiseAdjudicator(Protocol):
    model_name: str

    def adjudicate(self, payload: dict[str, Any]) -> dict[str, Any]: ...
