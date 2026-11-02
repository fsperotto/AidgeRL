"""Experience records; values are borrowed unless a Runner copy function is supplied."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Transition:
    """One executed action and its outcome, including final episode flags."""

    obs: Any
    action: Any
    reward: float
    next_obs: Any
    terminated: bool
    truncated: bool
    info: Any = None
    agent_id: Any = None
