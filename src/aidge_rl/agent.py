"""Small structural agent contract and baseline policies, without backend imports."""
from typing import Any, Protocol


class Policy(Protocol):
    """Action selection contract; reset/observe/learn are optional capabilities."""

    def act(self, obs: Any, *, info: Any = None, stream_id: Any = None) -> Any:
        """Select an action for an independent interaction stream."""
        ...


class ConstantAgent:
    """Return a configured action without changing its representation."""

    def __init__(self, action):
        """Store the action, which callers must treat as read-only."""
        self.action = action

    def act(self, obs, *, info=None, stream_id=None):
        """Return the configured action."""
        return self.action


class RandomAgent:
    """Sample an arbitrary Gymnasium-compatible action space."""

    def __init__(self, action_space):
        """Retain the supplied space without enumerating it."""
        self.action_space = action_space

    def reset(self, *, stream_id=None, seed=None):
        """Seed action sampling when an explicit seed is supplied."""
        if seed is not None:
            self.action_space.seed(seed)

    def act(self, obs, *, info=None, stream_id=None):
        """Sample directly from the action space."""
        return self.action_space.sample()
