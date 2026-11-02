"""Optional state/observation capability, independent of Gymnasium inheritance."""
from typing import Protocol, Any


class Environment(Protocol):
    """Single-agent Gymnasium-shaped interaction contract."""

    observation_space: Any
    action_space: Any

    def reset(self, *, seed=None, options=None):
        """Return an initial observation and optional information."""
        ...

    def step(self, action):
        """Return observation, reward, termination, truncation, and information."""
        ...


class StateEnv:
    """Optional mixin exposing a replaceable state-to-observation model.

    Concrete systems implement reset/step and define their spaces. It is also
    possible to combine this mixin with gymnasium.Env when Gymnasium is installed.
    """

    def __init__(self, observation_model=None):
        """Use an identity observation unless a callable model is provided."""
        self.observation_model = observation_model

    def observation(self, state, *, rng, agent_id=None):
        """Observe state, optionally selecting an agent-specific sensor model."""
        if self.observation_model is None:
            return state
        return self.observation_model(state, rng=rng, agent_id=agent_id)
