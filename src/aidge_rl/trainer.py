"""Backend-independent training coordination and interaction accounting."""
from dataclasses import dataclass
from .runner import Runner


@dataclass(frozen=True)
class TrainingResult:
    """Requested and measured environment interactions for one training call."""

    requested_steps: int
    actual_steps: int


class Trainer:
    """Coordinate an agent training capability or native Runner-based updates.

    train(env, total_steps, **options) owns backend execution when available.
    Otherwise native agents require observe() and either learn() or an explicit
    update callback. Environments remain caller-owned.
    """

    def __init__(self, agent, env, *, update=None, episode_horizon=None):
        """Retain the same mutable agent that Runner will later evaluate."""
        self.agent, self.env = agent, env
        self.update, self.episode_horizon = update, episode_horizon

    def train(self, total_steps, **options):
        """Train using capabilities, never backend-specific type checks."""
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")
        if hasattr(self.agent, "train"):
            if self.update is not None or self.episode_horizon is not None:
                raise ValueError("Runner update/horizon options do not apply to delegated training")
            result = self.agent.train(self.env, total_steps, **options)
            if not isinstance(result, TrainingResult):
                raise TypeError("Delegated train must return TrainingResult")
            return result
        seed = options.pop("seed", None)
        if options:
            raise TypeError(f"Unsupported native training options: {tuple(options)}")
        if not hasattr(self.agent, "observe") or (self.update is None and not hasattr(self.agent, "learn")):
            raise TypeError("Agent has no supported training capability")
        update = self.update or (lambda agent, transition: agent.learn())
        runner = Runner(self.env, self.agent, training=True, update=update,
                        episode_horizon=self.episode_horizon)
        for step in range(total_steps):
            if not runner.ready or runner.finished:
                runner.reset(seed=seed if step == 0 else None)
                if runner.finished:
                    raise ValueError("Cannot train on an environment with no active agents")
            runner.step()
        return TrainingResult(total_steps, total_steps)
