"""Compare two policies on two tasks with repeated, reproducible evaluation."""
import random
from aidge_rl import ConstantAgent, Experiment


class CounterEnv:
    """Tiny structured-observation task with configurable reward scale."""

    def __init__(self, scale=1):
        """Configure reward scaling without initializing an episode."""
        self.scale = scale
        self.observation_space = None  # Replace with a Space for backend validation.
        self.action_space = None

    def reset(self, *, seed=None, options=None):
        """Sample an episode target from an owned generator."""
        self.rng = random.Random(seed)
        self.target = self.rng.randrange(2)
        self.t = 0
        return {"target": self.target}, None

    def step(self, action):
        """Reward a matching choice for three rounds."""
        self.t += 1
        return {"target": self.target}, self.scale * (action == self.target), self.t == 3, False, None

    def close(self):
        """No resources are owned by this example."""


if __name__ == "__main__":
    experiment = Experiment(
        agents={"zero": lambda env, seed: ConstantAgent(0),
                "one": lambda env, seed: ConstantAgent(1)},
        environments={"normal": CounterEnv, "double": lambda: CounterEnv(2)},
        repetitions=4, episodes=5,
    )
    experiment.run()
    print(experiment.summary())
