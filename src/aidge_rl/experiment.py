"""Independent repeated evaluation across agent and environment factories."""
from dataclasses import dataclass
from statistics import mean, stdev
from .runner import Runner
from .trainer import Trainer


@dataclass(frozen=True)
class EpisodeResult:
    """A raw evaluation outcome, retaining its pairing and seed provenance."""

    agent: str
    environment: str
    repetition: int
    seed: int
    episode: int
    return_: float
    length: int
    terminated: bool
    truncated: bool


class Experiment:
    """Evaluate the Cartesian product of methods and environments.

    env factories take no arguments. Agent factories take (env, seed), returning
    a fresh or freshly loaded policy. In train_evaluate mode, Trainer trains
    it on a separate environment before evaluation. Repetitions are independent; policy
    parameters persist across evaluation episodes within a repetition. This
    experiment evaluates single-agent tasks; multi-agent scheduling lives in Runner.
    """

    def __init__(self, agents, environments, *, repetitions=3, episodes=5,
                 seed=0, episode_horizon=None, mode="evaluate", training_steps=None):
        """Store factories, seed schedule, and evaluate/train_evaluate mode."""
        if not agents or not environments or repetitions <= 0 or episodes <= 0:
            raise ValueError("Nonempty factories and positive counts required")
        self.agents, self.environments = dict(agents), dict(environments)
        self.repetitions, self.episodes = repetitions, episodes
        self.seed, self.episode_horizon = seed, episode_horizon
        if mode not in {"evaluate", "train_evaluate"}:
            raise ValueError("mode must be evaluate or train_evaluate")
        if mode == "train_evaluate" and (not isinstance(training_steps, int) or training_steps <= 0):
            raise ValueError("train_evaluate requires positive training_steps")
        self.mode, self.training_steps = mode, training_steps
        self.training_results = []
        self.results = []

    def run(self):
        """Evaluate fresh objects per pair/repetition, closing envs on failure."""
        self.results = []
        self.training_results = []
        for env_name, env_factory in self.environments.items():
            for agent_name, agent_factory in self.agents.items():
                for repetition in range(self.repetitions):
                    # Nonoverlapping blocks keep episode seeds reproducible and
                    # matched between methods on the same environment.
                    base_seed = self.seed + repetition * (self.episodes + 1)
                    env = env_factory()
                    try:
                        agent = agent_factory(env, base_seed)
                        if self.mode == "train_evaluate":
                            train_env = env_factory()
                            try:
                                if train_env is env:
                                    raise ValueError("Factories must create separate training/evaluation envs")
                                train_env.reset(seed=base_seed)
                                result = Trainer(agent, train_env).train(self.training_steps)
                                self.training_results.append((agent_name, env_name, repetition, result))
                            finally:
                                if train_env is not env and hasattr(train_env, "close"):
                                    train_env.close()
                        runner = Runner(env, agent, episode_horizon=self.episode_horizon)
                        for episode in range(self.episodes):
                            episode_seed = base_seed + episode + 1
                            runner.reset(seed=episode_seed)
                            total, length, final = 0.0, 0, None
                            while not runner.finished:
                                final = runner.step()
                                total += float(final.reward)
                                length += 1
                            self.results.append(EpisodeResult(
                                agent_name, env_name, repetition, episode_seed,
                                episode, total, length,
                                bool(final and final.terminated),
                                bool(final and final.truncated)))
                    finally:
                        if hasattr(env, "close"):
                            env.close()
        return list(self.results)

    def summary(self):
        """Aggregate repetition means, avoiding episode-level pseudoreplication.

        Standard deviation is sample SD across independent repetition means;
        it is None for a single repetition. Raw episode data remain available.
        """
        groups = {}
        for result in self.results:
            key = (result.agent, result.environment)
            groups.setdefault(key, {}).setdefault(result.repetition, []).append(result.return_)
        output = {}
        for key, repetitions in groups.items():
            values = [mean(returns) for returns in repetitions.values()]
            output[key] = {"mean_return": mean(values),
                           "std_return": stdev(values) if len(values) > 1 else None,
                           "repetitions": len(values),
                           "episodes": sum(map(len, repetitions.values()))}
        return output
