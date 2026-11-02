"""Uniform construction, training ownership, and experiment isolation tests."""
import unittest
from aidge_rl import Agent, Trainer, Experiment, TrainingResult, ConstantAgent
from examples.evaluation import CounterEnv
from tests.test_core import RecordingAgent


class FakeSB3:
    """Minimal backend with observable construction and continuation behavior."""
    def __init__(self, policy, env, **options):
        """Retain native constructor arguments."""
        self.env, self.options, self.num_timesteps = env, options, 0
    def set_env(self, env):
        """Rebind training environment."""
        self.env = env
    def learn(self, total_timesteps, **kwargs):
        """Overshoot a rollout to verify measured interaction accounting."""
        self.num_timesteps += total_timesteps + 1
    def predict(self, obs, **kwargs):
        """Expose a policy usable after training."""
        return 0, None


class LearningAgent(RecordingAgent):
    """Native agent using Runner ingestion and per-transition updates."""
    def learn(self):
        """Increment learned state independent of episode resets."""
        self.action += 1


class TrainingTests(unittest.TestCase):
    """Test the same public API across execution strategies."""
    def test_delegated_and_continued(self):
        """Defer model creation and keep the trained object for prediction."""
        agent = Agent(package="SB3", method=FakeSB3, learning_rate=0.1)
        with self.assertRaises(RuntimeError):
            agent.act({})
        self.assertEqual(Trainer(agent, CounterEnv()).train(3), TrainingResult(3, 4))
        model = agent.model
        self.assertEqual(Trainer(agent, CounterEnv()).train(2).actual_steps, 3)
        self.assertIs(agent.model, model)
        self.assertEqual(agent.act({}), 0)

    def test_native_and_validation(self):
        """Native classes train without any package-specific dispatch."""
        agent = Agent(method=LearningAgent)
        self.assertEqual(Trainer(agent, CounterEnv()).train(5).actual_steps, 5)
        self.assertEqual(len(agent.transitions), 5)
        self.assertEqual(agent.action, 5)
        for options in ({"method": "PPO"}, {"package": "unknown", "method": "PPO"}):
            with self.assertRaises(ValueError):
                Agent(**options)
        with self.assertRaises(TypeError):
            Trainer(ConstantAgent(0), CounterEnv()).train(2)

    def test_train_evaluate_isolation(self):
        """Fresh paired environments train first and evaluate without updating."""
        instances, agents = [], []
        def env_factory():
            """Track separately owned environments."""
            env = CounterEnv()
            env.closed = False
            env.close = lambda: setattr(env, "closed", True)
            instances.append(env)
            return env
        def agent_factory(env, seed):
            """Retain policies to check evaluation does not add experience."""
            agent = Agent(method=LearningAgent)
            agents.append(agent)
            return agent
        exp = Experiment({"native": agent_factory}, {"task": env_factory},
                         repetitions=2, episodes=2, mode="train_evaluate", training_steps=4)
        self.assertEqual(len(exp.run()), 4)
        self.assertEqual(len(instances), 4)
        self.assertTrue(all(e.closed for e in instances))
        self.assertTrue(all(len(a.transitions) == 4 for a in agents))
        self.assertEqual(len(exp.training_results), 2)
