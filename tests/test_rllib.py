"""RLlib adapter contracts and an explicitly opt-in real-backend smoke test."""
import os
import subprocess
import sys
import unittest
from aidge_rl import Agent, Trainer, list_methods, list_packages
from aidge_rl.adapters import RLlibAgent
from examples.evaluation import CounterEnv


class Algorithm:
    """Minimal RLlib training accounting and resource-ownership fixture."""
    def __init__(self):
        """Retain persistent sampled counters."""
        self.steps, self.stopped = 0, False
    def train(self):
        """Complete fixed-size sampling iterations."""
        self.steps += 8
        return {"num_env_steps_sampled_lifetime": self.steps}
    def stop(self):
        """Record shutdown without touching global Ray."""
        self.stopped = True


class RLlibTests(unittest.TestCase):
    """Validate optional registration and delegated training semantics."""
    def test_lazy_import(self):
        """Importing/constructing AidgeRL does not load Ray or Torch."""
        subprocess.run([sys.executable, '-c',
            "import sys; from aidge_rl import Agent; "
            "a=Agent(package='rllib', method='PPO'); "
            "assert 'ray' not in sys.modules and 'torch' not in sys.modules"], check=True)
        self.assertIn('rllib', list_packages())
        self.assertEqual(list_methods('RLlib'), ['PPO'])
        with self.assertRaises(ValueError):
            Agent(package='rllib', method='DQN')

    def test_budget_continuation_and_close(self):
        """Count actual environment steps and preserve caller-owned resources."""
        agent = RLlibAgent()
        env = CounterEnv()
        model = Algorithm()
        agent.model, agent._env = model, env
        self.assertEqual(Trainer(agent, env).train(5).actual_steps, 8)
        self.assertEqual(Trainer(agent, env).train(10).actual_steps, 16)
        with self.assertRaises(ValueError):
            Trainer(agent, CounterEnv()).train(1)
        with self.assertRaises(TypeError):
            Trainer(agent, env).train(1, unsupported=True)
        agent.close()
        agent.close()
        self.assertTrue(model.stopped)
        with self.assertRaises(RuntimeError):
            Trainer(agent, env).train(1)

    def test_experiment_closes_owned_agent(self):
        """Backend resources are released after repeated evaluation."""
        from aidge_rl import Experiment, ConstantAgent
        created = []
        def factory(env, seed):
            """Track an owned agent with a resource cleanup hook."""
            agent = ConstantAgent(0)
            agent.closed = False
            agent.close = lambda: setattr(agent, 'closed', True)
            created.append(agent)
            return agent
        Experiment({'constant': factory}, {'counter': CounterEnv},
                   repetitions=2, episodes=1).run()
        self.assertTrue(all(agent.closed for agent in created))

    def test_missing_metrics(self):
        """Never substitute iteration counts for missing environment counters."""
        agent = RLlibAgent()
        env = CounterEnv()
        agent.model, agent._env = Algorithm(), env
        agent.model.train = lambda: {}
        with self.assertRaises(RuntimeError):
            Trainer(agent, env).train(1)

    @unittest.skipUnless(os.environ.get('AIDGE_RLLIB_SMOKE') == '1',
                         'Set AIDGE_RLLIB_SMOKE=1 after installing .[rllib]')
    def test_real_ppo(self):
        """Exercise actual discrete and continuous training/inference locally."""
        import gymnasium as gym
        from aidge_rl import Runner
        for name in ('CartPole-v1', 'Pendulum-v1'):
            env, evaluation = gym.make(name), gym.make(name)
            agent = Agent(package='rllib', method='PPO', seed=42,
                          training={'train_batch_size_per_learner': 128,
                                    'minibatch_size': 64, 'num_epochs': 1})
            try:
                self.assertGreaterEqual(Trainer(agent, env).train(128).actual_steps, 128)
                record = Runner(evaluation, agent).run(max_steps=1, seed=43)[0]
                self.assertTrue(evaluation.action_space.contains(record.action))
                self.assertGreaterEqual(Trainer(agent, env).train(1).actual_steps, 1)
            finally:
                agent.close()
                env.close()
                evaluation.close()
