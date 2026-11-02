"""skrl optional dependency, lifecycle, and real PPO contracts."""
import os
import subprocess
import sys
import unittest
from unittest.mock import patch
from aidge_rl import Agent, Trainer, Runner, list_methods
from aidge_rl.adapters import SkrlAgent
from examples.evaluation import CounterEnv


class SkrlTests(unittest.TestCase):
    """Cover registration, errors and persistent backend ownership."""
    def test_lazy_registration(self):
        """Construction/discovery does not import optional frameworks."""
        subprocess.run([sys.executable, '-c',
            "import sys; from aidge_rl import Agent; "
            "Agent(package='skrl',method='PPO'); "
            "assert 'skrl' not in sys.modules and 'torch' not in sys.modules"], check=True)
        self.assertEqual(list_methods('SKRL'), ['PPO'])
        with self.assertRaises(ValueError):
            Agent(package='skrl', method='SAC')
        with self.assertRaises(RuntimeError):
            SkrlAgent().act([1])

    def test_training_errors(self):
        """Reject unsupported request paths before delegating any execution."""
        agent = SkrlAgent()
        with self.assertRaises(TypeError):
            Trainer(agent, CounterEnv()).train(1, unsupported=True)
        with patch('importlib.metadata.version', return_value='1.4.3'):
            with self.assertRaises(ImportError):
                Trainer(agent, CounterEnv()).train(1)
        agent.close()
        agent.close()
        with self.assertRaises(RuntimeError):
            Trainer(agent, CounterEnv()).train(1)

    @unittest.skipUnless(os.environ.get('AIDGE_SKRL_SMOKE') == '1',
                         'Install .[skrl] and set AIDGE_SKRL_SMOKE=1')
    def test_real_discrete_and_continuous(self):
        """Train, play a complete episode, save, and resume on actual Torch PPO."""
        import tempfile
        from pathlib import Path
        import gymnasium as gym
        import torch
        for name in ('CartPole-v1', 'Pendulum-v1'):
            env, evaluation = gym.make(name), gym.make(name)
            agent = Agent(package='skrl', method='PPO', seed=42,
                          training={'rollouts': 16, 'mini_batches': 2, 'learning_epochs': 1})
            try:
                rng = torch.get_rng_state()
                self.assertEqual(Trainer(agent, env).train(32).actual_steps, 32)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                model = agent.model
                optimizer = model.optimizer
                parameters = [v.detach().clone() for v in model.policy.parameters()]
                records = Runner(evaluation, agent).run(seed=43)
                self.assertTrue(records[-1].terminated or records[-1].truncated)
                self.assertTrue(all(evaluation.action_space.contains(t.action) for t in records))
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(parameters, model.policy.parameters())))
                self.assertEqual(Trainer(agent, env).train(1).actual_steps, 16)
                self.assertIs(agent.model, model)
                self.assertIs(model.optimizer, optimizer)
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(parameters, model.policy.parameters())))
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / 'agent.pt'
                    agent.save(path)
                    self.assertTrue(path.is_file())
                with self.assertRaises(ValueError):
                    Trainer(agent, evaluation).train(1)
            finally:
                agent.close()
                env.close()
                evaluation.close()
