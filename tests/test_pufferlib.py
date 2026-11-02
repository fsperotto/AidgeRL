"""PufferLib adapter contract tests and optional real-backend training smoke."""
import os
import signal
import subprocess
import sys
import unittest
from unittest.mock import patch
from aidge_rl import Agent, Trainer, Runner, list_methods
from aidge_rl.adapters import PufferLibAgent
from examples.evaluation import CounterEnv


class PufferLibTests(unittest.TestCase):
    """Check lifecycle without requiring optional dependencies."""

    def test_lazy_registration(self):
        """Discovery/construction must not import PufferLib or Torch."""
        subprocess.run([sys.executable, '-c',
            "import sys; from aidge_rl import Agent; "
            "Agent(package='pufferlib',method='PPO'); "
            "assert 'pufferlib' not in sys.modules and 'torch' not in sys.modules"], check=True)
        self.assertEqual(list_methods('PufferLib'), ['PPO'])
        with self.assertRaises(ValueError):
            Agent(package='pufferlib', method='SAC')

    def test_import_restores_interrupt_handler(self):
        """Do not retain upstream's process-exiting import-time signal hook."""
        handler = signal.getsignal(signal.SIGINT)
        def import_backend(name):
            """Simulate the upstream module's import side effect."""
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            return 'backend'
        with patch('importlib.metadata.version', return_value='3.0.0'), patch(
                'importlib.import_module', side_effect=import_backend):
            self.assertEqual(PufferLibAgent._load_backend(), 'backend')
        self.assertEqual(signal.getsignal(signal.SIGINT), handler)
        with patch('importlib.metadata.version', return_value='5.0.0'):
            with self.assertRaises(ImportError):
                PufferLibAgent._load_backend()

    def test_training_continuation_and_cleanup(self):
        """Count executed transitions, excluding backend reset samples."""
        agent = PufferLibAgent()
        env = CounterEnv()
        class Backend:
            """Backend fixture whose rollout counter differs from real steps."""
            config = {}
            global_step = 0
            def evaluate(self):
                """Collect eight simulator steps plus a reset sample."""
                agent._steps += 8
                self.global_step += 9
            def train(self):
                """Retain backend state without reinitialization."""
            def close(self):
                """Release owned buffers."""
            class utilization:
                """Minimal monitoring-thread cleanup contract."""
                @staticmethod
                def stop():
                    """Stop monitoring."""
                @staticmethod
                def join(timeout):
                    """Wait for monitoring to exit."""
        agent.backend, agent._env = Backend(), env
        self.assertEqual(Trainer(agent, env).train(5).actual_steps, 8)
        self.assertEqual(Trainer(agent, env).train(9).actual_steps, 16)
        with self.assertRaises(ValueError):
            Trainer(agent, CounterEnv()).train(1)
        agent.close()
        agent.close()
        with self.assertRaises(RuntimeError):
            Trainer(agent, env).train(1)

    @unittest.skipUnless(os.environ.get('AIDGE_PUFFER_SMOKE') == '1',
                         'Install .[pufferlib] and set AIDGE_PUFFER_SMOKE=1')
    def test_real_cartpole(self):
        """Train actual PuffeRL, play an episode, then continue the same policy."""
        import gymnasium as gym
        env, evaluation = gym.make('CartPole-v1'), gym.make('CartPole-v1')
        agent = Agent(package='pufferlib', method='PPO', seed=42,
                      training={'batch_size': 64, 'minibatch_size': 32,
                                'bptt_horizon': 8, 'update_epochs': 1})
        try:
            self.assertGreaterEqual(Trainer(agent, env).train(64).actual_steps, 64)
            model = agent.model
            records = Runner(evaluation, agent).run(seed=43)
            self.assertTrue(records[-1].terminated or records[-1].truncated)
            self.assertTrue(all(evaluation.action_space.contains(t.action) for t in records))
            self.assertGreaterEqual(Trainer(agent, env).train(1).actual_steps, 1)
            self.assertIs(agent.model, model)
        finally:
            agent.close()
            env.close()
            evaluation.close()
