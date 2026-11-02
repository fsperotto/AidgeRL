"""Optional PufferLib 3.0.0 Python PPO-variant adapter.

Training delegates to PuffeRL, without copying its optimization algorithm.
The first supported boundary is one borrowed Gymnasium environment, flat Box
observations, and Discrete actions on CPU. Upstream reward clipping and
termination/truncation handling are preserved, not silently rewritten.
"""
import importlib
import importlib.metadata
import signal
import threading
import warnings


class PufferLibAgent:
    """Lazy feed-forward PuffeRL integration retaining one trained policy."""

    def __init__(self, *, method='PPO', observation_space=None, action_space=None,
                 seed=42, hidden_size=128, training=None, deterministic=True):
        """Store constructor options without importing optional frameworks."""
        if method != 'PPO':
            raise ValueError("PufferLib currently supports method='PPO' only")
        self.observation_space, self.action_space = observation_space, action_space
        self.seed, self.hidden_size = seed, hidden_size
        self.training_options = dict(training or {})
        self.deterministic = deterministic
        self.model = self.backend = self._vecenv = self._env = None
        self._steps = 0
        self._closed = False

    @staticmethod
    def _load_backend():
        """Load the pinned Python backend without retaining its SIGINT handler.

        Upstream installs a process-exiting SIGINT handler at import time. Require
        main-thread initialization so we can restore the caller's handler, and
        contain the import's warning-filter changes as well.
        """
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('Initialize PufferLib training on the main thread')
        try:
            version = importlib.metadata.version('pufferlib')
        except importlib.metadata.PackageNotFoundError as exc:
            raise ImportError("Install the optional extra: pip install 'aidge-rl[pufferlib]'") from exc
        if version != '3.0.0':
            raise ImportError(f'PufferLib 3.0.0 is required; found {version}')
        handler = signal.getsignal(signal.SIGINT)
        try:
            with warnings.catch_warnings():
                return importlib.import_module('pufferlib.pufferl')
        finally:
            signal.signal(signal.SIGINT, handler)

    def _build(self, env, total_steps):
        """Borrow a single environment through PufferLib's serial vectorizer."""
        import ast
        import configparser
        import math
        import tempfile
        import gymnasium as gym
        import torch
        if not isinstance(env.observation_space, gym.spaces.Box) or len(env.observation_space.shape) != 1:
            raise NotImplementedError('PufferLib adapter requires flat Box observations')
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise NotImplementedError('Initial PufferLib adapter supports Discrete actions only')
        for name in ('observation_space', 'action_space'):
            declared = getattr(self, name)
            if declared is not None and declared != getattr(env, name):
                raise ValueError(f'Incompatible {name}')
        backend_module = self._load_backend()
        import pufferlib.vector
        import pufferlib.emulation
        import pufferlib.models
        from pathlib import Path
        parser = configparser.ConfigParser()
        parser.read(Path(pufferlib.__file__).parent / 'config/default.ini')
        config = {}
        for key, value in parser['train'].items():
            try:
                config[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                config[key] = value
        config.update(device='cpu', optimizer='adam', compile=False, use_rnn=False,
                      amp=False, cpu_offload=False, precision='float32',
                      batch_size=128, minibatch_size=64, bptt_horizon=16,
                      anneal_lr=False, learning_rate=3e-4, env='aidge_rl', seed=self.seed)
        reserved = {'seed', 'env', 'data_dir', 'total_timesteps'} & set(self.training_options)
        if reserved:
            raise ValueError(f'Adapter-owned training options: {sorted(reserved)}')
        unknown = set(self.training_options) - set(config)
        if unknown:
            raise ValueError(f'Unknown PufferLib training options: {sorted(unknown)}')
        config.update(self.training_options)
        if (config['device'] != 'cpu' or config['use_rnn'] or config['compile'] or
                config['cpu_offload'] or config['amp'] or config['anneal_lr']):
            raise NotImplementedError('Initial adapter supports CPU feed-forward, no compile/AMP/offload/LR annealing')
        batch, mini, horizon = config['batch_size'], config['minibatch_size'], config['bptt_horizon']
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (batch, mini, horizon)):
            raise ValueError('Batch, minibatch, and horizon must be positive integers')
        if batch % mini or mini % horizon or mini < 2:
            raise ValueError('Batch must divide into minibatches, minibatches into horizons; minibatch >= 2')
        config['total_timesteps'] = max(batch, math.ceil(total_steps / batch) * batch)
        self._directory = tempfile.TemporaryDirectory(prefix='aidge_puffer_')
        config['data_dir'] = self._directory.name
        owner = self
        class BorrowedEnv(gym.Wrapper):
            """Count actual transitions without transferring simulator ownership."""
            def step(self, action):
                """Count successful underlying steps, excluding reset observations."""
                result = self.env.step(action)
                owner._steps += 1
                return result
            def close(self):
                """Leave the supplied environment caller-owned."""
        borrowed = BorrowedEnv(env)
        def creator(buf=None, seed=0):
            """Provide the explicit one-environment serial buffer boundary."""
            return pufferlib.emulation.GymnasiumPufferEnv(env=borrowed, buf=buf, seed=seed)
        try:
            self._vecenv = pufferlib.vector.make(creator, backend=pufferlib.vector.Serial,
                                                num_envs=1, seed=self.seed)
            # Fork RNG state so model initialization does not reseed the host.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed)
                self.model = pufferlib.models.Default(self._vecenv, hidden_size=self.hidden_size).to('cpu')
                self._rng_state = torch.get_rng_state()
            self.backend = backend_module.PuffeRL(config, self._vecenv, self.model)
            self._env = env
            self.observation_space, self.action_space = env.observation_space, env.action_space
        except BaseException:
            self.close()
            raise

    def train(self, env, total_steps, **options):
        """Train whole backend batches, reporting actual simulator transitions."""
        from ..trainer import TrainingResult
        if options:
            raise TypeError('Pass backend options through training={...} at construction')
        if self._closed:
            raise RuntimeError('This PufferLib agent has been closed')
        if self.backend is None:
            self._build(env, total_steps)
        elif env is not self._env:
            raise ValueError('Continuation requires the original training environment')
        before = self._steps
        self.backend.config['total_timesteps'] = self.backend.global_step + total_steps
        self.backend.total_epochs = max(1, (self.backend.config['total_timesteps'] +
            self.backend.config.get('batch_size', 1) - 1) // self.backend.config.get('batch_size', 1))
        import contextlib
        context = contextlib.nullcontext()
        if hasattr(self, '_rng_state'):
            import torch
            context = torch.random.fork_rng(devices=[])
        with context:
            if hasattr(self, '_rng_state'):
                torch.set_rng_state(self._rng_state)
            try:
                return self._train_batches(total_steps, before)
            finally:
                if hasattr(self, '_rng_state'):
                    self._rng_state = torch.get_rng_state()

    def _train_batches(self, total_steps, before):
        """Execute backend batches inside the adapter-owned random stream."""
        from ..trainer import TrainingResult
        while self._steps - before < total_steps:
            previous = self._steps
            self.backend.evaluate()  # Upstream name: rollout collection, not AidgeRL evaluation.
            self.backend.train()
            if self._steps <= previous:
                raise RuntimeError('PufferLib did not execute environment transitions')
        return TrainingResult(total_steps, self._steps - before)

    def reset(self, *, stream_id=None, seed=None):
        """Feed-forward policy execution carries no per-episode state."""

    def act(self, obs, *, info=None, stream_id=None):
        """Select an action from the same policy that PuffeRL optimized."""
        if self.model is None or self._closed:
            raise RuntimeError('Train the PufferLib agent before acting')
        import numpy as np
        import torch
        with torch.no_grad():
            logits, _ = self.model.forward_eval(torch.as_tensor(np.asarray(obs)[None]))
            action = (logits.argmax(dim=-1) if self.deterministic
                      else torch.distributions.Categorical(logits=logits).sample())
        return int(action.item())

    def save(self, path):
        """Save policy weights explicitly outside the temporary training folder."""
        if self.model is None or self._closed:
            raise RuntimeError('No live policy to save')
        import torch
        torch.save(self.model.state_dict(), path)

    def close(self):
        """Release backend threads/buffers and temporary checkpoint artifacts."""
        if self._closed:
            return
        self._closed = True
        try:
            if self.backend is not None:
                self.backend.close()
            elif self._vecenv is not None:
                self._vecenv.close()
        finally:
            if self.backend is not None:
                self.backend.utilization.stop()
                self.backend.utilization.join(timeout=2)
            if hasattr(self, '_directory'):
                self._directory.cleanup()
