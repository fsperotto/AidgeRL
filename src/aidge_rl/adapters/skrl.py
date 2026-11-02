"""Optional skrl 2.1.1 Torch PPO adapter, using its persistent StepTrainer."""


class SkrlAgent:
    """Lazy, single-environment feed-forward PPO for Discrete or Box actions."""

    def __init__(self, *, method='PPO', observation_space=None, action_space=None,
                 seed=42, hidden_sizes=(64, 64), training=None, deterministic=True):
        """Store native PPO options without importing optional frameworks."""
        if method != 'PPO':
            raise ValueError("skrl currently supports method='PPO' only")
        if not hidden_sizes or any(not isinstance(n, int) or n <= 0 for n in hidden_sizes):
            raise ValueError('hidden_sizes must contain positive integers')
        self.observation_space, self.action_space = observation_space, action_space
        self.seed, self.hidden_sizes = seed, tuple(hidden_sizes)
        self.training_options = dict(training or {})
        self.deterministic = deterministic
        self.model = self.backend = self._env = None
        self._steps = 0
        self._closed = False

    def _build(self, env):
        """Construct skrl models, memory and one persistent step trainer."""
        import importlib.metadata
        try:
            version = importlib.metadata.version('skrl')
        except importlib.metadata.PackageNotFoundError as exc:
            raise ImportError("Install the optional backend: pip install 'aidge-rl[skrl]'") from exc
        if version != '2.1.1':
            raise ImportError(f'skrl 2.1.1 is required; found {version}')
        import gymnasium as gym
        import torch
        from skrl.agents.torch.ppo import PPO, PPO_CFG
        from skrl.memories.torch import RandomMemory
        from skrl.envs.wrappers.torch.gymnasium_envs import GymnasiumWrapper
        from skrl.trainers.torch import StepTrainer
        from skrl.utils.model_instantiators.torch import categorical_model, gaussian_model, deterministic_model
        if not isinstance(env.observation_space, gym.spaces.Box) or len(env.observation_space.shape) != 1:
            raise NotImplementedError('skrl adapter requires flat Box observations')
        if not isinstance(env.action_space, (gym.spaces.Discrete, gym.spaces.Box)):
            raise NotImplementedError('skrl adapter supports Discrete or Box actions')
        if isinstance(env.action_space, gym.spaces.Box) and len(env.action_space.shape) != 1:
            raise NotImplementedError('Continuous actions must be a flat Box')
        for name in ('observation_space', 'action_space'):
            declared = getattr(self, name)
            if declared is not None and declared != getattr(env, name):
                raise ValueError(f'Incompatible {name}')
        options = dict(rollouts=32, learning_epochs=4, mini_batches=4, learning_rate=3e-4)
        options.update(self.training_options)
        if 'experiment' in options:
            raise ValueError('Logging/checkpoint settings are adapter-owned; use save() explicitly')
        options['experiment'] = {'write_interval': 0, 'checkpoint_interval': 0}
        config = PPO_CFG(**options)
        rollouts = config.rollouts
        if rollouts < 2 or config.mini_batches < 1 or rollouts % config.mini_batches:
            raise ValueError('rollouts >= 2 must divide evenly into positive mini_batches')
        self._rollouts = rollouts
        owner = self
        class BorrowedEnv(gym.Wrapper):
            """Normalize optional info, seed once and preserve caller ownership."""
            def __init__(self):
                """Keep track of the first training reset."""
                super().__init__(env)
                self.first_reset = True
            def reset(self, *, seed=None, options=None):
                """Initialize the environment's RNG without reseeding each episode."""
                if self.first_reset and seed is None:
                    seed = owner.seed
                self.first_reset = False
                obs, info = self.env.reset(seed=seed, options=options)
                return obs, info or {}
            def step(self, action):
                """Count actual successful transitions, excluding resets."""
                obs, reward, terminated, truncated, info = self.env.step(action)
                owner._steps += 1
                return obs, reward, terminated, truncated, info or {}
            def close(self):
                """Do not close caller-owned resources."""
        class CPUWrapper(GymnasiumWrapper):
            """Keep tensors on CPU even when skrl detects an available GPU."""
            @property
            def device(self):
                """Expose the explicit device required by this adapter."""
                return torch.device('cpu')
        wrapped = CPUWrapper(BorrowedEnv())
        kwargs = dict(observation_space=env.observation_space, state_space=None,
                      action_space=env.action_space, device='cpu')
        network = [{'name': 'net', 'input': 'OBSERVATIONS',
                    'layers': list(self.hidden_sizes), 'activations': 'tanh'}]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            factory = categorical_model if isinstance(env.action_space, gym.spaces.Discrete) else gaussian_model
            policy = factory(**kwargs, network=network, output='ACTIONS',
                             **({'clip_actions': True} if factory is gaussian_model else {}))
            value = deterministic_model(**kwargs, network=network, output='ONE')
            memory = RandomMemory(memory_size=rollouts, num_envs=1, device='cpu')
            self.model = PPO(models={'policy': policy, 'value': value}, memory=memory,
                             **kwargs, cfg=config)
            self.backend = StepTrainer(env=wrapped, agents=self.model, cfg={
                'headless': True, 'disable_progressbar': True,
                'close_environment_at_exit': False})
            self._rng_state = torch.get_rng_state()
        self._env = env
        self.observation_space, self.action_space = env.observation_space, env.action_space

    def train(self, env, total_steps, **options):
        """Complete PPO rollout boundaries while retaining optimizer and memory."""
        import math
        from ..trainer import TrainingResult
        if options:
            raise TypeError('Pass PPO options through training={...} at construction')
        if self._closed:
            raise RuntimeError('This skrl agent has been closed')
        if self.backend is None:
            self._build(env)
        elif self._env is not env:
            raise ValueError('Continuation requires the original training environment')
        target = self._steps + math.ceil(total_steps / self._rollouts) * self._rollouts
        before = self._steps
        self.backend.cfg.timesteps = target
        import torch
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self._rng_state)
            try:
                while self._steps < target:
                    previous = self._steps
                    self.backend.train(timestep=previous, timesteps=target)
                    if self._steps != previous + 1:
                        raise RuntimeError('Expected one successful environment transition per skrl step')
            finally:
                self._rng_state = torch.get_rng_state()
        return TrainingResult(total_steps, self._steps - before)

    def reset(self, *, stream_id=None, seed=None):
        """Feed-forward inference does not require per-episode state."""

    def act(self, obs, *, info=None, stream_id=None):
        """Use skrl policy outputs; inference does not collect training data."""
        if self.model is None or self._closed:
            raise RuntimeError('Train the skrl agent before acting')
        import numpy as np
        import torch
        import gymnasium as gym
        self.model.enable_training_mode(False)
        with torch.no_grad():
            actions, outputs = self.model.act(torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32),
                                              None, timestep=self._steps, timesteps=max(1, self._steps))
            if self.deterministic:
                if isinstance(self.action_space, gym.spaces.Discrete):
                    actions = outputs['net_output'].argmax(dim=-1)
                else:
                    actions = outputs['mean_actions']
            action = actions[0].cpu().numpy()
        if isinstance(self.action_space, gym.spaces.Discrete):
            return int(action.item())
        return np.clip(action, self.action_space.low, self.action_space.high).astype(self.action_space.dtype)

    def save(self, path):
        """Save the backend's model/optimizer checkpoint explicitly."""
        if self.model is None or self._closed:
            raise RuntimeError('No live skrl model to save')
        self.model.save(str(path))

    def close(self):
        """Mark the agent closed without closing its caller-owned environment."""
        self._closed = True
