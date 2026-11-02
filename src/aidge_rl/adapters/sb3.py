"""SB3 public-API adapters separating inference from native training."""


class SB3Agent:
    """Adapt predict(), keeping recurrent execution state per stream."""

    def __init__(self, model=None, *, method=None, observation_space=None,
                 action_space=None, policy="MlpPolicy", deterministic=True, **options):
        """Wrap an already constructed SB3-compatible model."""
        if (model is None) == (method is None):
            raise ValueError("Supply exactly one model or method")
        self.method, self.policy, self.options = method, policy, options
        self.observation_space, self.action_space = observation_space, action_space
        self.model = model
        self.deterministic = deterministic
        self._states = {}

    def reset(self, *, stream_id=None, seed=None):
        """Clear only this stream's recurrent state; preserve model weights."""
        self._states.pop(stream_id, None)

    def act(self, obs, *, info=None, stream_id=None):
        """Call predict with an episode-start mask and save returned state."""
        if self.model is None:
            raise RuntimeError("Train the agent or provide an existing model before acting")
        first = stream_id not in self._states
        action, state = self.model.predict(
            obs, state=self._states.get(stream_id), episode_start=[first],
            deterministic=self.deterministic,
        )
        self._states[stream_id] = state
        return action


    def train(self, env, total_steps, **options):
        """Initialize lazily and delegate to SB3, preserving continuation state."""
        from ..trainer import TrainingResult
        for name in ("observation_space", "action_space"):
            expected = getattr(self, name)
            if expected is not None and expected != getattr(env, name):
                raise ValueError(f"Training environment has incompatible {name}")
        if self.model is None:
            method = self.method
            if isinstance(method, str):
                # An explicit whitelist prevents arbitrary backend attribute lookup.
                from ..agent import SB3_METHODS
                if method not in SB3_METHODS:
                    raise ValueError(f"Unknown SB3 method: {method}")
                import stable_baselines3
                method = getattr(stable_baselines3, method)
            if not isinstance(method, type):
                raise TypeError("SB3 method must be a supported name or class")
            self.model = method(self.policy, env, **self.options)
        else:
            self.model.set_env(env)
        if options.get("reset_num_timesteps") is True:
            raise ValueError("Continuation training requires reset_num_timesteps=False")
        options["reset_num_timesteps"] = False
        before = self.model.num_timesteps
        self.model.learn(total_timesteps=total_steps, **options)
        self._states.clear()
        return TrainingResult(total_steps, self.model.num_timesteps - before)


class SB3Trainer:
    """Legacy compatibility adapter; new code should use aidge_rl.Trainer."""

    def __init__(self, model):
        """Retain an existing model for older callers."""
        self.model = model

    def learn(self, total_timesteps, **kwargs):
        """Delegate the legacy call unchanged."""
        return self.model.learn(total_timesteps=total_timesteps, **kwargs)
