"""SB3 public-API adapters separating inference from native training."""


class SB3Agent:
    """Adapt predict(), keeping recurrent execution state per stream."""

    def __init__(self, model, *, deterministic=True):
        """Wrap an already constructed SB3-compatible model."""
        self.model = model
        self.deterministic = deterministic
        self._states = {}

    def reset(self, *, stream_id=None, seed=None):
        """Clear only this stream's recurrent state; preserve model weights."""
        self._states.pop(stream_id, None)

    def act(self, obs, *, info=None, stream_id=None):
        """Call predict with an episode-start mask and save returned state."""
        first = stream_id not in self._states
        action, state = self.model.predict(
            obs, state=self._states.get(stream_id), episode_start=[first],
            deterministic=self.deterministic,
        )
        self._states[stream_id] = state
        return action


class SB3Trainer:
    """Delegate training ownership to SB3; do not ingest Runner transitions."""

    def __init__(self, model):
        """Retain a model already configured with its training environment."""
        self.model = model

    def learn(self, total_timesteps, **kwargs):
        """Use the backend's native collection and optimization lifecycle."""
        return self.model.learn(total_timesteps=total_timesteps, **kwargs)
