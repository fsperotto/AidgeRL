"""Callable adapter for existing simulators, without imposing a simulator API."""


class SimulatorEnv:
    """Expose Gymnasium-shaped reset/step through supplied conversion functions."""

    def __init__(self, simulator, *, observation_space, action_space,
                 reset, step, close=None):
        """Callbacks receive the simulator first and return standard env results."""
        self.simulator = simulator
        self.observation_space = observation_space
        self.action_space = action_space
        self._reset = reset
        self._step = step
        self._close = close

    def reset(self, *, seed=None, options=None):
        """Initialize through the supplied simulator conversion function."""
        return self._reset(self.simulator, seed=seed, options=options)

    def step(self, action):
        """Advance through the supplied simulator conversion function."""
        return self._step(self.simulator, action)

    def close(self):
        """Release resources when a close callback was supplied."""
        if self._close is not None:
            self._close(self.simulator)
