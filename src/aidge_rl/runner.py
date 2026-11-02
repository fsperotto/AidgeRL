"""Single and simultaneous multi-agent interaction with explicit lifecycle rules."""
from .transitions import Transition


class Runner:
    """Coordinate interaction; agents= explicitly selects the parallel interface.

    Agents may implement reset(stream_id, seed), observe(transition, stream_id),
    and learn(). Updates are scheduled by the supplied update(agent, transition)
    callback. Episode hooks receive (event, runner, payload). No hidden learning
    occurs in evaluation mode. Caller-owned environments close only on close().
    """

    def __init__(self, env, agent=None, *, agents=None, training=False,
                 update=None, callbacks=(), episode_horizon=None, copy_value=None):
        """Store objects; do not reset environments or construct models."""
        if (agent is None) == (agents is None):
            raise ValueError("Supply exactly one of agent or agents")
        if episode_horizon is not None and episode_horizon <= 0:
            raise ValueError("episode_horizon must be positive")
        self.env = env
        self.multi = agents is not None
        self.agents = dict(agents) if self.multi else {None: agent}
        self.training, self.update = training, update
        self.callbacks = tuple(callbacks)
        self.episode_horizon = episode_horizon
        self.copy_value = copy_value or (lambda value: value)
        self.ready = False
        self.finished = True
        self.steps = 0
        self._seen = set()

    def _emit(self, event, payload=None):
        """Notify observers without imposing an instrumentation framework."""
        for callback in self.callbacks:
            callback(event, self, payload)

    def _start_agent(self, key, seed=None):
        """Reset execution state for a newly active stream."""
        agent = self.agents[key]
        if hasattr(agent, "reset"):
            agent.reset(stream_id=key, seed=seed)
        self._seen.add(key)

    def reset(self, *, seed=None, options=None):
        """Start an episode, retaining all learned agent parameters."""
        obs, info = self.env.reset(seed=seed, options=options)
        self.obs = obs if self.multi else {None: obs}
        self.infos = (info or {}) if self.multi else {None: info}
        self.steps = 0
        self._seen = set()
        keys = list(self.env.agents) if self.multi else [None]
        for key in keys:
            self._start_agent(key, seed)
        self.finished = not bool(keys)
        self.ready = True
        self._emit("episode_started")
        return obs, info

    def step(self):
        """Execute one joint step; return a Transition or agent-ID mapping.

        Parallel environments must provide final observations for departing
        agents. Missing final observations raise rather than corrupt experience.
        Autoresetting vector environments are not accepted by this runner.
        """
        if not self.ready or self.finished:
            raise RuntimeError("Call reset() before stepping an unfinished episode")
        keys = list(self.env.agents) if self.multi else [None]
        actions, previous = {}, {}
        for key in keys:
            if key not in self._seen:
                self._start_agent(key)
            previous[key] = self.copy_value(self.obs[key])
            actions[key] = self.agents[key].act(
                self.obs[key], info=self.infos.get(key), stream_id=key)
        # Snapshot actions before simulators can reuse mutable buffers.
        stored_actions = {k: self.copy_value(v) for k, v in actions.items()}
        obs, rewards, terms, truncs, infos = self.env.step(
            actions if self.multi else actions[None])
        if not self.multi:
            obs, rewards, terms, truncs, infos = (
                {None: value} for value in (obs, rewards, terms, truncs, infos))
        infos = infos or {}
        self.steps += 1
        limit = self.episode_horizon is not None and self.steps >= self.episode_horizon
        transitions = {}
        for key in keys:
            transitions[key] = Transition(
                previous[key], stored_actions[key], rewards[key],
                self.copy_value(obs[key]), bool(terms[key]),
                bool(truncs[key] or limit), infos.get(key), key)
        # Build all records before callbacks or learners can mutate anything.
        self.obs, self.infos = obs, infos
        self.finished = limit or (not self.env.agents if self.multi else
                                  terms[None] or truncs[None])
        for key, transition in transitions.items():
            agent = self.agents[key]
            if self.training:
                if hasattr(agent, "observe"):
                    agent.observe(transition, stream_id=key)
                if self.update is not None:
                    self.update(agent, transition)
            if transition.terminated or transition.truncated:
                self._seen.discard(key)
        self._emit("transition", transitions if self.multi else transitions[None])
        if self.finished:
            self._emit("episode_finished")
        return transitions if self.multi else transitions[None]

    def run(self, *, max_steps=None, episodes=1, seed=None):
        """Run up to episodes completed episodes and return collected records.

        max_steps limits collection without marking an otherwise continuing
        episode truncated. A subsequent run resumes that episode. Use step()
        directly for bounded-memory collection through callbacks.
        """
        if episodes <= 0 or (max_steps is not None and max_steps < 0):
            raise ValueError("episodes must be positive and max_steps nonnegative")
        records, completed = [], 0
        while completed < episodes and (max_steps is None or len(records) < max_steps):
            if not self.ready or self.finished:
                self.reset(seed=seed if not records else None)
                if self.finished:
                    completed += 1
                    continue
            records.append(self.step())
            if self.finished:
                completed += 1
        return records

    def close(self):
        """Close the environment explicitly, leaving agent ownership unchanged."""
        if hasattr(self.env, "close"):
            self.env.close()

    def __enter__(self):
        """Allow explicit environment ownership through a context manager."""
        return self

    def __exit__(self, *exc):
        """Close the environment even when execution raises."""
        self.close()
