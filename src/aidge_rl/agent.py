"""Small structural agent contract and baseline policies, without backend imports."""
from typing import Any, Protocol


class Policy(Protocol):
    """Action selection contract; reset/observe/learn are optional capabilities."""

    def act(self, obs: Any, *, info: Any = None, stream_id: Any = None) -> Any:
        """Select an action for an independent interaction stream."""
        ...


class ConstantAgent:
    """Return a configured action without changing its representation."""

    def __init__(self, action):
        """Store the action, which callers must treat as read-only."""
        self.action = action

    def act(self, obs, *, info=None, stream_id=None):
        """Return the configured action."""
        return self.action


class RandomAgent:
    """Sample an arbitrary Gymnasium-compatible action space."""

    def __init__(self, action_space):
        """Retain the supplied space without enumerating it."""
        self.action_space = action_space

    def reset(self, *, stream_id=None, seed=None):
        """Seed action sampling when an explicit seed is supplied."""
        if seed is not None:
            self.action_space.seed(seed)

    def act(self, obs, *, info=None, stream_id=None):
        """Sample directly from the action space."""
        return self.action_space.sample()


_PACKAGES = {}


def register_package(name, factory):
    """Register a lazy adapter factory taking method, spaces, and options.

    The factory must return the common policy contract. Training may be exposed
    through train(env, total_steps, **options). Imports belong inside factories.
    """
    key = name.casefold()
    if key in _PACKAGES:
        raise ValueError(f"Package already registered: {name}")
    _PACKAGES[key] = factory


def _sb3_factory(*, method, observation_space, action_space, **options):
    """Import only the lightweight adapter until the backend is actually needed."""
    from .adapters.sb3 import SB3Agent
    return SB3Agent(method=method, observation_space=observation_space,
                    action_space=action_space, **options)


register_package("sb3", _sb3_factory)


class Agent:
    """Uniform facade around native methods and explicitly selected packages.

    Native classes receive supplied spaces plus options. External classes need
    a package adapter; their module/class names are never guessed. Existing
    objects use implementation= and must not be combined with method/options.
    """

    def __init__(self, method=None, *, package=None, observation_space=None,
                 action_space=None, implementation=None, **options):
        """Resolve a named package or instantiate a native method class."""
        if implementation is not None:
            if method is not None or options:
                raise ValueError("implementation cannot be combined with method/options")
            if package is None:
                self.implementation = implementation
            elif package.casefold() == "sb3":
                from .adapters.sb3 import SB3Agent
                self.implementation = SB3Agent(implementation)
            else:
                raise ValueError("Existing external objects currently support package='sb3'")
        elif package is not None:
            try:
                factory = _PACKAGES[package.casefold()]
            except KeyError:
                raise ValueError(f"Unknown package: {package}") from None
            self.implementation = factory(method=method,
                observation_space=observation_space, action_space=action_space, **options)
        elif isinstance(method, type):
            if observation_space is not None:
                options["observation_space"] = observation_space
            if action_space is not None:
                options["action_space"] = action_space
            self.implementation = method(**options)
        else:
            raise ValueError("Supply a native class, package/method, or implementation")
        self.observation_space = observation_space
        self.action_space = action_space

    def __getattr__(self, name):
        """Expose backend capabilities without inventing unsupported methods."""
        return getattr(self.implementation, name)
