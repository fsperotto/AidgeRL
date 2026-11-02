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
_PACKAGE_METHODS = {}
_PACKAGE_MODULES = {}


def register_package(name, factory, *, methods=(), module=None):
    """Register a lazy adapter factory taking method, spaces, and options.

    The factory must return the common policy contract. Training may be exposed
    through train(env, total_steps, **options). Imports belong inside factories.
    """
    if not isinstance(name, str) or not name.strip() or not callable(factory):
        raise ValueError("A nonempty package name and callable factory are required")
    if isinstance(methods, str):
        raise TypeError("methods must be an iterable of names, not a string")
    methods = tuple(methods)
    if any(not isinstance(method, str) or not method for method in methods):
        raise ValueError("Method names must be nonempty strings")
    key = name.casefold()
    if key in _PACKAGES:
        raise ValueError(f"Package already registered: {name}")
    _PACKAGES[key] = factory
    _PACKAGE_METHODS[key] = tuple(sorted(set(methods)))
    _PACKAGE_MODULES[key] = module


def _sb3_factory(*, method, observation_space, action_space, **options):
    """Import only the lightweight adapter until the backend is actually needed."""
    from .adapters.sb3 import SB3Agent
    return SB3Agent(method=method, observation_space=observation_space,
                    action_space=action_space, **options)


SB3_METHODS = ("A2C", "DDPG", "DQN", "PPO", "SAC", "TD3")
register_package("sb3", _sb3_factory, methods=SB3_METHODS, module="stable_baselines3")



def _rllib_factory(*, method, observation_space, action_space, **options):
    """Resolve the optional adapter without importing the Ray runtime."""
    from .adapters.rllib import RLlibAgent
    return RLlibAgent(method=method, observation_space=observation_space,
                      action_space=action_space, **options)


register_package("rllib", _rllib_factory, methods=("PPO",), module="ray")


def list_packages(*, installed_only=False):
    """List registered integrations, optionally requiring a discoverable backend.

    This is an AidgeRL support catalog, not a scan of every installed RL package.
    Discovery uses find_spec without importing backend code; it cannot guarantee
    that all transitive dependencies are usable. Custom entries without module
    metadata are included in the supported catalog but excluded by installed_only.
    """
    from importlib.util import find_spec
    packages = []
    for name in sorted(_PACKAGES):
        module = _PACKAGE_MODULES[name]
        if installed_only:
            if module is None:
                continue
            try:
                if find_spec(module) is None:
                    continue
            except (ImportError, ValueError, AttributeError):
                continue
        packages.append(name)
    return packages


def list_methods(package=None, *, installed_only=False):
    """List declared method names for one package, or a package-to-method mapping.

    Names are those supported by the adapter, not every backend algorithm.
    Custom class methods need no registration and cannot be enumerated. An empty
    catalog for a custom integration means methods were not declared.
    """
    packages = list_packages(installed_only=installed_only)
    if package is None:
        return {name: list(_PACKAGE_METHODS[name]) for name in packages}
    key = package.casefold()
    if key not in _PACKAGES:
        raise ValueError(f"Unknown or unsupported package: {package}")
    return list(_PACKAGE_METHODS[key]) if key in packages else []


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
