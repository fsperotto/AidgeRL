# AidgeRL

A lightweight layer for RL interaction and repeated evaluation, using composition
and native observation/action representations. The core has no runtime dependencies.

## Install and test

Python 3.10 or newer:

```bash
pip install -e .
python -m unittest discover -s tests -v
python examples/evaluation.py
python examples/heterogeneous.py
```

Optional integrations: `pip install -e '.[gymnasium,sb3]'`.

## One agent and environment

```python
from aidge_rl import Runner, ConstantAgent

runner = Runner(env, ConstantAgent(action))
obs, info = runner.reset(seed=42)
transition = runner.step()
runner.close()
```

Ordinary Gymnasium environments work directly. `info=None` is supported.
Custom policies implement `act(obs, *, info=None, stream_id=None)`; optional
`reset(*, stream_id=None, seed=None)` clears execution state. In training mode,
`observe(transition, *, stream_id=None)` receives experience. An explicit
`update(agent, transition)` callback schedules updates. Evaluation is the default.

`runner.run(episodes=1, max_steps=None)` returns records. A collection step limit
leaves an unfinished episode resumable; `episode_horizon` explicitly truncates
an episode before experience delivery. For long runs use `step()` and callbacks
instead of retaining all transitions. `copy_value=copy.deepcopy` snapshots mutable
buffers; the default borrows values without copying. Callers own env resources;
a Runner context manager closes them explicitly on exit.

## Heterogeneous simultaneous agents

```python
runner = Runner(parallel_env, agents={"motor": motor_agent, "camera": camera_agent})
records = runner.run()
```

`agents=` explicitly selects a PettingZoo Parallel-shaped interface. Each space
and representation may differ. The environment exposes `agents` (participants
scheduled next), and step returns agent-ID dictionaries with final observations
for every participant that acted. Empty `env.agents` ends the episode. Policies
can be shared when they keep independent state per stream ID. Sharing a stateful
policy between different Runner instances requires separate policy wrappers.
Transition callbacks receive joint records for centralized processing.

## Repeated experiments

```python
from aidge_rl import Experiment

experiment = Experiment(
    agents={"method_a": lambda env, seed: make_agent_a(env, seed),
            "method_b": lambda env, seed: load_agent_b(env, seed)},
    environments={"task_a": make_env_a, "task_b": make_env_b},
    repetitions=10, episodes=5, seed=42, episode_horizon=1000,
)
rows = experiment.run()
statistics = experiment.summary()
```

Each factory is called afresh per pair/repetition. Evaluation does not train.
Factories may train first using a separate environment, and must seed their
backend and close any training resources. Raw rows retain episode provenance;
summary SD is computed across repetition means, not all episodes pooled together.
Repeated evaluation of one fixed checkpoint measures evaluation variability,
not independent training variability. Experiment currently evaluates single-agent tasks.

## External models and simulators

```python
from aidge_rl.adapters import SB3Agent, SB3Trainer, SimulatorEnv

agent = SB3Agent(model, deterministic=True)  # public predict(), per-stream state
SB3Trainer(model).learn(total_timesteps=10_000)  # SB3 owns training and collection
```

The wrapped model remains available as `.model`. SB3Trainer is used independently
of Runner; externally collected transitions are not silently injected into SB3.
SimulatorEnv accepts a simulator plus reset/step conversion callbacks and spaces.
StateEnv is an optional mixin for replaceable observation models; external
environments need not inherit from it or expose their hidden state.

## Status and design

See [requirements](docs/requirements.md). Current support includes single-agent
and simultaneous heterogeneous multi-agent interaction, callbacks, statistical
evaluation, and thin model/simulator adapters. Native learning algorithms,
vector/autoreset and turn-based adapters, and Aidge graph export remain future work.
SB3 adapters are tested with public-API fixtures; real-backend integration needs
validation in an environment with SB3 installed.
