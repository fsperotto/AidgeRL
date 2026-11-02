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

## Common agent construction and training

```python
import gymnasium as gym
from aidge_rl import Agent, Trainer, Runner

train_env = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1")
try:
    agent = Agent(
        package="SB3", method="PPO", policy="MlpPolicy",
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        learning_rate=3e-4, n_steps=128, seed=42, device="cpu",
    )
    result = Trainer(agent, train_env).train(total_steps=10_000)
    print(result)  # Requested and actual interactions; rollouts may overshoot.
    records = Runner(eval_env, agent).run(episodes=1, seed=43)
    agent.model.save("ppo_cartpole")
finally:
    train_env.close()
    eval_env.close()
```

`method=PPO` with `package="sb3"` also accepts an external class. Without a
package, `method=MyNativeAgent` constructs a class implementing the common
contract. `Agent(implementation=object)` wraps an existing native object;
`Agent(package="sb3", implementation=model)` adapts an existing SB3 model.
External classes require an explicit package; no backend inference is attempted.
Use `register_package(name, factory)` to register lazy adapter construction.

SB3 imports/model construction are deferred until training when using a method
name. `agent.act()` before that initialization raises an actionable error.
Trainer owns orchestration, while the adapter decides whether training delegates
to the backend. Native agents can use observe/learn with Runner; choose an
explicit update callback for algorithms needing a different update schedule.
Training does not change the agent identity. Repeated calls continue learning.
Trainer leaves environment ownership to its caller.

## Train-and-evaluate experiments

```python
experiment = Experiment(
    agents={"PPO": lambda env, seed: Agent(
        package="sb3", method="PPO", seed=seed, device="cpu")},
    environments={"CartPole": lambda: gym.make("CartPole-v1")},
    mode="train_evaluate", training_steps=10_000,
    repetitions=3, episodes=5, seed=42,
)
results = experiment.run()
print(experiment.summary())
print(experiment.training_results)
```

Each repetition creates a fresh agent and separate training/evaluation
environments. Evaluate-only mode remains the default for pretrained policy
factories. Factory functions must return fresh objects and seed their backend.

## External models and simulators

Direct `SB3Agent(model)` prediction adaptation remains available. `SB3Trainer`
is retained for compatibility only; new code should use `Trainer(agent, env)`.
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

## Discover supported packages and methods

```python
from aidge_rl import list_packages, list_methods

print(list_packages())                       # ['pufferlib', 'rllib', 'sb3', 'skrl']
print(list_methods('SB3'))                    # ['A2C', 'DDPG', 'DQN', 'PPO', 'SAC', 'TD3']
print(list_methods())                         # Mapping of package names to methods
print(list_packages(installed_only=True))     # Discoverable backends in this runtime
```

The default catalog lists supported adapters even before their optional backend
is installed. `installed_only=True` checks module discoverability without importing
backend code; it does not validate every transitive dependency. Native classes
passed directly to Agent are not an enumerable package catalog. Custom adapters
can declare methods/module metadata through
`register_package(name, factory, methods=(...), module="import_name")`.

### CleanRL compatibility

CleanRL has a [PyPI distribution](https://pypi.org/project/cleanrl/), but its
[official documentation](https://docs.cleanrl.dev/) explicitly says it is not a
modular library and is not intended to be imported. Its standalone algorithm
scripts do not provide an SB3-equivalent training API accepting an arbitrary
existing environment and returning a reusable trained algorithm object.
AidgeRL therefore does not advertise or register a CleanRL package adapter.
A future integration would need an explicitly scoped script/checkpoint bridge
or adapted algorithm implementation, rather than a thin SB3-style wrapper.


## Optional Ray RLlib PPO

Install explicitly: `pip install -e '.[rllib]'`. This extra pins Ray/RLlib 2.49.2
and adds Torch; a plain AidgeRL install does not install or import either.

```python
agent = Agent(package="rllib", method="PPO", seed=42,
              training={"train_batch_size_per_learner": 256,
                        "minibatch_size": 64, "num_epochs": 2})
try:
    result = Trainer(agent, train_env).train(total_steps=1024)
    records = Runner(eval_env, agent).run(episodes=1)
finally:
    agent.close()
```

See `examples/train_rllib.py`. The initial adapter supports local single-agent,
feed-forward PPO, flat Box observations, and Discrete/Box actions. It applies
RLlib action normalization/clipping during inference and reports sampled
lifetime-step differences, including rollout overshoot. Repeated training uses
the same model and original environment. `agent.model` exposes the algorithm;
`agent.save(path)` saves a checkpoint. Loading checkpoints, recurrent policies,
distributed factories, and joint multi-agent training are not implemented yet.
External RLlib classes are not accepted by this first PPO-only adapter.

Trainer leaves environments caller-owned. `agent.close()` stops the algorithm;
it never explicitly shuts down shared Ray. RLlib may initialize Ray during build;
applications own any global Ray runtime and shutdown policy. Experiment closes
its factory-created agent and environments. Existing instances supplied directly
to Agent with package='rllib' are not yet supported.

Run optional real-backend smoke tests with:
`AIDGE_RLLIB_SMOKE=1 PYTHONPATH=src:. python -m unittest tests.test_rllib -v`.
The adapter is covered by dependency-free contract tests. Ray 2.49.2 installed
successfully, but the attempted real smoke test could not start its local GCS
service in the restricted implementation environment. The pinned version is the
target API version, not a claim of verified end-to-end runtime compatibility.

## Optional Python PufferLib PPO

This adapter targets the **PyPI release `pufferlib==3.0.0`**, not the moving 3.0
branch or native 5.0 stack. Dependencies remain optional:

```bash
pip install -e '.[pufferlib]'
```

PufferLib builds native extensions even for its Python trainer. Its upstream
build may need NumPy, Torch, a C/C++ toolchain and system graphics dependencies.
On Linux, ensure the configured `CC`/`CXX` compilers exist. An upstream
`NO_OCEAN=1` metadata bug was observed for 3.0.0; that shortcut is not recommended.
If building against an already installed Torch, upstream supports
`--no-build-isolation` after installing its build requirements. Use a separate
virtual environment because this historical backend constrains NumPy below 2,
Gymnasium to <=0.29.1 and PettingZoo to <=1.24.1. Its extra cannot be combined
with the newer Gymnasium/RLlib extras in the same dependency resolution.

```python
agent = Agent(package="pufferlib", method="PPO", seed=42,
              training={"batch_size": 128, "minibatch_size": 64,
                        "bptt_horizon": 16, "update_epochs": 1})
try:
    result = Trainer(agent, train_env).train(total_steps=1024)
    records = Runner(eval_env, agent).run(seed=43)
    agent.save("puffer_policy.pt")
finally:
    agent.close()
```

See `examples/train_pufferlib.py`. Initial scope: one caller-owned Gymnasium env,
CPU feed-forward default policy, flat Box observations and Discrete actions.
This is PufferLib's PPO **variant**, preserving its reward clipping and upstream
termination/truncation behavior. It is not algorithmically identical to SB3 PPO.
Rollout collection remains inside PuffeRL, not AidgeRL Runner. The adapter counts
actual simulator step calls rather than upstream reset samples, so requested
budgets may overshoot by a rollout. Continuation retains policy/optimizer state
and requires the same training environment; LR annealing is disabled.

Initialization must occur on the main thread so the adapter can restore the
caller's SIGINT handler after upstream imports. Imports may still have upstream
side effects, including resource symlinks; training configures Torch's numerical
settings. `close()` releases backend buffers, monitoring threads and temporary
checkpoints without closing the caller's environment. Explicit `save()` writes
policy weights; checkpoint restoration is not implemented. GPU, recurrent,
continuous-action and multiprocessing/vector-environment support are future work.

Opt-in backend verification:
`AIDGE_PUFFER_SMOKE=1 PYTHONPATH=src:. python -m unittest tests.test_pufferlib -v`.


Validation: the actual PyPI PufferLib 3.0.0 source built successfully using GCC
and its CartPole smoke passed training, a complete Runner episode, continuation,
and cleanup. This is a functional check, not an algorithm performance benchmark.

## Optional skrl Torch PPO

Install the backend only when needed: `pip install -e '.[skrl]'`.
This integration pins **skrl 2.1.1** and uses its PyTorch implementation.

```python
agent = Agent(package="skrl", method="PPO", seed=42,
              hidden_sizes=(64, 64),
              training={"rollouts": 32, "learning_epochs": 4,
                        "mini_batches": 4, "learning_rate": 3e-4})
try:
    print(Trainer(agent, train_env).train(total_steps=1024))
    records = Runner(eval_env, agent).run(seed=43)
    agent.save("skrl_agent.pt")
finally:
    agent.close()
```

See `examples/train_skrl.py`. Initial support is CPU, one Gymnasium environment,
feed-forward PPO, flat Box observations and Discrete/flat Box actions. The adapter
uses skrl's model instantiators, RandomMemory and persistent StepTrainer; no PPO
loss or optimizer is reimplemented. Backend options are native PPO_CFG fields
inside `training={...}`; backend logging/checkpoint automation is disabled, and
`save()` explicitly writes the backend checkpoint. Automatic preprocessors are
not enabled unless configured by the caller.

Training completes whole rollouts and reports actual simulator steps, so a small
requested budget may overshoot. Continuing uses the same environment, policy,
optimizer, memory and trainer. Training's Torch RNG is isolated from the host RNG.
Runner evaluation does not ingest experience or update parameters. Continuous
actions are clipped to the declared bounds; deterministic inference uses the
policy mean for continuous actions and argmax for discrete actions.

Caller owns the environments; the adapter registers no exit-time environment
close hook. `agent.close()` disables further use. It does not close the supplied
environment. Class/existing-object adaptation, GPU, JAX/Warp, recurrence,
vectorization, multi-agent algorithms and checkpoint restoration remain future
work; discovery advertises only PPO, not the full upstream algorithm catalog.

Real skrl 2.1.1 tests passed on CartPole and Pendulum: training, complete episode
inference, parameter-preserving evaluation, model/optimizer continuation and
checkpoint saving. Run them explicitly with:
`AIDGE_SKRL_SMOKE=1 PYTHONPATH=src:. python -m unittest tests.test_skrl -v`.
