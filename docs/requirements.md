# AidgeRL requirements and design direction

Status: initial design baseline; no implementation implied.  
Date: 2026-11-02

This document records the requirements and architectural direction discussed before implementation. Requirements describe intended behavior; proposed names and signatures remain subject to review. Open questions are listed explicitly.

## 1. Purpose and scope

AidgeRL shall provide a simple, generic, lightweight, and efficient interface for reinforcement learning and controller/system interaction. It shall accommodate native implementations and thin adapters around external implementations, including Gymnasium, PettingZoo, and Stable Baselines3 (SB3).

The same notation and concepts should remain usable across:
- One agent interacting with one environment.
- Heterogeneous agents interacting with a shared environment.
- One policy serving multiple environment instances.
- Combined vectorized and multi-agent execution.
- Fully and partially observable systems.
- Inference/evaluation and supported training integrations.

Generality must not make the single-agent case cumbersome. Integration must preserve the external implementation's semantics and state its limitations.

## 2. Architectural principles

- Prefer composition, ordinary objects, and small structural protocols over deep inheritance.
- Do not require external environments or policies to inherit AidgeRL classes.
- Separate environment dynamics, action selection, experience collection, learning, orchestration, and instrumentation.
- Introduce abstractions when a concrete use case needs them; avoid speculative class hierarchies.
- Keep public contracts small and explicitly documented.
- Preserve native observation/action structures unless conversion is necessary.
- Support a minimal inference/controller path without buffers, optimizers, or training dependencies.
- Limit the number of heavy dependencies and mandatory dependencies in general, if possible without making the own code heavy.
- Be verbose on comments to explain commands and variables, except if too obvious.
- Systematically use doc strings to explain files, classes, methods, and functions.

## 3. Environment contract

For single-agent interaction, retain Gymnasium semantics:

```python
obs, info = env.reset(seed=None, options=None)
next_obs, reward, terminated, truncated, info = env.step(action)
```

info can be None.

The actual reset signature uses keyword-only seed/options. Construction configures an object; reset initializes an episode. Reset must not call an object's constructor again.

Environments expose observation_space and action_space. Discrete, continuous, and structured spaces must be supported without assuming finite cardinality, integer indexing, or exhaustive enumeration.

Preserve terminated and truncated separately through collection and learning. Every transition delivered to the learner must include the final termination and truncation flags for that transition. A runner stopping collection does not automatically imply that the underlying task terminated.

Initial-state control and other episode-specific configuration should use reset options where compatible. Preserve render()/close() compatibility; external visualization remains optional.

## 4. State and observation

An optional state-aware environment capability shall distinguish:
- state_space: underlying system states.
- observation_space: information available to an agent.
- An observation model mapping system state to an observation.

A proposed name is StateEnv, rather than defining mandatory fully/partially observable inheritance branches.

Observation models may be deterministic or stochastic. Randomness must be explicit or use an owned, seedable generator. Fully observed systems may use an identity observation model.

Allow observation models to be replaced independently of dynamics. Define one authoritative observation boundary so adapters do not transform observations twice.

For multi-agent systems, support a common state and agent-specific observation models/spaces. Action-dependent sensing may require the previous action as an additional observation-model input.

State access is optional for external environments. Do not infer that an observation is the full state or automatically expose hidden state to policies through info. Privileged training/state access must be explicit.

## 5. Agent and policy contract

Separate the following responsibilities conceptually:

```python
agent.reset(...)          # Episode-local execution state
action = agent.act(obs, ...)
agent.observe(transition) # Where experience ingestion is supported
agent.learn(...)          # Where externally scheduled updates are supported
```

These are proposed contracts, not methods that every inference-only wrapper must implement.

Episode reset preserves learned parameters, optimizer state, and persistent experience by default. Reinitializing knowledge and starting an independent experiment are distinct operations.

A convenience interaction facade may support the originally proposed symmetry:

```python
action = agent.reset(initial_obs)
action = agent.step(next_obs, reward, terminated, truncated, info=...)
```

If implemented, it must process the final transition without choosing a valid next action for a finished stream. Exact facade names and return types remain open.

Do not require an environment reference merely to select actions; spaces or an interface specification may suffice. Optional random and constant agents provide simple baselines.

Learning/exploration schedules are algorithm-specific. An on-policy/off-policy label alone must not determine update ordering.

## 6. Experience and execution state

Experience must preserve obs, action, reward, next_obs, terminated, truncated, and relevant info. Support action-selection metadata such as behavior log-probabilities and value estimates without imposing it on all algorithms.

Document ownership and lifetime of observation/action buffers; avoid both accidental mutation of stored experience and unconditional copies.

Keep policy parameters distinct from per-stream execution state. Recurrent state, exploration noise, and pending transition context must be independent for each interaction stream.

Use explicit environment and agent identities where required. A conceptual stream key is (env_id, agent_id). Shared policy parameters do not imply shared recurrent state.

## 7. Runner and orchestration

A separate Runner coordinates environment interaction and agents through composition.

Proposed usage:

```python
runner = Runner(env, agent)
runner.run(max_steps=10_000)

runner = Runner(env, agents={
    "helicopter": helicopter_agent,
    "camera": camera_agent,
})
```

Provide reset(), step(), and run() with documented outputs and stopping conditions. Keep manual stepping usable for debugging and interactive interfaces.

Construct objects once, or accept factories when deferred construction is needed. Episode reset must not reconstruct them.

Separate interaction execution from experiment comparisons, repeated runs, and seed sweeps. Independent repetitions require fresh training state; episode resets do not.

Define resource ownership, close behavior, interruption, and exception handling. Library code should not terminate the host process.

## 8. Heterogeneous multi-agent interaction

Use agent-ID mappings at the multi-agent boundary. Each agent may have a different observation/action space, representation, policy, and lifecycle.

Joint actions and observations are Cartesian-product structures, not necessarily numerical vectors. Do not force heterogeneous values into a stacked array.

For simultaneous interaction:
- Gather actions for agents scheduled to act.
- Advance shared dynamics once with the joint action.
- Route resulting observations, rewards, flags, and info by agent ID.
- Distinguish individual agent termination from the shared environment's lifecycle.
- Deliver final experience even when an agent leaves the active set.

Support policy sharing without requiring it. Joint experience and optional global training state must remain available for centralized learners.

Turn-based interaction requires an explicit scheduling adapter; it must not be silently treated as simultaneous or vectorized execution.

## 9. Single-agent normalization and vectorization

Keep ordinary observations/actions in the single-agent public interface. A small adapter may normalize a single agent into an internal one-agent mapping.

Select interface adapters explicitly or through declared capabilities. Do not identify multi-agent interaction by checking whether an observation is a dictionary: single-agent observations may also be dictionaries.

Vectorization and multi-agent interaction are independent dimensions. Pack only compatible policy calls, using explicit ordering and masks where needed. Batching separate observations must not accidentally give each decentralized policy the joint observation.

Normalize vector-environment autoreset semantics:
- Preserve final observations needed for learning.
- Distinguish reset observations from transition successors.
- Exclude reset-only events from collected transitions.
- Reset execution state only for affected streams.
- Preserve termination/truncation semantics.

## 10. External integrations

Use thin adapters that can receive already constructed external objects.

Distinguish two integration levels:
1. Policy/inference adaptation: translate the common action-selection contract to an external prediction API.
2. Training adaptation: delegate to the external training loop, or implement a specifically supported bridge for externally collected experience.

Do not promise universal observe()/learn() compatibility. For SB3, prediction adaptation can use predict(); training may delegate to learn(). Externally driven collection/training requires an explicit, algorithm-aware integration.

Avoid relying on private backend methods in the generic layer. Isolate unavoidable version-specific integration and document supported versions and capabilities.

Do not require q_net, value tables, discrete actions, or enumeration merely to wrap a model. Preserve native recurrent-state and episode-start handling.

Use adapter-side configuration translation where helpful; avoid mirroring every backend constructor argument in the core API. Preserve access to the wrapped object.

An existent simulator should be easily integrated as an environment using an andapter.

## 11. Instrumentation, survival, and finite-space extensions

Preserve useful PyRL ideas through optional components:
- Lifecycle callbacks for logging, evaluation, checkpoints, and visualization.
- External renderers and interactive stepping.
- Replay/rollout storage where required by native learners.
- Callable exploration schedules/strategies.
- Finite-space indexing, tabular algorithms, visitation counts, and sampled/exhaustive policy analysis.
- Survival/budget tracking and survival-aware policies.

Budget, ruin, and the rule budget += reward are not universal agent/environment properties. Put them in an explicit extension with well-defined task, observation, and termination semantics.

Diagnostics must be opt-in. Do not allocate dense state-action-transition tables or extract neural Q tables during ordinary execution.

## 12. Performance and dependencies

- No compulsory space enumeration or table extraction in the interaction path.
- Avoid unnecessary packing, tensor/array conversions, copies, and repeated model construction.
- Allow compatible batched policy execution.
- Keep callbacks optional and avoid unnecessary per-step instrumentation.
- Profile concrete implementations before claiming performance improvements.
- Keep dependencies minimal and external integrations optional where imports permit.
- Importing the core must not require Pygame, SB3, or every supported backend.

## 13. Package organization

Use a src layout, pyproject.toml, explicit public exports, and the import name aidge_rl. Proposed distribution name: aidge-rl, subject to availability.

Start with a few responsibility-based modules such as agent.py, environment.py, runner.py, and transitions.py. Add algorithms/ and adapters/ as implementations require them.

Keep examples/, tests/, and docs/ outside the importable package. Avoid empty speculative modules, generic manager hierarchies, and an unstructured utils.py.

README.md introduces the project and links to detailed documents. This document records requirements. AGENTS.md, if introduced, gives coding-assistant instructions rather than replacing product requirements.

## 14. Verification and initial implementation sequence

Use meaningful behavioral/contract tests for:
- Single-agent interaction, terminal transition delivery, and reset persistence.
- Heterogeneous multi-agent routing and lifecycle boundaries.
- Independent execution state with shared policies.
- Supported continuous and structured spaces.
- Adapter semantics, including supported autoreset modes.
- SB3 inference compatibility without DQN-specific assumptions.
- Explicit training ownership and supported training integrations.

Keep examples executable and aligned with public signatures.

Implement incrementally:
1. Minimal package metadata, interaction contracts, and a single-agent runner.
2. Simple native agents/environments and small executable examples.
3. Heterogeneous simultaneous multi-agent routing.
4. SB3 policy adaptation and an explicitly scoped training integration.
5. Vector/turn-based adapters, native algorithms, and optional extensions as needed.

This sequence is a proposal, not a commitment to implement all features in the first release.

## 15. Open decisions

- Exact public method names, signatures, and return types.
- Whether the symmetric reset/step agent facade is included initially.
- Transition representation and optional action metadata.
- Internal normalization layout and adapter selection mechanism.
- Multi-agent environment completion and dynamic membership conventions.
- Execution-state ownership and reset masks.
- Training integration scope per backend and algorithm.
- Python/dependency versions, build backend, and initial release scope.
- Aidge backend/graph integration boundaries and deployment/export interfaces.

## 16. PyRL reference

Source reviewed: [fsperotto/PyRL, commit 6f37774](https://github.com/fsperotto/PyRL/tree/6f37774e0d8e99175ab962f8b4a835dc825e91dd).

Preserve the design ideas in agent.py, env.py, system.py, sim.py, renderer.py, and external-model wrappers. Do not port their discrete-space, budget, reset-training, or backend-private assumptions into the generic core.


## 17. Experiment: repeated evaluation across methods and environments

Experiment orchestrates statistical evaluation separately from Runner interaction.
It accepts named agent factories and environment factories and evaluates their
Cartesian product: different agents on the same environment, the same method on
different environments, or both. Each pair is repeated independently.

- Construct a fresh environment and a fresh or freshly loaded agent for every
  method/environment/repetition combination. Factories must not return shared
  mutable objects. Agent factories receive the evaluation environment and seed.
- Evaluate several episodes per repetition without resetting learned parameters.
  Evaluation must not collect training experience or update models implicitly.
- Agent factories may train beforehand on separate training environments, or load
  checkpoints. Report whether variation measures training seeds, evaluation seeds,
  or both; repeating one checkpoint does not measure training variability.
- Use reproducible, matched environment episode seeds between methods. This does
  not guarantee identical stochastic trajectories when methods consume randomness
  differently. Agent factories are responsible for backend/model seeding.
- Keep raw results labeled with method, environment, repetition, seed, episode,
  return, length, and termination/truncation flags.
- Summarize returns within each repetition, then report the mean and sample
  standard deviation across independent repetition means. A single repetition
  has undefined sample standard deviation (None), not zero. More advanced
  confidence intervals and hypothesis tests are optional future analysis.
- Release evaluation environments on success and exceptions. Factory-created
  training resources remain the factory's responsibility.
- Experiment evaluates single-agent tasks in evaluate or train_evaluate mode. Multi-agent team evaluation,
  persistence/export, parallel experiments, and advanced statistics are extensions.

## 18. Initial implementation boundary

The initial release supplies a uniform Agent facade and Trainer, dependency-free protocols, ConstantAgent and
RandomAgent, a state-observation mixin, a single/parallel multi-agent Runner,
Transition records, Experiment, an existing-simulator callable adapter, and
SB3 public prediction/native-training adapters. Use update callbacks for native
learning schedules; the Runner never silently calls a backend training loop.
The symmetric agent facade, native PPO/SAC, buffers, vector autoreset adapters,
turn-based scheduling, and Aidge graph export are not yet implemented.

Parallel environments must expose env.agents, act simultaneously, and retain
final observations/rewards/flags for each agent that acted, including departing
agents. Empty env.agents ends the shared episode. New agents require registered
policies and their initial observation in the preceding environment result.
Hooks receive (event, runner, payload); transition payloads retain joint records.
Agent reset/act/observe use explicit stream_id; reset also accepts an optional
seed. Borrowed observations must remain valid until consumed; copy_value can
provide defensive snapshots for simulators that reuse buffers.

## 19. Uniform Agent construction and Trainer (implemented API)

These decisions supersede earlier proposals exposing backend-specific trainers.

- Agent(package="sb3", method="PPO", observation_space=..., action_space=...,
  **options) resolves a supported method string through an explicit package
  registry. Package names are case-insensitive; method names are backend names.
- Agent(package="sb3", method=PPO, **options) accepts a backend class. External
  classes require an explicit package; never infer adaptation from class names.
- Agent(method=NativeClass, **options) constructs a native policy. Supplied spaces
  are forwarded to that class only when provided.
- Agent(implementation=existing) wraps an existing native object;
  package="sb3" selects adaptation of an existing SB3 model. Do not combine
  existing implementations with constructor options or a method.
- register_package(name, factory) allows lazy integration without backend imports
  in the core. Factories receive method, spaces, and backend options.
- SB3 construction is deferred until an environment is passed to Trainer.
  Validate declared spaces against that environment. Expose the wrapped model.
- Trainer(agent, env).train(total_steps=...) is the backend-agnostic training API.
  It uses the agent's train capability, or a Runner-based native loop using
  observe and learn/an explicit update callback. Unsupported training fails
  explicitly. No backend type checks belong in Trainer.
- Delegated training returns TrainingResult(requested_steps, actual_steps).
  Native training counts executed interactions; SB3 measures the change in
  num_timesteps and may overshoot the budget to finish rollouts. Repeated calls
  continue training without reconstructing weights or resetting step counters.
- Episode reset never clears learned parameters. The same trained Agent must
  work with Runner on a separate evaluation environment.
- Trainer does not close caller-owned environments. Backend training options
  are forwarded only on the delegated path. Runner-specific schedules/horizons
  cannot silently change delegated execution.
- Experiment defaults to mode="evaluate". mode="train_evaluate" requires positive
  training_steps, creates separate train/evaluation envs per repetition, trains
  through Trainer, evaluates through Runner, and closes both environments.
  Store measured training_results separately from evaluation episode results.
- Factories seed agent/model initialization. Experiment supplies matched seeds
  to evaluation episodes and seeds the separate training environment before use.
- SB3Trainer remains only as a legacy compatibility API; new examples and user
  workflows must use Agent and Trainer.

## 20. Supported backend and method discovery

- Expose list_packages(installed_only=False) and
  list_methods(package=None, installed_only=False) at package top level.
- Return sorted fresh lists of supported package names and declared method names;
  omitting package returns a package-to-method mapping. Package matching is
  case-insensitive; unknown/unsupported packages raise ValueError.
- Distinguish supported integrations from installed backends. The optional
  installed filter checks module discoverability, without importing frameworks
  or promising complete runtime/dependency compatibility.
- Extend register_package with optional methods and module metadata; preserve
  existing two-argument registration. Class-based methods are not enumerable.
- Keep SB3 method discovery and method-name resolution on one authoritative list.
- CleanRL was investigated: its PyPI distribution does not make it an importable
  modular RL framework like SB3. Its documentation explicitly discourages imports.
  Do not register a nominal CleanRL adapter or copy training scripts into the core.
  An explicit future script/checkpoint bridge requires its own design and tests.


## 21. Optional RLlib integration

- Register package rllib with PPO only and include it in discovery without imports.
- Dependencies must be an opt-in rllib extra (target Ray 2.49.2 plus Torch).
- Use the common Agent/Trainer/Runner interfaces, lazy PPOConfig construction,
  the new RLModule inference API, and backend-owned training iterations.
- First scope is local single-agent feed-forward PPO, flat Box observations,
  and Box/Discrete actions. Reject unsupported methods and spaces explicitly.
- Preserve native distribution sampling and action normalization/clipping.
- Report actual additional sampled environment steps, including rollout overshoot;
  fail if metrics do not advance. Continue with the original training environment.
- Caller owns training/evaluation environments. The borrowing wrapper must not
  close them. Agent.close stops its algorithm without shutting down shared Ray.
  Experiment must close its owned agents as well as environments on failure/success.
- Save checkpoints through the algorithm public API; restoration is future scope.
- Include dependency-free contract tests and opt-in real CartPole/Pendulum smoke
  tests. Clearly distinguish target version from verified backend execution.
- Distributed environment factories, recurrent policies, existing-model wrapping,
  and multi-agent training/policy views require separate follow-up work.

## 22. Historical Python PufferLib integration

- Pin the PyPI distribution pufferlib==3.0.0 as an optional pufferlib extra.
  The moving 3.0 branch and native 5.0 are distinct APIs and are not supported.
- Register PPO in method discovery and construct through the common Agent API.
  No optional framework import occurs until backend training is requested.
- Delegate rollout collection and updates to upstream PuffeRL; never copy PPO.
- Initial boundary: single borrowed Gymnasium env, serial vectorizer, CPU default
  feed-forward policy, flat Box observations and Discrete actions. Reject other
  spaces/execution modes explicitly. Backend-specific training options live in
  training={...}; resource paths, step budget and seed remain adapter-owned.
- Count actual simulator step() calls, excluding upstream initial/reset sample
  accounting. Continue with the same environment and same policy/optimizer.
- Adapt trained logits to deterministic/stochastic Runner action selection.
- Preserve and document upstream PPO-variant semantics, reward clipping and
  termination/truncation behavior; do not imply equivalence to SB3 PPO.
- Restore the host SIGINT handler and import warning filters. Main-thread import
  is required; other upstream side effects must be documented.
- Temporary automatic checkpoints belong to the adapter; explicit weight saves
  belong to the caller. Close backend threads and vectors without closing the
  borrowed environment. Provide dependency-free and optional real smoke tests.
- GPU, recurrence, continuous actions, multiple envs, and checkpoint restoration
  require follow-up support. Installation can require native build tooling.

Validation completed against the actual PyPI 3.0.0 distribution, including
Gymnasium 0.29.1/PettingZoo 1.24.1: CartPole training, a complete Runner episode,
continuation and cleanup passed. This historical extra conflicts with newer
Gymnasium/RLlib extras and must use its own environment. Adapter-owned Torch RNG
state makes training sampling reproducible across repetitions without reseeding
the host Torch RNG.
