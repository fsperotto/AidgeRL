"""Behavioral contracts for lifecycle, heterogeneous routing, and experiments."""
import unittest
from copy import deepcopy
from aidge_rl import ConstantAgent, Experiment, Runner, StateEnv
from aidge_rl.adapters import SB3Agent, SB3Trainer, SimulatorEnv
from examples.evaluation import CounterEnv
from examples.heterogeneous import TeamEnv


class RecordingAgent(ConstantAgent):
    """Retain experiences and reset identities to inspect runner semantics."""

    def __init__(self):
        """Initialize persistent learning state."""
        super().__init__(0)
        self.transitions = []
        self.resets = []

    def reset(self, *, stream_id=None, seed=None):
        """Record execution reset without deleting knowledge."""
        self.resets.append(stream_id)

    def observe(self, transition, *, stream_id=None):
        """Retain the complete transition."""
        self.transitions.append(transition)


class CoreTests(unittest.TestCase):
    """Check public behavior with backend-independent fixtures."""

    def test_horizon_visible_and_reset_preserves_learning(self):
        """Deliver final truncation before observe, preserving learned state."""
        agent = RecordingAgent()
        runner = Runner(CounterEnv(), agent, training=True, episode_horizon=1)
        runner.run()
        self.assertTrue(agent.transitions[0].truncated)
        runner.run()
        self.assertEqual(len(agent.transitions), 2)

    def test_evaluation_and_collection_cut(self):
        """Evaluation does not learn and batch boundaries do not end episodes."""
        agent = RecordingAgent()
        runner = Runner(CounterEnv(), agent)
        first = runner.run(max_steps=1)
        self.assertFalse(first[0].truncated)
        self.assertFalse(runner.finished)
        self.assertEqual(len(runner.run()), 2)
        self.assertEqual(agent.transitions, [])
        self.assertEqual(len(agent.resets), 1)

    def test_heterogeneous_and_shared_state(self):
        """Route different representations and reset shared-policy streams separately."""
        records = Runner(TeamEnv(), agents={
            "motor": ConstantAgent(0.2), "camera": ConstantAgent({"mode": 1})}).run()
        self.assertEqual(records[0]["camera"].reward, 2)
        shared = RecordingAgent()
        runner = Runner(TeamEnv(), agents={"motor": shared, "camera": shared})
        runner.reset()
        self.assertEqual(shared.resets, ["motor", "camera"])

    def test_experiment_independence_and_statistics(self):
        """Construct every pairing anew and match episode seed schedules."""
        created, agents = [], []
        def env_factory():
            """Retain identities and close calls."""
            env = CounterEnv()
            env.closed = False
            env.close = lambda: setattr(env, "closed", True)
            created.append(env)
            return env
        def agent_factory(env, seed):
            """Retain independent policy identities."""
            agent = RecordingAgent()
            agents.append(agent)
            return agent
        experiment = Experiment({"a": agent_factory, "b": agent_factory},
                                {"x": env_factory, "y": env_factory},
                                repetitions=3, episodes=2)
        results = experiment.run()
        self.assertEqual(len(results), 24)
        self.assertEqual(len({id(a) for a in agents}), 12)
        self.assertTrue(all(e.closed for e in created))
        self.assertTrue(all(not a.transitions for a in agents))
        self.assertEqual(experiment.summary()["a", "x"]["repetitions"], 3)
        self.assertEqual([r.seed for r in results if r.agent == "a" and r.environment == "x"],
                         [r.seed for r in results if r.agent == "b" and r.environment == "x"])
        self.assertEqual(results, experiment.run())

    def test_exception_closes_environment(self):
        """Factories failing during setup still release their evaluation env."""
        env = CounterEnv()
        env.closed = False
        env.close = lambda: setattr(env, "closed", True)
        def fail(env, seed):
            """Simulate a failed checkpoint load."""
            raise ValueError("load failure")
        with self.assertRaises(ValueError):
            Experiment({"bad": fail}, {"env": lambda: env}).run()
        self.assertTrue(env.closed)

    def test_sb3_public_api_and_per_stream_reset(self):
        """Preserve prediction state without requiring a DQN-specific q_net."""
        class Model:
            """Public prediction/training API fixture."""
            def predict(self, obs, **kwargs):
                """Expose state routing in the returned action."""
                return kwargs, 42
            def learn(self, **kwargs):
                """Expose delegated training arguments."""
                return kwargs
        model = Model()
        agent = SB3Agent(model)
        self.assertEqual(agent.act({}, stream_id="a")["episode_start"], [True])
        self.assertEqual(agent.act({}, stream_id="a")["state"], 42)
        self.assertEqual(agent.act({}, stream_id="b")["episode_start"], [True])
        agent.reset(stream_id="a")
        self.assertEqual(agent.act({}, stream_id="a")["episode_start"], [True])
        self.assertEqual(SB3Trainer(model).learn(100), {"total_timesteps": 100})

    def test_mutable_observation_snapshot(self):
        """An explicit copy policy preserves simulator buffers before step."""
        class Mutable(CounterEnv):
            """Reuse the same observation buffer."""
            def reset(self, **kwargs):
                """Allocate one mutable value."""
                self.buffer = [0]
                return self.buffer, None
            def step(self, action):
                """Mutate the value in place."""
                self.buffer[0] += 1
                return self.buffer, 0, True, False, None
        record = Runner(Mutable(), ConstantAgent(0), copy_value=deepcopy).run()[0]
        self.assertEqual(record.obs, [0])
        self.assertEqual(record.next_obs, [1])

    def test_departing_agent_final_transition(self):
        """Deliver final experience when one participant leaves before another."""
        class Departing:
            """Parallel fixture with asymmetric lifetimes."""
            def reset(self, **kwargs):
                """Start both participants."""
                self.agents = ["a", "b"]
                return {"a": 0, "b": 0}, None
            def step(self, actions):
                """Remove one participant while retaining its final data."""
                before = list(self.agents)
                self.agents = ["b"] if "a" in before else []
                return ({k: 1 for k in before}, {k: 1 for k in before},
                        {k: k not in self.agents for k in before},
                        {k: False for k in before}, None)
        a, b = RecordingAgent(), RecordingAgent()
        records = Runner(Departing(), agents={"a": a, "b": b}, training=True).run()
        self.assertEqual(len(records), 2)
        self.assertEqual(len(a.transitions), 1)
        self.assertTrue(a.transitions[0].terminated)
        self.assertFalse(b.transitions[0].terminated)
        self.assertTrue(b.transitions[1].terminated)

    def test_state_and_simulator_adapter(self):
        """Keep state mapping and simulator conversion independent."""
        self.assertEqual(StateEnv().observation(3, rng=None), 3)
        model = StateEnv(lambda s, **kwargs: s[kwargs["agent_id"]])
        self.assertEqual(model.observation({"a": 4}, rng=None, agent_id="a"), 4)
        env = CounterEnv()
        adapted = SimulatorEnv(env, observation_space=None, action_space=None,
                               reset=lambda e, **kw: e.reset(**kw),
                               step=lambda e, a: e.step(a))
        self.assertEqual(len(Runner(adapted, ConstantAgent(0)).run()), 3)


if __name__ == "__main__":
    unittest.main()
