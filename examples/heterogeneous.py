"""Run agents whose actions and observations have different representations."""
from aidge_rl import ConstantAgent, Runner


class TeamEnv:
    """One simultaneous joint step with a scalar and a structured action."""

    def reset(self, *, seed=None, options=None):
        """Activate both participants with heterogeneous observations."""
        self.agents = ["motor", "camera"]
        return {"motor": [0.0, 1.0], "camera": {"image": [[1, 2]]}}, None

    def step(self, actions):
        """Advance shared dynamics once and supply final observations."""
        assert isinstance(actions["motor"], float)
        assert isinstance(actions["camera"], dict)
        self.agents = []
        return ({"motor": [1.0, 1.0], "camera": {"image": [[2, 3]]}},
                {"motor": 1.0, "camera": 2.0},
                {"motor": True, "camera": True},
                {"motor": False, "camera": False}, None)


if __name__ == "__main__":
    runner = Runner(TeamEnv(), agents={
        "motor": ConstantAgent(0.5), "camera": ConstantAgent({"exposure": 2})})
    print(runner.run())
