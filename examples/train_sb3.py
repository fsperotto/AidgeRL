"""Optional SB3 training using the same public Agent/Trainer/Runner interface."""
import gymnasium as gym
from aidge_rl import Agent, Trainer, Runner


if __name__ == "__main__":
    train_env, eval_env = gym.make("CartPole-v1"), gym.make("CartPole-v1")
    try:
        agent = Agent(package="sb3", method="PPO", n_steps=64,
                      batch_size=32, seed=42, device="cpu")
        print(Trainer(agent, train_env).train(total_steps=256))
        records = Runner(eval_env, agent).run(seed=43)
        print("Episode return:", sum(t.reward for t in records))
    finally:
        train_env.close()
        eval_env.close()
