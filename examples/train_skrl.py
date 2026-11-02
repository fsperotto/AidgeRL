"""Optional skrl Torch PPO training and evaluation through the common API."""
import gymnasium as gym
from aidge_rl import Agent, Trainer, Runner


if __name__ == '__main__':
    env, evaluation = gym.make('Pendulum-v1'), gym.make('Pendulum-v1')
    agent = Agent(package='skrl', method='PPO', seed=42,
                  training={'rollouts': 32, 'learning_epochs': 2, 'mini_batches': 4})
    try:
        print(Trainer(agent, env).train(total_steps=1024))
        records = Runner(evaluation, agent).run(seed=43)
        print('Episode return:', sum(t.reward for t in records))
    finally:
        agent.close()
        env.close()
        evaluation.close()
