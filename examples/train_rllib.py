"""Optional local RLlib PPO training: install aidge-rl[rllib] first."""
import gymnasium as gym
from aidge_rl import Agent, Trainer, Runner


if __name__ == '__main__':
    env, evaluation = gym.make('CartPole-v1'), gym.make('CartPole-v1')
    agent = Agent(package='rllib', method='PPO', seed=42,
                  training={'train_batch_size_per_learner': 256,
                            'minibatch_size': 64, 'num_epochs': 2})
    try:
        print(Trainer(agent, env).train(total_steps=1024))
        records = Runner(evaluation, agent).run(seed=43)
        print('Episode return:', sum(t.reward for t in records))
    finally:
        agent.close()  # Stop backend resources separately from environment ownership.
        env.close()
        evaluation.close()
