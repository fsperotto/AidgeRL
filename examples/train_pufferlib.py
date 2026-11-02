"""Optional CPU Python PufferLib PPO example (requires release 3.0.0)."""
import gymnasium as gym
from aidge_rl import Agent, Trainer, Runner


if __name__ == '__main__':
    env, evaluation = gym.make('CartPole-v1'), gym.make('CartPole-v1')
    agent = Agent(package='pufferlib', method='PPO', seed=42,
                  training={'batch_size': 128, 'minibatch_size': 64,
                            'bptt_horizon': 16, 'update_epochs': 1})
    try:
        print(Trainer(agent, env).train(total_steps=1024))
        records = Runner(evaluation, agent).run(seed=43)
        print('Episode return:', sum(t.reward for t in records))
    finally:
        agent.close()
        env.close()
        evaluation.close()
