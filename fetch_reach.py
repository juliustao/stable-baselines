import sys

import gym
import numpy as np

from stable_baselines import HER, SAC, DDPG, TD3

env = gym.make("FetchReach-v1")

model_name = "HER_SAC_FetchReach-v1_default"
log_path = f"../logs/{model_name}"
save_path = f"../models/{model_name}"

def train():
    """
    # SAC hyperparams
    FetchReach-v1:
    n_timesteps: !!float 20000
    policy: 'MlpPolicy'
    model_class: 'sac'
    n_sampled_goal: 4
    goal_selection_strategy: 'future'
    buffer_size: 1000000
    ent_coef: 'auto'
    batch_size: 256
    gamma: 0.95
    learning_rate: 0.001
    learning_starts: 1000
    """

    # SAC hyperparams:
    model = HER(
        'MlpPolicy',
        env,
        SAC,
        n_sampled_goal=4,
        goal_selection_strategy='future',
        buffer_size=1000000,
        ent_coef='auto',
        batch_size=256,
        gamma=0.95,
        learning_rate=0.001,
        learning_starts=1000,
        verbose=1,
        tensorboard_log=log_path,
    )

    model.learn(20000)
    model.save(save_path)


def test():
    # Load saved model
    model = HER.load(save_path, env=env)

    obs = env.reset()

    # Evaluate the agent
    episode_reward = 0
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        episode_reward += reward
        if done or info.get('is_success', False):
            print("Reward:", episode_reward, "Success?", info.get('is_success', False))
            episode_reward = 0.0
            obs = env.reset()


if __name__ == "__main__":
    # To train model, check the constants above, then run
    # python fetch_reach.py train
    mode = sys.argv[1]
    if mode == "train":
        train()
    elif mode == "eval" or mode == "test":
        test()
    else:
        raise ValueError("check that you are in train or eval mode")