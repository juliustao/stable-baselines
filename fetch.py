import sys

import gym
import numpy as np

from stable_baselines import HER, SAC, DDPG, TD3


class DoneOnSuccessWrapper(gym.Wrapper):
    # https://github.com/araffin/rl-baselines-zoo/blob/master/utils/wrappers.py
    # wrapper for FetchPush-v1 and FetchPickAndPlace-v1
    """
    Reset on success and offsets the reward.
    Useful for GoalEnv.
    """
    def __init__(self, env, reward_offset=1.0):
        super(DoneOnSuccessWrapper, self).__init__(env)
        self.reward_offset = reward_offset

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        done = done or info.get('is_success', False)
        reward += self.reward_offset
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.env.compute_reward(achieved_goal, desired_goal, info)
        return reward + self.reward_offset


def train(env_name, model_name):
    env = gym.make(env_name)

    log_path = f"../logs/{model_name}"
    save_path = f"../models/{model_name}"

    # https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/her.yml
    # SAC hyperparams
    if env_name == "FetchReach-v1":
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
        #n_timesteps = 20000
        n_timesteps = 25000  # overshoot the ideal in case we do not converge
    elif env_name == "FetchPush-v1":
        model = HER(
            'MlpPolicy',
            env,
            SAC,
            n_sampled_goal=4,
            goal_selection_strategy='future',
            buffer_size=1000000,
            ent_coef='auto',
            #batch_size=256,
            gamma=0.95,
            #learning_rate=0.001,
            learning_starts=1000,
            train_freq=1,
            verbose=1,
            tensorboard_log=log_path,
        )
        #n_timesteps = 3e6
        n_timesteps = int(3.5e6)  # overshoot the ideal in case we do not converge
        env = DoneOnSuccessWrapper(env)
    else:
        raise ValueError("Unsupported environment")

    model.learn(n_timesteps)
    model.save(save_path)


def test(env_name, model_name):
    env = gym.make(env_name)
    save_path = f"../models/{model_name}"

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
    # python3 fetch.py env_name model_name mode
    # i.e., python3 fetch.py FetchReach-v1 HER_SAC_FetchReach-v1_initial test
    assert len(sys.argv) == 4, "Please exactly specify env_name, model_name, mode"
    [_, env_name, model_name, mode] = sys.argv

    if mode == "train":
        train(env_name, model_name)
    elif mode == "test":
        test(env_name, model_name)
    else:
        raise ValueError("check that you are in train or eval mode")
