import os
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


def make_env(env_name):
    # https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/her.yml
    env = gym.make(env_name)
    if env_name in ["FetchPush-v1", "FetchPickAndPlace-v1"]:
        print("\nUsing DoneOnSuccessWrapper\n")
        env = DoneOnSuccessWrapper(env)
    return env


def train(env_name, model_name):
    env = make_env(env_name)

    log_path = f"../logs/{model_name}"
    save_path = f"../models/{model_name}"  # this is without the .zip
    zip_save_path = os.path.abspath(save_path + ".zip")

    if os.path.isfile(zip_save_path):
        # load saved model
        model = HER.load(
            save_path,
            env=env,
            verbose=1,
            tensorboard_log=log_path,
        )
        print(f"\nLoaded previous saved model from {zip_save_path}\n")
        n_timesteps = int(1e6)
    else:
        # SAC hyperparams
        if env_name == "FetchReach-v1":
            # https://github.com/araffin/rl-baselines-zoo/blob/master/trained_agents/her/FetchReach-v1/config.yml
            # https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/her.yml
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
            n_timesteps = 24000  # overshoot in case we do not converge in IORT or IOIT
        # elif env_name == "FetchPush-v1":
        #     # https://github.com/araffin/rl-baselines-zoo/blob/master/trained_agents/her/FetchPush-v1/config.yml
        #     model = HER(
        #         'MlpPolicy',
        #         env,
        #         SAC,
        #         n_sampled_goal=4,
        #         goal_selection_strategy='future',
        #         buffer_size=1000000,
        #         ent_coef='auto',
        #         batch_size=256,
        #         gamma=0.95,
        #         learning_rate=0.001,
        #         learning_starts=1000,
        #         verbose=1,
        #         tensorboard_log=log_path,
        #     )
        #     #n_timesteps = int(3e6)
        #     n_timesteps = int(3.6e6)  # overshoot in case we do not converge in IORT or IOIT
        elif env_name == "FetchPush-v1":
            # https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/her.yml
            model = HER(
                'MlpPolicy',
                env,
                SAC,
                n_sampled_goal=4,
                goal_selection_strategy='future',
                buffer_size=1000000,
                ent_coef='auto',
                gamma=0.95,
                learning_starts=1000,
                train_freq=1,
                verbose=1,
                tensorboard_log=log_path,
            )
            #n_timesteps = int(3e6)
            n_timesteps = int(3.6e6)  # overshoot in case we do not converge in IORT or IOIT
        elif env_name == "FetchPickAndPlace-v1":
            # https://github.com/araffin/rl-baselines-zoo/blob/master/trained_agents/her/FetchPickAndPlace-v1/config.yml
            # https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/her.yml
            model = HER(
                'MlpPolicy',
                env,
                SAC,
                n_sampled_goal=4,
                goal_selection_strategy='future',
                buffer_size=1000000,
                ent_coef='auto',
                gamma=0.95,
                learning_starts=1000,
                train_freq=1,
                verbose=1,
                tensorboard_log=log_path,
            )
            #n_timesteps = int(4e6)
            n_timesteps = int(4.8e6)  # overshoot in case we do not converge in IORT or IOIT
        else:
            raise ValueError("Unsupported environment")

    print(f"\nNumber of train timesteps: {n_timesteps}\n")
    model.learn(n_timesteps)
    model.save(save_path)

    # epoch_timesteps = int(5e3)  # 100k timesteps in an epoch
    # epoch_timesteps = min(epoch_timesteps, n_timesteps)
    # for step in range(0, n_timesteps, epoch_timesteps):
    #     model.learn(epoch_timesteps)
    #     model.save(save_path)
    #     current_timestep = epoch_timesteps * (step + 1)
    #     print("\nSaved model at timestep {}\n".format(current_timestep))
    #     model = HER.load(save_path, env=env)


def test(env_name, model_name):
    env = make_env(env_name)
    save_path = f"../models/{model_name}"

    # Load saved model
    model = HER.load(save_path, env=env)

    obs = env.reset()

    # Evaluate the agent
    episode_reward = 0.0
    cumul_steps = 0
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        episode_reward += reward
        cumul_steps += 1
        if done or info.get('is_success', False):
            print("Reward:{:>6}\tSuccess?{:>4}\tFinalStep:{:>4}".format(episode_reward, info.get('is_success', False), cumul_steps))
            episode_reward = 0.0
            obs = env.reset()


if __name__ == "__main__":
    # To train model, check the constants above, then run
    # python3 fetch.py env_name model_name mode
    # i.e., python3 fetch.py FetchReach-v1 HER_SAC_FetchReach-v1_initial test
    assert len(sys.argv) == 4, "Please exactly specify env_name, model_name, mode"
    [_, env_name, model_name, mode] = sys.argv

    print(f"Environment: {env_name}")
    print(f"Saved model name: {model_name}")
    print(f"Running mode: {mode}")
    if mode == "train":
        train(env_name, model_name)
    elif mode == "test":
        test(env_name, model_name)
    else:
        raise ValueError("check that you are in train or eval mode")
