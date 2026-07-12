import gymnasium as gym
import gym_hil
import imageio
import numpy as np
import mujoco as mj
from randomize import RandomizeWrapper


def rollout(env,policy, max_steps = 100):
    obs, info = env.reset()
    
    frames = []
    total_rewrd = 0
    while True:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(obs["pixels"]["front"])   # grab the front camera each step
        total_rewrd += reward
        if terminated or truncated:
            break

    success = total_rewrd > 0
    return frames, success, total_rewrd

def evaluate(env, policy, n_episodes = 50):
    successes = 0
    returns = []
    for i in range(n_episodes):
        frames, success, total_reward = rollout(env, policy)
        successes += success
        returns.append(total_reward)
    success_rate = successes / n_episodes
    avg_return = sum(returns) / n_episodes
    print(f"Success rate: {success_rate:.2f}, Average return: {avg_return:.2f}")
    return success_rate, avg_return

def random_policy(obs):
    return env.action_space.sample()


env = gym.make("gym_hil/PandaPickCubeBase-v0", image_obs=True)
rand_env = RandomizeWrapper(env)

for i in range(4):
    obs, info = rand_env.reset()
    imageio.imwrite(f"reset_{i}.png", obs["pixels"]["front"])

evaluate(rand_env, random_policy, n_episodes=50)
env.close()