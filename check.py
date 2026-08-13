import gymnasium as gym, gym_hil
from randomize import RandomizeWrapper
env = gym.make("gym_hil/PandaPickCubeBase-v0", image_obs=True)
rand_env = RandomizeWrapper(env)
print("has _get_obs:", hasattr(rand_env.unwrapped, "_get_obs"))
print([m for m in dir(rand_env.unwrapped) if "obs" in m.lower()])