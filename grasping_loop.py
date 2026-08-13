import gym_hil
import gymnasium as gym
import mujoco as mj
import imageio
from randomize import RandomizeWrapper 
import numpy as np

env = RandomizeWrapper(gym.make("gym_hil/PandaPickCubeBase-v0", image_obs= True))