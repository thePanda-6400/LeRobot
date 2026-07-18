#script to collect data 

from pathlib import Path
import csv

import gymnasium as gym
import gym_hil
import imageio
import mujoco

from randomize import RandomizeWrapper


N_SAMPLES = 300
DATA = Path("data")
IMAGES = DATA / "images"
IMAGES.mkdir(parents=True, exist_ok=True)   # makes data/ and data/images/

env = gym.make("gym_hil/PandaPickCubeBase-v0", image_obs=True)
rand_env = RandomizeWrapper(env)

m = rand_env.unwrapped.model
d = rand_env.unwrapped.data
block_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "block")

rows = []
for i in range(N_SAMPLES):
    obs, info = rand_env.reset()

    image = obs["pixels"]["front"]
    position = d.xpos[block_body].copy()          # .copy() matters!

    filename = f"img_{i:04d}.png"                 # img_0000.png, img_0001.png, ...
    imageio.imwrite(IMAGES / filename, image)
    rows.append([filename, position[0], position[1], position[2]])

    if i % 50 == 0:
        print(f"{i}/{N_SAMPLES}")

with open(DATA / "labels.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "x", "y", "z"])   # header
    writer.writerows(rows)

rand_env.close()
print(f"saved {len(rows)} samples to {DATA}")