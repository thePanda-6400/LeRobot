import torch
import numpy as np
import gymnasium as gym
import gym_hil
import mujoco
from model import CNN
from randomize import RandomizeWrapper

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- load the trained model ---
model = CNN().to(device)
model.load_state_dict(torch.load("grasp_model.pt", map_location=device))
model.eval()

# --- perception: observation -> predicted position ---
def predict_position(obs):
    image = obs["pixels"]["front"]
    image = torch.from_numpy(image).float() / 255.0
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(image)
    return pred.squeeze(0).cpu().numpy() / 100.0

# --- set up env ---
env = gym.make("gym_hil/PandaPickCubeBase-v0", image_obs=True)
rand_env = RandomizeWrapper(env)

m, d = rand_env.unwrapped.model, rand_env.unwrapped.data
block_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "block")

# --- 6.1: one prediction ---
obs, info = rand_env.reset()
print("predicted position:", predict_position(obs))

# --- 6.2: pred vs true over several resets ---
for i in range(5):
    obs, info = rand_env.reset()
    pred = predict_position(obs)
    true = d.xpos[block_body].copy()
    error = ((pred - true) ** 2).mean() ** 0.5
    print(f"pred {pred.round(3)} | true {true.round(3)} | error {error:.4f}")

rand_env.close()   # <-- only ONE close, at the very end