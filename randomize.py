import gymnasium as gym
import mujoco
import numpy as np


class RandomizeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        m = self.env.unwrapped.model

        # cube colour
        self.cube_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "block")

        # cube position: find where the block's joint lives in qpos
        block_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "block")
        jnt = m.body_jntadr[block_body]
        self.qpos_adr = m.jnt_qposadr[jnt]        
    def reset(self, **kwargs):
        m = self.env.unwrapped.model

        # colour + lighting: before reset (model properties) Domain randomization applied so that the model properties are randomized before the reset, which is important for ensuring that the environment starts in a randomized state.
        m.geom_rgba[self.cube_geom_id, :3] = np.random.uniform(0, 1, size=3)
        m.light_diffuse[:] = np.random.uniform(0.4, 1.0)

        obs, info = self.enw
        # position: after reset (dynamic state)
        d = self.env.unwrapped.data
        d.qpos[self.qpos_adr + 0] += np.random.uniform(-0.05, 0.05)   # x jitter
        d.qpos[self.qpos_adr + 1] += np.random.uniform(-0.05, 0.05)   # y jitter
        mujoco.mj_forward(m, d)

        return obs, info