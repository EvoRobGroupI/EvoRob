import os
import numpy as np
from src.world.robot.controllers import MLP
from src.utils.Filesys import get_project_root
from src.world.World import World
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils
import imageio
import mujoco
from gymnasium.spaces import Box

# ----------------------------------------------------------------------
# Environment and World (minimal)
# ----------------------------------------------------------------------
class TurtleGymEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"]
    }

    def __init__(self, render_mode="rgb_array", camera_name="topdown"):
        xml_path = os.path.join(get_project_root(), "CombinedSliderTurtle.xml")
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"Cannot find XML at {xml_path}")

        frame_skip = 2
        m = mujoco.MjModel.from_xml_path(xml_path)
        obs_dim = int(m.nq + m.nv)
        act_dim = int(m.nu)

        # Build the observation and action spaces
        obs_high = np.inf * np.ones(obs_dim, dtype=np.float64)
        act_high = np.ones(act_dim, dtype=np.float32)
        observation_space = Box(low=-obs_high, high=obs_high, dtype=np.float64)
        action_space = Box(low=-act_high, high=act_high, dtype=np.float32)

        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            camera_name=camera_name,
        )
        self.action_space = action_space
        utils.EzPickle.__init__(self)

    def _get_yaw(self):
    #     # Suppose this is the very first sensor, so its quaternion is at data.sensordata[0:4]
        # print(self.data.xquat[41])
        w, x, y, z = self.data.xquat[41]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
        yaw_rad = float(np.arctan2(siny_cosp, cosy_cosp))
        yaw_deg = np.degrees(yaw_rad)
        return yaw_deg

    def _get_obs(self):
        obs = np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
        yaw = self._get_yaw()
        obs = np.concatenate([obs, [yaw]])
        return obs
    
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        x_pos, y_pos, x_vel = self.data.qpos[0], self.data.qpos[1], self.data.qvel[0]
        yaw = 0 # You can keep your yaw computation if needed
        terminated = (x_pos < -1) or (abs(y_pos) > 2.0) or (abs(yaw)>80)
        truncated = False
        reward = x_vel + x_pos
        info = {"x_pos": x_pos, "y_pos": y_pos, "x_vel": x_vel}
        return obs, reward, terminated, truncated, info
    def reset_model(self):
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        if hasattr(self.data, "act") and self.data.act is not None:
            self.data.act[:] = 0.0
        return self._get_obs()

class TurtleWorld(World):
    def __init__(self):
        self.env = TurtleGymEnv()
        obs, _ = self.env.reset()
        self.state_space = obs.shape[0]
        self.action_space = self.env.action_space.shape[0]
        self.controller = MLP.NN_najaroController(self.state_space, self.action_space)
        self.n_params = self.controller.n_params

# ----------------------------------------------------------------------
# Video Generation
# ----------------------------------------------------------------------
def generate_ea_video(controller, video_name="Turtle_EA.mp4"):
    env = TurtleGymEnv(render_mode="rgb_array", camera_name="topdown")
    obs, _ = env.reset()
    frames = []
    max_steps = 500
    for step in range(max_steps):
        action = controller.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        frame = env.render()
        frames.append(frame)
        if step % 100 == 0: print("step: ", step)
        if terminated or truncated:
            break
    imageio.mimsave(video_name, frames, fps=30)
    print(f"Saved video: {video_name}")

# ----------------------------------------------------------------------
# Main: Select generation, load individual, and make video
# ----------------------------------------------------------------------
def main():
    gen = 35  # <<==== CHANGE THIS to the generation you want
    results_dir = os.path.join(get_project_root(), "results", "TurtleWorld", "CMAES")
    best_ind = np.load(os.path.join(results_dir, f"{gen}", "x_best.npy"))
    world = TurtleWorld()
    world.controller.geno2pheno(best_ind)
    vid_name = f"Best_Turtle_EA_{gen}.mp4"
    generate_ea_video(world.controller, vid_name)

if __name__ == "__main__":
    main()
