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
    """
    A Gymnasium/MuJoCo environment that loads 'CombinedSliderTurtle.xml' from the project root.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"]
    }

    def __init__(self, render_mode=None, camera_name=None):
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

        # for i, name in enumerate(self.model.body_names):
        #     print(f"body index {i:2d}: {name}") 

        # print(self.data.body("turtle").id)
        # self.once = True
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
    
    # def _get_yaw(self):
    #     # data.xmat is a flat (n_bodies × 9) array. For body 0, the first 9 entries are its 3×3 rotation.
    #     # In row-major:   [ R00, R01, R02,
    #     #                  R10, R11, R12,
    #     #                  R20, R21, R22 ]
    #     #
    #     # If the turtle only spins around z, then yaw = atan2(R10, R00).
    #     # Read the first two entries of the base’s rotation matrix:
        
    #     # if self.data.qpos[0] > 2 and self.once:
    #     #     print((self.data.xquat))
    #     #     self.once = False
    #     # print(len(self.data.xmat[0]))
    #     # print(self.data.xquat[41])
    #     R00 = float(self.data.xmat[0][0])  # entry (0,0)
    #     R10 = float(self.data.xmat[3][0])  # entry (1,0)

    #     # If both are effectively zero (i.e. R is not yet valid), return 0.0:
    #     if abs(R00) < 1e-8 and abs(R10) < 1e-8:
    #         return 0.0

    #     # Otherwise compute yaw = atan2(R10, R00)
    #     return float(np.arctan2(R10, R00))
    
    def _get_yaw(self):
    #     # Suppose this is the very first sensor, so its quaternion is at data.sensordata[0:4]
        # print(self.data.xquat[41])
        w, x, y, z = self.data.xquat[41]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
        yaw_rad = float(np.arctan2(siny_cosp, cosy_cosp))
        yaw_deg = np.degrees(yaw_rad)
        return yaw_deg

    def step(self, action):

        # # Before simulating, record the “old” state
        # old_qpos = self.data.qpos.copy()
        # old_qvel = self.data.qvel.copy()

        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()

        x_pos = self.data.qpos[0]
        y_pos = self.data.qpos[1]
        x_vel = self.data.qvel[0]

        yaw = self._get_yaw()

        terminated = (x_pos < -1) or (abs(y_pos) > 3.0)

        milestone = int(x_pos*2 // 1)
        if milestone > getattr(self, "_last_ms", -1):
            self._last_ms = milestone
            r_milestone = 1.0
        else:
            r_milestone = 0.0
        direction_adj = 1 - 2*(abs(yaw)+0.01)/90

        v_thresh = 0.02      # ≈ 1 cm/s
        k_still  = 0.5       # penalty magnitude

        # … after you’ve got x_vel …
        if abs(x_vel) < v_thresh:
            r_still = -k_still
        else:
            r_still = 0.0

        reward = 0.8*np.max(x_vel,0)*np.cos(np.radians(yaw)) + 1.4*r_milestone - 0.5*abs(y_pos) + r_still 

        truncated = False
        info = {
            "x_pos": x_pos,
            "y_pos": y_pos,
            "x_vel": x_vel
        }

        return obs, reward, terminated, truncated, info
    
    # def step(self, action):

    #     # # Before simulating, record the “old” state
    #     # old_qpos = self.data.qpos.copy()
    #     # old_qvel = self.data.qvel.copy()


    #     self.do_simulation(action, self.frame_skip)
    #     obs = self._get_obs()

    #     x_pos = self.data.qpos[0]
    #     y_pos = self.data.qpos[1]
    #     x_vel = self.data.qvel[0]

    #     yaw_rate = self.data.qvel[5]   # its within 5, max 3 really
    #     yaw = self._get_yaw()


    #     # reward = x_vel + x_pos
    #     terminated = (x_pos < -1) or (abs(y_pos) > 3.0)

    #     direction_adj = 1 - 2*(abs(yaw)+0.01)/90
    #     reward = x_vel*direction_adj + x_pos

    #     truncated = False
    #     info = {
    #         "x_pos": x_pos,
    #         "y_pos": y_pos,
    #         "x_vel": x_vel
    #     }
    #     return obs, reward, terminated, truncated, info

    def reset_model(self):
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        if hasattr(self.data, "act") and self.data.act is not None:
            self.data.act[:] = 0.0  # Reset all actuators to zero
        return self._get_obs()

class TurtleWorld(World):
    def __init__(self):
        self.env = TurtleGymEnv()
        obs, _ = self.env.reset()
        self.state_space = obs.shape[0]
        self.action_space = self.env.action_space.shape[0]
        self.controller = MLP.NNController(self.state_space, self.action_space)
        self.n_params = self.controller.n_params

# ----------------------------------------------------------------------
# Video Generation
# ----------------------------------------------------------------------
def generate_ea_video(controller, video_name="Turtle_EA.mp4"):
    env = TurtleGymEnv(render_mode="rgb_array", camera_name="topdown")
    obs, _ = env.reset()
    frames = []
    max_steps = 1500
    for step in range(max_steps):
        action = controller.get_action(obs)
        # mod_action = [action[0], action[1], action[0], action[1]]
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
    gen = 80  # <<==== CHANGE THIS to the generation you want
    results_dir = os.path.join(get_project_root(), "results", "TurtleWorld", "CMAES")
    best_ind = np.load(os.path.join(results_dir, f"{gen}", "x_best.npy"))
    world = TurtleWorld()
    world.controller.geno2pheno(best_ind)
    vid_name = f"Best_Turtle_EA_{gen}.mp4"
    generate_ea_video(world.controller, vid_name)

if __name__ == "__main__":
    main()
