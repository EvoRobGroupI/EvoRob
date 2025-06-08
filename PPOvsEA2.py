import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium import utils
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv

from stable_baselines3.ppo import PPO as SB3_PPO
from stable_baselines3.common.env_util import make_vec_env

from src.EA.CMAES import CMAES, CMAES_opts
from src.world.robot.controllers import MLP
from src.utils.Filesys import get_project_root
from src.world.World import World

# Verify MuJoCo binding is available
try:
    import mujoco
except ImportError:
    raise RuntimeError("Could not import `mujoco`. Make sure MuJoCo Python bindings are installed.")


# -----------------------------------------------------------------------------
# Environment definition (unchanged)
# -----------------------------------------------------------------------------
# class TurtleGymEnv(MujocoEnv, utils.EzPickle):
#     """
#     A Gymnasium/MuJoCo environment that loads 'CombinedSliderTurtle.xml' from the project root.
#     """

#     metadata = {
#         "render_modes": ["human", "rgb_array", "depth_array"]
#     }

#     def __init__(self, render_mode=None, camera_name=None):
#         xml_path = os.path.join(get_project_root(), "CombinedSliderTurtle.xml")
#         if not os.path.isfile(xml_path):
#             raise FileNotFoundError(f"Cannot find XML at {xml_path}")

#         frame_skip = 2
#         m = mujoco.MjModel.from_xml_path(xml_path)
#         obs_dim = int(m.nq + m.nv)
#         act_dim = int(m.nu)

#         # Build the observation and action spaces
#         obs_high = np.inf * np.ones(obs_dim, dtype=np.float64)
#         act_high = np.ones(act_dim, dtype=np.float32)
#         observation_space = Box(low=-obs_high, high=obs_high, dtype=np.float64)
#         action_space = Box(low=-act_high, high=act_high, dtype=np.float32)

#         MujocoEnv.__init__(
#             self,
#             xml_path,
#             frame_skip,
#             observation_space=observation_space,
#             render_mode=render_mode,
#             camera_name=camera_name,
#         )
#         self.action_space = action_space

#         # for i, name in enumerate(self.model.body_names):
#         #     print(f"body index {i:2d}: {name}") 

#         print(self.data.body("turtle").id)
#         self.once = True
#         utils.EzPickle.__init__(self)

#     def _get_obs(self):
#         return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
    
#     # def _get_yaw(self):
#     #     # data.xmat is a flat (n_bodies × 9) array. For body 0, the first 9 entries are its 3×3 rotation.
#     #     # In row-major:   [ R00, R01, R02,
#     #     #                  R10, R11, R12,
#     #     #                  R20, R21, R22 ]
#     #     #
#     #     # If the turtle only spins around z, then yaw = atan2(R10, R00).
#     #     # Read the first two entries of the base’s rotation matrix:
        
#     #     # if self.data.qpos[0] > 2 and self.once:
#     #     #     print((self.data.xquat))
#     #     #     self.once = False
#     #     # print(len(self.data.xmat[0]))
#     #     # print(self.data.xquat[41])
#     #     R00 = float(self.data.xmat[0][0])  # entry (0,0)
#     #     R10 = float(self.data.xmat[3][0])  # entry (1,0)

#     #     # If both are effectively zero (i.e. R is not yet valid), return 0.0:
#     #     if abs(R00) < 1e-8 and abs(R10) < 1e-8:
#     #         return 0.0

#     #     # Otherwise compute yaw = atan2(R10, R00)
#     #     return float(np.arctan2(R10, R00))
    
#     def _get_yaw(self):
#     #     # Suppose this is the very first sensor, so its quaternion is at data.sensordata[0:4]
#         # print(self.data.xquat[41])
#         w, x, y, z = self.data.xquat[41]
#         siny_cosp = 2.0 * (w * z + x * y)
#         cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
#         yaw_rad = float(np.arctan2(siny_cosp, cosy_cosp))
#         yaw_deg = np.degrees(yaw_rad)
#         return yaw_deg

#     def step(self, action):

#         # # Before simulating, record the “old” state
#         # old_qpos = self.data.qpos.copy()
#         # old_qvel = self.data.qvel.copy()


#         self.do_simulation(action, self.frame_skip)
#         obs = self._get_obs()

#         # After simulating, compare qpos/qvel
#         # new_qpos = self.data.qpos.copy()
#         # new_qvel = self.data.qvel.copy()
#         # print(
#         #     "[DEBUG step] action:", action,
#         #     "\n             old_qpos[:3]:", old_qpos[:3], "new_qpos[:3]:", new_qpos[:3],
#         #     "\n             old_qvel[:3]:", old_qvel[:3], "new_qvel[:3]:", new_qvel[:3]
#         # )

#         # x_pos = new_qpos[0]
#         # y_pos = new_qpos[1]
#         # x_vel = new_qvel[0]

#         x_pos = self.data.qpos[0]
#         y_pos = self.data.qpos[1]
#         x_vel = self.data.qvel[0]

#         # if x_pos > 2:
#         #     print(self.data.xquat[0])

#         # w, x, y, z = self.data.

#         yaw_rate = self.data.qvel[5]   # its within 5, max 3 really
#         yaw = self._get_yaw()
#         # if abs(yaw) > 90: print(yaw)
#         # print(yaw)

#         # reward = x_vel + x_pos
#         terminated = (x_pos < -1) or (abs(y_pos) > 3.0)

#         direction_adj = 1 - 2*(abs(yaw)+0.01) / 90
#         reward = x_vel * direction_adj + x_pos
#         # reward = x_vel*(0.1-(abs(yaw)+0.01)/90)
#         # reward = x_vel + x_pos
#         # terminated = (x_pos < -1) or (abs(y_pos) > 3.0) or (abs(yaw) > 45) 
#         # reward = x_vel + x_pos - abs(yaw)
#         # terminated = (x_pos < -1) or (abs(y_pos) > 3.0) or (abs(yaw) > 45) 
#         # if terminated and (abs(yaw) > 10): print("terminated")

#         # reward = x_vel + x_pos# - 0.5 * abs(y_pos) - abs(yaw) # - 100 * abs(yaw_rate)
#         # terminated = (x_vel < -0.05) or (abs(y_pos) > 3.0) #or abs(yaw) > 90 #or (abs(yaw_rate) > 2)
        

#         # terminated = False
#         truncated = False
#         info = {
#             "x_pos": x_pos,
#             "y_pos": y_pos,
#             "x_vel": x_vel
#         }
#         return obs, reward, terminated, truncated, info
    
#     # def step(self, action):
#     #     self.do_simulation(action, self.frame_skip)
#     #     obs = self._get_obs()
#     #     x_pos = self.data.qpos[0]
#     #     y_pos = self.data.qpos[1]
#     #     x_vel = self.data.qvel[0]
#     #     reward = x_vel + x_pos - 0.5*y_pos
#     #     terminated = (x_vel < -0.05) or (abs(y_pos) > 1.0)
#     #     truncated = False
#     #     info = {
#     #         "x_pos": x_pos,
#     #         "y_pos": y_pos,
#     #         "x_vel": x_vel
#     #     }
#     #     return obs, reward, terminated, truncated, info

#     def reset_model(self):
#         self.data.qpos[:] = self.init_qpos
#         self.data.qvel[:] = self.init_qvel
#         if hasattr(self.data, "act") and self.data.act is not None:
#             self.data.act[:] = 0.0  # Reset all actuators to zero
#         return self._get_obs()
###########################################################################

import os
import numpy as np
import mujoco

from gymnasium import utils
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv

from src.utils.Filesys import get_project_root


class TurtleGymEnv(MujocoEnv, utils.EzPickle):
    """
    MuJoCo environment that casts a 160° “fan” of rays (±80° around the
    turtle’s local +X axis) to detect poles. At each timestep, we record
    the two smallest ray‐hit distances (or ∞ if no pole in that direction)
    and append them to [qpos, qvel]. Thus the observation has shape (nq+nv+2,).
    """
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"]
    }
    def __init__(self, render_mode=None, camera_name=None):
        # ────────── 1) Load the merged XML (turtle + slider + poles) ──────────
        xml_path = os.path.join(get_project_root(), "CombinedSliderTurtle.xml")
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"Cannot find `{xml_path}`")

        # 2) Build the MuJoCo model & data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # After building self.model
        first_rf_sid = mujoco.mj_name2id(
            self.model,                     # MjModel*
            mujoco.mjtObj.mjOBJ_SENSOR,     # enum for “sensor”
            "rf_-80"                        # name we just gave in XML
        )
        if first_rf_sid == -1:
            raise RuntimeError("Sensor 'rf_-80' not found; check XML names.")

        self.rf_adr = self.model.sensor_adr[first_rf_sid]   # start index in sensordata
        self.N_RAYS = 21


        # 3) Extract dimensions
        nq = self.model.nq      # number of position DOFs
        nv = self.model.nv      # number of velocity DOFs
        nu = self.model.nu      # number of actuators

        # 4) We will return obs = [qpos(=nq), qvel(=nv), d1, d2] → length = nq + nv + 2
        # obs_dim = nq + nv + 4
        obs_dim = nq + nv + self.N_RAYS
        act_dim = nu

        # 5) Build Gym spaces
        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float32
        )

        # 6) Initialize the MujocoEnv base class
        super().__init__(
            xml_path,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode,
            camera_name=camera_name,
        )
        self.action_space = action_space

        # ─── 7) Find the turtle’s body ID ───────────────────────────────────
        self.turtle_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "turtle"
        )
        if self.turtle_body_id == -1:
            raise RuntimeError("Could not find body named 'turtle' in the model.")

        # ─── 8) Gather all “pole” body IDs (assume names “pole_i_j”) ─────────
        self.pole_body_ids = []
        for bid in range(self.model.nbody):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if nm is not None and nm.startswith("pole_"):
                self.pole_body_ids.append(bid)

        if len(self.pole_body_ids) == 0:
            raise RuntimeError(
                "No poles found: ensure each pole’s <body> is named 'pole_i_j'."
            )

        # ─── 9) Precompute cos(80°) for the angular cutoff ─────────────────
        self.cos80 = np.cos(np.radians(80.0))

        # ─── 10) Decide how many rays to cast (odd # so that 0° is exactly the center)
        self.N_RAYS = 21  # can adjust to 11, 41, etc.

        utils.EzPickle.__init__(self)

    def turtle_forward_vector(self):
        """
        Return the turtle’s body‐local +X axis expressed in world coordinates.
        MuJoCo stores each body k’s 3×3 rotation in data.xmat[k][0..8], row-major.
        The first COLUMN of R_k is (R[0,0], R[1,0], R[2,0]) = (x-axis of the body).
        In row-major layout: R00=data.xmat[k][0], R10=data.xmat[k][3], R20=data.xmat[k][6].
        """
        idx = self.turtle_body_id
        R00 = self.data.xmat[idx][0]
        R10 = self.data.xmat[idx][3]
        R20 = self.data.xmat[idx][6]
        fwd = np.array([R00, R10, R20], dtype=np.float32)
        norm = np.linalg.norm(fwd)
        if norm < 1e-8:
            # Fallback if degenerate
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return fwd / norm

    def _get_yaw(self):
        """
        Compute the turtle’s yaw angle (in degrees) from its base quaternion.
        This is the same helper you had previously. For instance:
        """
        # Suppose the turtle’s “nose” quaternion is at data.xquat[41]
        # (adjust the index if yours is different).
        w, x, y, z = self.data.xquat[41]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw_rad = float(np.arctan2(siny_cosp, cosy_cosp))
        return np.degrees(yaw_rad)


    def _get_obs(self):
        """
        Build an observation of length (nq + nv + 4):
          [ qpos₀ … qposₙq₋₁,
            qvel₀ … qvelₙv₋₁,
            d₁, idx₁,  d₂, idx₂ ]
        Here (d₁, idx₁) are the smallest‐distance ray and its index,
        and (d₂, idx₂) are the second‐smallest.

        Alternatively, you could replace idx₁/idx₂ by angle₁/angle₂ if you
        prefer a continuous angle rather than an integer index.
        """

        # ————————— 1) Flatten qpos, qvel —————————
        q = self.data.qpos.ravel().astype(np.float32)   # shape (nq,)
        v = self.data.qvel.ravel().astype(np.float32)   # shape (nv,)

        # ————————— 2) Turtle’s position & heading —————————
        tpos = self.data.xpos[self.turtle_body_id]        # (3,)
        fwd  = self.turtle_forward_vector()               # (3,)

        # Build a horizontal “right” vector
        up    = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(fwd, up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right = right / np.linalg.norm(right)

        # ————————— 3) Ray‐cast setup —————————
        pnt       = np.zeros((3, 1), dtype=np.float64)
        vec       = np.zeros((3, 1), dtype=np.float64)
        geomgroup = np.zeros((6, 1), dtype=np.uint8)
        flg_static   = 0
        bodyexclude  = 0
        geomid_out   = np.zeros((1, 1), dtype=np.int32)

        # We’ll store (distance, ray_idx) for every ray that hits a pole
        hits = []

        # Precompute the 21 angles in degrees
        angles_deg = np.linspace(-80.0, 80.0, self.N_RAYS)  # length = 21

        ray_origin = tpos.copy()                              # copy the 3‐vector
        ray_origin[2] = tpos[2] - 0.2   # shift 0.6 m above shell midpoint

        distances = self.data.sensordata[self.rf_adr : self.rf_adr + self.N_RAYS]

        # if getattr(self, "_dbg_counter", 0) % 20 == 0:
        #     print("rays (m):", np.round(distances, 2))
        # self._dbg_counter = getattr(self, "_dbg_counter", 0) + 1


        # normalise or clip so the network sees values in [0,1]
        MAX_SENSE = 6.0
        # d_norm = np.clip(distances, 0, MAX_SENSE) / MAX_SENSE   # shape (21,)
        dist_clipped = np.where((distances < 0) | (distances > MAX_SENSE),
                                MAX_SENSE,
                                distances)              # shape (21,)
        d_norm_inv = 1 - dist_clipped / MAX_SENSE  
        # print("rays (m):", np.round(d_norm_inv, 2))
        # --- 5) build observation ---
        obs = np.concatenate([q, v, d_norm_inv.astype(np.float32)], axis=0)
                # ───────────── DEBUG PRINT (REMOVE OR COMMENT OUT LATER) ─────────────
        # Print once every 20 calls so the console is readable.
        # Feel free to change the modulus or add a toggle flag.
        # if getattr(self, "_dbg_counter", 0) % 20 == 0:
        #     # angles for reference: -80, -72, …, +80
        #     print("range-finder m (m):", np.round(distances, 3))
        # self._dbg_counter = getattr(self, "_dbg_counter", 0) + 1
        # idx = np.argpartition(distances, 2)[:2]     # indices of two smallest
        # d1, d2 = distances[idx]
        # i1, i2 = idx


        # ————————— 4) Extract the two smallest hits or cap if none/one —————————
        # MAX_SENSE = 6.0   # any value > max possible turtle→pole distance

        # if len(hits) == 0:
        #     # no ray hit any pole
        #     d1, i1 = MAX_SENSE, -1   # index = -1 means “no hit”
        #     d2, i2 = MAX_SENSE, -1
        # elif len(hits) == 1:
        #     (d1, i1) = hits[0]
        #     d2, i2 = MAX_SENSE, -1
        # else:
        #     # sort by distance, take two smallest
        #     hits_sorted = sorted(hits, key=lambda x: x[0])
        #     (d1, i1), (d2, i2) = hits_sorted[:2]

        # d1 = min(d1, MAX_SENSE)/MAX_SENSE  # now in [0,1]
        # d2 = min(d2, MAX_SENSE)/MAX_SENSE
        # # If you want indices in [0,1], do:  i1_norm = i1/(self.N_RAYS-1), etc.

        # i1_norm = i1/(self.N_RAYS - 1)
        # i2_norm = i2/(self.N_RAYS - 1)
        # ————————— 5) Build the final observation —————————
        #
        # Option A: append raw indices (int)  → 4 extra scalars: (d1, i1, d2, i2)
        # obs = np.concatenate([
        #     q,                                      # shape (nq,)
        #     v,                                      # shape (nv,)
        #     np.array([
        #         d1,
        #         float(i1_norm),   # cast index to float32 so that obs is all float32
        #         d2,
        #         float(i2_norm)
        #     ], dtype=np.float32)
        # ], axis=0)

        return obs


    def step(self, action):
        # 1) Send the action into MuJoCo
        self.do_simulation(action, self.frame_skip)

        # 2) Get the new observation (with the two nearest‐pole distances appended)
        obs = self._get_obs()

        # 3) Compute reward & done flags exactly as before in your original code:
        #    (Example reward: forward‐speed adjusted by yaw, termination if out of bounds.)
        x_pos = self.data.qpos[0]
        y_pos = self.data.qpos[1]
        x_vel = self.data.qvel[0]
        yaw = self._get_yaw()        # if you still use that helper to compute yaw

        # Termination: if turtle falls off or y‐position too large
        terminated = (x_pos < -1) or (abs(y_pos) > 3.0)

        # Example reward‐shaping (yours may differ):
        direction_adj = 1 - 2 * (abs(yaw) + 0.01) / 90
        # reward = x_vel * direction_adj + x_pos
        reward = x_vel*direction_adj + 0.1*x_pos - 0.2*y_pos
        # We do not use “truncated” here, so always False—for Gymnasium compatibility:
        truncated = False

        # You can put anything you like into info; here we carry over x_pos, y_pos, x_vel
        info = {
            "x_pos": x_pos,
            "y_pos": y_pos,
            "x_vel": x_vel
        }

        # 4) Return exactly five values: obs, reward, terminated, truncated, info
        return obs, reward, terminated, truncated, info


    def reset_model(self):
        # Reset joint positions and velocities
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        if hasattr(self.data, "act") and self.data.act is not None:
            self.data.act[:] = 0.0
        return self._get_obs()
#####################################################################################

# -----------------------------------------------------------------------------
# Wrapper to unify EA controller interface
# -----------------------------------------------------------------------------
class TurtleWorld(World):
    """
    Wraps TurtleGymEnv to match the EA pipeline interface:
      - reset() → initial observation
      - step(a) → (next_obs, reward, done, info)
      - geno2pheno(g) → loads genotype into MLP controller
      - evaluate_individual(g) → cumulative reward over one rollout
    """

    def __init__(self):
        self.env = TurtleGymEnv()
        obs, _ = self.env.reset()
        self.state_space = obs.shape[0]
        self.action_space = self.env.action_space.shape[0]
        self.controller = MLP.NNController(self.state_space, self.action_space)
        self.n_params = self.controller.n_params

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, truncated, info

    def geno2pheno(self, genotype):
        self.controller.geno2pheno(genotype)
        return self.controller

    def evaluate_individual(self, genotype):
        max_steps = 1000
        self.geno2pheno(genotype)

        # reset() → (obs, info)
        obs = self.reset()

        total_reward = 0.0
        for _ in range(max_steps):
            # clip the raw network output into [-1, +1]
            raw_action = self.controller.get_action(obs)
            action     = np.clip(raw_action, -1.0, +1.0)
            # action = 
            # step(action) → (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, _ = self.step(action)

            total_reward += reward

            # “done” if either terminated or truncated is True
            if terminated or truncated:
                break

        return total_reward

    
    


# -----------------------------------------------------------------------------
# EA runner (unchanged)
# -----------------------------------------------------------------------------
def run_EA(ea, world):
    for gen in range(ea.n_gen):
        population = ea.ask()
        fitnesses = np.empty(ea.n_pop)
        # for i, indiv in enumerate(population):
        #     fitnesses[i] = world.evaluate_individual(indiv)
        from joblib import Parallel, delayed
        raw_fitnesses = Parallel(n_jobs=-1)(
            delayed(world.evaluate_individual)(indiv) for indiv in population)

        raw_fitnesses = np.asarray(raw_fitnesses, dtype=np.float64)   # ← ADD THIS
        # ranks = fitnesses.argsort().argsort()
        # fitnesses = ranks.astype(np.float32)
        # ranks      = raw_fitnesses.argsort().argsort().astype(np.float64)        # 0 … n_pop-1
        fitnesses  = raw_fitnesses.astype(np.float64)

        ea.tell(population, fitnesses)  
        print(f"Generation {gen:3d}: "
        f"best reward = {raw_fitnesses.max():9.1f}   "
        f"mean = {raw_fitnesses.mean():7.1f} ± {raw_fitnesses.std():6.1f}")
        # ea.tell(population, fitnesses)
    
    # Print summary before telling EA
        # best_idx = np.argmax(fitnesses)
        # worst_idx = np.argmin(fitnesses)
        # print(
        #     f"[EA gen {gen}] "
        #     f"best fitness = {fitnesses[best_idx]:.2f}, "
        #     f"worst fitness = {fitnesses[worst_idx]:.2f}"
        # )


# -----------------------------------------------------------------------------
# Video generation for EA-trained controller (unchanged)
# -----------------------------------------------------------------------------
def generate_ea_video(controller, video_name: str = "Turtle_EA.mp4"):
    env = TurtleGymEnv(render_mode="rgb_array", camera_name="topdown")
    obs, _ = env.reset()
    frames = []
    max_steps = 1000

    for _ in range(max_steps):
        action = controller.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        frame = env.render()
        frames.append(frame)
        if terminated or truncated:
            break

    import imageio
    imageio.mimsave(video_name, frames, fps=30)




# -----------------------------------------------------------------------------
# Video generation for PPO-trained policy
# -----------------------------------------------------------------------------
def generate_ppo_video(model, video_name: str = "Turtle_PPO.mp4"):
    env = TurtleGymEnv(render_mode="rgb_array", camera_name="topdown")
    obs, _ = env.reset()
    frames = []
    max_steps = 1000

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        frame = env.render()
        frames.append(frame)
        if terminated or truncated:
            break

    import imageio
    imageio.mimsave(video_name, frames, fps=30)


# -----------------------------------------------------------------------------
# Main entry point with algorithm switch
# -----------------------------------------------------------------------------
def main():
    # Choose algorithm: "CMAES" or "PPO"
    algorithm = "EA"  # <-- change this to "PPO" to use PPO instead of CMAES

    world = TurtleWorld()

    if algorithm == "EA":
        # --------------------
        # CMA-ES configuration
        # --------------------
        n_parameters = world.n_params
        CMAES_opts["min"] = -1
        CMAES_opts["max"] = 1
        CMAES_opts["num_parents"] = 50
        CMAES_opts["num_generations"] = 1
        CMAES_opts["mutation_sigma"] = 0.3

        population_size = 100
        results_dir = os.path.join(get_project_root(), "results", "TurtleWorld", "CMAES")
        # os.makedirs(results_dir, exist_ok=True)
        if os.path.isdir(results_dir):
            import shutil
            shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        ea = CMAES(population_size, n_parameters, CMAES_opts, results_dir)
        run_EA(ea, world)

        # Load best individual and generate video
        # best_ind = np.load(os.path.join(results_dir, "39", "x_best.npy"))

        ###################
        best_fitness = -np.inf
        best_genotype = None

        for gen_name in os.listdir(results_dir):
            gen_path = os.path.join(results_dir, gen_name)
            f_path = os.path.join(gen_path, "f_best.npy")
            x_path = os.path.join(gen_path, "x_best.npy")
            if os.path.isfile(f_path) and os.path.isfile(x_path):
                f_val = np.load(f_path)
                if f_val > best_fitness:
                    best_fitness = f_val
                    best_genotype = np.load(x_path)
        
        print(f"best fitness across generations: {best_fitness}")

        if best_genotype is None:
            raise RuntimeError(f"No `f_best.npy`/`x_best.npy` found under {results_dir}")

        best_ind = best_genotype
        ####################
        world.controller.geno2pheno(best_ind)
        generate_ea_video(world.controller, "Best_Turtle_EA.mp4")
        print("CMA-ES evolution complete. Saved 'Best_Turtle_EA.mp4'.")

    elif algorithm == "PPO":
        # --------------------
        # PPO configuration
        # --------------------

        import torch

        print("Torch version:", torch.__version__)
        # If you also want the CUDA version PyTorch was built with:
        print("CUDA version:", torch.version.cuda)

        # Wrap the environment for Stable-Baselines3
        def make_env():
            return TurtleGymEnv()

        # Use a vectorized environment for PPO
        vec_env = make_vec_env(make_env, n_envs=4)

        # Create the PPO model

        ## IF PAST MODEL WAS INTERRUPTED

        # ppo_model = SB3_PPO.load("ppo_interrupted.zip", env=vec_env, device="cuda")
        # remaining_timesteps = 500000
        # ppo_model.learn(total_timesteps=remaining_timesteps)

        # You can adjust total_timesteps as needed
        total_timesteps = 100_000
        ppo_model = SB3_PPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            # tensorboard_log=os.path.join(get_project_root(), "ppo_tensorboard"),
            device=torch.device("cuda")
        )

        # Train the PPO agent
        try:
            ppo_model.learn(total_timesteps=total_timesteps)
        except KeyboardInterrupt:
            print("Interrupted—saving model before exit.")
            ppo_model.save("ppo_interrupted.zip")
            raise

        # Save the trained model
        ppo_model.save(os.path.join(get_project_root(), "results", "TurtleWorld", "PPO", "ppo_turtle"))
        print(f"PPO training complete ({total_timesteps} timesteps). Model saved.")

        # Generate video using the trained PPO model
        generate_ppo_video(ppo_model, "Best_Turtle_PPO.mp4")
        print("PPO policy rollout complete. Saved 'Best_Turtle_PPO.mp4'.")

    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose 'CMAES' or 'PPO'.")


if __name__ == "__main__":
    main()
