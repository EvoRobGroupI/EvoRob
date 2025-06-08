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

        # ────── 1) Load the MuJoCo model in order to extract nq, nv, nu ──────
        frame_skip = 2
        raw_model = mujoco.MjModel.from_xml_path(xml_path)

        self.prev_potential = None
        self.k_shaping     = 0.5   # scaling factor for how strong you want the pillar signal
        self.gamma         = 0.99  # discount for potential shaping

        # Extract dimensions and set N_RAYS = 21
        nq = int(raw_model.nq)
        nv = int(raw_model.nv)
        nu = int(raw_model.nu)
        self.N_RAYS = 21

        # Now the true obs_dim must include those 21 range‐finder readings:
        obs_dim = nq + nv + self.N_RAYS
        act_dim = nu

        # Build the (correctly sized) observation and action spaces (use float32)
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

        # ────── 2) Initialize MujocoEnv, which sets self.model and self.data ──────
        super().__init__(
            xml_path,
            frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            camera_name=camera_name,
        )
        self.action_space = action_space

        # ────── 3) Locate the very first “rf_-80” sensor to get its address ──────
        first_rf_sid = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            "rf_-80"
        )
        if first_rf_sid == -1:
            raise RuntimeError("Sensor 'rf_-80' not found; check CombinedSliderTurtle.xml")
        self.rf_adr = self.model.sensor_adr[first_rf_sid]

        # ────── 4) (Optional) Print the turtle’s body ID or any debug info ──────
        # print(self.data.body("turtle").id)
        self.once = True
        utils.EzPickle.__init__(self)



    def _get_obs(self):

        # ————————— 1) Flatten qpos, qvel —————————
        q = self.data.qpos.ravel().astype(np.float32)   # shape (nq,)
        v = self.data.qvel.ravel().astype(np.float32)   # shape (nv,)

        # ————————— 2) Turtle’s position & heading —————————
        # tpos = self.data.xpos[self.turtle_body_id]        # (3,)
        # fwd  = self.turtle_forward_vector()               # (3,)

        # Build a horizontal “right” vector
        # up    = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        # right = np.cross(fwd, up)
        # if np.linalg.norm(right) < 1e-6:
        #     right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # else:
        #     right = right / np.linalg.norm(right)

        # ray_origin = tpos.copy()                              # copy the 3‐vector
        # ray_origin[2] = tpos[2] - 0.2   # shift 0.6 m above shell midpoint

        distances = self.data.sensordata[self.rf_adr : self.rf_adr + self.N_RAYS]

        MAX_SENSE = 6.0
        dist_clipped = np.where((distances < 0) | (distances > MAX_SENSE),
                                MAX_SENSE,
                                distances)              # shape (21,)
        d_norm_inv = 1 - dist_clipped / MAX_SENSE  

        obs = np.concatenate([q, v, d_norm_inv.astype(np.float32)], axis=0)
        # print("rays (m):", np.round(d_norm_inv, 2))
        return obs


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

        rays   = obs[-self.N_RAYS:] 
        x_pos = self.data.qpos[0]
        y_pos = self.data.qpos[1]
        x_vel = self.data.qvel[0]

        yaw = self._get_yaw()

        terminated = (x_pos < -1) or (abs(y_pos) > 3.0)

        milestone = int(x_pos*2 // 1)
        if milestone > getattr(self, "_last_ms", -1):
            self._last_ms = milestone
            r_milestone = 1.0
            # if x_pos > 0.1:
                # print(f"last milestone: {milestone}")
                # print(f"milestone achieved: {x_pos}")
        else:
            # print("no milestone")
            r_milestone = 0.0
        direction_adj = 1 - 2*(abs(yaw)+0.01)/90

        beam_angles = np.linspace(-80, 80, self.N_RAYS)     # in degrees
        max_angle   = np.max(np.abs(beam_angles))           # =80

        # raw weight ∝ |angle| so center=0, edges=1
        raw_w = np.abs(beam_angles) / max_angle 
        weights = raw_w / np.sum(raw_w)                     # sum(weights)=1

        # 3) In step(), after obs and rays = obs[-self.N_RAYS:]
        phi = self.k_shaping * np.dot(rays, weights)
        # phi = self.k_shaping * np.sum(rays)
        if (self.prev_potential is None) or (x_vel < 0):
            F = 0.0
        else:
            F = np.max(self.gamma * phi - self.prev_potential, 0)
            self.prev_potential = phi

        

        phi_norm = phi / self.k_shaping
        g = 1 + 2*(1 - phi_norm)

        v_thresh = 0.02      # ≈ 1 cm/s
        k_still  = 0.5       # penalty magnitude

        # … after you’ve got x_vel …
        if abs(x_vel) < v_thresh:
            r_still = -k_still
        else:
            r_still = 0.0

        reward = 0.3*g*np.max(x_vel,0)*np.cos(np.radians(yaw)) + r_milestone + F - 0.5*abs(y_pos) + r_still 
        # + np.random.normal(0, 0.5)
        # if reward > 3:
        #     print(f"reward: {reward}")

        #     print(f"xvelxdir: {x_vel*direction_adj}")
        #     print(f"xvel: {x_vel}")
        # print(f"direction_adj: {direction_adj}")
        # print(f"xpos: {x_pos}")
        # print(f"yaw: {yaw}")
        truncated = False
        info = {
            "x_pos": x_pos,
            "y_pos": y_pos,
            "x_vel": x_vel
        }

        return obs, reward, terminated, truncated, info
    
    # def step(self, action):
    #     self.do_simulation(action, self.frame_skip)
    #     obs = self._get_obs()

    #     # ───────── basic kinematics ──────────────────
    #     x_pos  = self.data.qpos[0]
    #     y_pos  = self.data.qpos[1]
    #     x_vel  = self.data.qvel[0]
    #     yaw    = self._get_yaw()            # deg  (–180 … +180)
    #     xv_abs = abs(x_vel)

    #     # ───────── termination ───────────────────────
    #     terminated = (x_pos < -6.0) or (abs(y_pos) > 3.0)

    #     # ───────── reward terms ──────────────────────
    #     # 1) raw forward speed (m/s)            →   + x_vel
    #     r_speed = x_vel

    #     # 2) *progress bonus* → +1.0 every time you surpass a 1 m milestone
    #     #    Gives a sparse kick so CMA-ES/PPO cannot exploit just “wiggling”.
    #     milestone = int(x_pos // 1)          # metres already crossed
    #     if milestone > getattr(self, "_last_ms", -1):
    #         self._last_ms = milestone
    #         r_milestone = +1.0
    #     else:
    #         r_milestone = 0.0

    #     # 3) lane penalty for side drift (y)            →   –|y|
    #     r_drift = -0.25 * abs(y_pos)

    #     # 4) yaw penalty (keeps turtle pointing fwd)    →   –|yaw|
    #     r_yaw   = -0.01 * abs(yaw)          # 0.01 ≈ 1 deg = –0.01

    #     # 5) actuator-smoothness penalty (optional)
    #     # r_smooth = -0.1 * np.sum(np.square(action))
    #     direction_adj = 1 - 2*(abs(yaw)+0.01)/90
    # #     reward = x_vel*direction_adj
    #     # ───────── combine, then clip ────────────────
    #     reward = r_speed + r_milestone + r_drift + r_yaw
    #     # reward = np.clip(reward, -5.0, +5.0)

    #     # ───────── done  ─────────────────────────────
    #     truncated = False
    #     info = dict(
    #         x_pos=x_pos, y_pos=y_pos, x_vel=x_vel,
    #         r_speed=r_speed, r_milestone=r_milestone,
    #         r_drift=r_drift, r_yaw=r_yaw
    #     )
    #     return obs, reward, terminated, truncated, info


    def reset_model(self):
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        self.prev_potential = None
        if hasattr(self.data, "act") and self.data.act is not None:
            self.data.act[:] = 0.0  # Reset all actuators to zero
        return self._get_obs()
    # def reset_model(self):
    #     # look up the "tucked" keyframe
    #     key_id = mujoco.mj_name2id(
    #         self.model,
    #         mujoco.mjtObj.mjOBJ_KEY,
    #         "tucked"
    #     )
    #     # reset the entire state (qpos, qvel, etc.) to that keyframe
    #     mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
    #     return self._get_obs()




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
        return obs, reward, done, info

    def geno2pheno(self, genotype):
        self.controller.geno2pheno(genotype)
        return self.controller

    def evaluate_individual(self, genotype):
        max_steps = 1200
        self.geno2pheno(genotype)
        obs = self.reset()

        # Print the very first action(s):
        # action_sample = self.controller.get_action(obs)
        # print("  [DEBUG] initial action:", action_sample)
        steps_taken = 0

        total_reward = 0.0
        for _ in range(max_steps):
            action = self.controller.get_action(obs)
            obs, reward, done, _ = self.step(action)
            total_reward += reward
            steps_taken += 1
            if done:
                break

        # print(f"[DEBUG] genotype rollout finished: steps_taken = {steps_taken}/{max_steps}")
    

        return total_reward


# -----------------------------------------------------------------------------
# EA runner (unchanged)
# -----------------------------------------------------------------------------
# def run_EA(ea, world):
#     for gen in range(ea.n_gen):
#         population = ea.ask()
#         fitnesses = np.empty(ea.n_pop)
#         for i, indiv in enumerate(population):
#             fitnesses[i] = world.evaluate_individual(indiv)
#         ea.tell(population, fitnesses)
    
    # Print summary before telling EA
        # best_idx = np.argmax(fitnesses)
        # worst_idx = np.argmin(fitnesses)
        # print(
        #     f"[EA gen {gen}] "
        #     f"best fitness = {fitnesses[best_idx]:.2f}, "
        #     f"worst fitness = {fitnesses[worst_idx]:.2f}"
        # )

# top of the file ────────────────────────────────────────────────
from joblib import Parallel, delayed             # (pip install joblib)
import multiprocessing as mp

# …

def run_EA(ea: CMAES, world: TurtleWorld):
    n_jobs = mp.cpu_count()          # or set e.g. 8, 32, …

    # -- one helper so that *each* process instantiates its own env --------
    def evaluate_in_subprocess(genotype: np.ndarray):
        # Re-create a fresh TurtleWorld *inside* the subprocess
        local_world = TurtleWorld()           # <= 2-3 ms, cheap
        return local_world.evaluate_individual(genotype)

    # -- evolutionary loop -------------------------------------------------
    for gen in range(ea.n_gen):
        population  = ea.ask()

        fitnesses   = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(evaluate_in_subprocess)(ind) for ind in population
        )
        fitnesses   = np.asarray(fitnesses, dtype=np.float64)

        ea.tell(population, fitnesses)

        # print(f"Gen {gen:3d}  best = {fitnesses.max():8.1f}   "
        #       f"mean = {fitnesses.mean():7.1f}")

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
        CMAES_opts["num_parents"] = 30
        # CMAES_opts["num_generations"] = 150
        # CMAES_opts["mutation_sigma"] = 0.5
        CMAES_opts["num_generations"] = 200
        CMAES_opts["mutation_sigma"]  = 0.5        # start twice as large
        CMAES_opts["sigma_restart"]   = 0.3        # grow back if stuck
        CMAES_opts["tolx"]            = 1e-12      # disable premature stop

        population_size = 150
        results_dir = os.path.join(get_project_root(), "results", "TurtleWorld", "CMAES")
        # os.makedirs(results_dir, exist_ok=True)
        if os.path.isdir(results_dir):
            import shutil
            shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        # ea = CMAES(population_size, n_parameters, CMAES_opts, results_dir)
        #######################################################
        # --- grab Xavier‐initialized weights as the CMA-ES starting mean ---
        model = world.controller.model
        lin_flat = model.lin.ravel()
        out_flat = model.output.ravel()
        # if you added biases b1,b2 in your network, include them too:
        try:
            b1_flat = model.b1.ravel()
            b2_flat = model.b2.ravel()
            init_mean = np.concatenate([lin_flat, b1_flat, out_flat, b2_flat])
        except AttributeError:
            init_mean = np.concatenate([lin_flat, out_flat])

        ea = CMAES(
            population_size,
            n_parameters,
            CMAES_opts,
            results_dir,
            init_mean=init_mean
        )
        #########################################################
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