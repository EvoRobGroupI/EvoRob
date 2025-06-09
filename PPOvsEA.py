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

    # def step(self, action):

    #     old_qpos = self.data.qpos.copy()
    #     old_qvel = self.data.qvel.copy()

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
    #     reward = x_vel*direction_adj + 2*x_pos

    #     truncated = False
    #     info = {
    #         "x_pos": x_pos,
    #         "y_pos": y_pos,
    #         "x_vel": x_vel
    #     }
    #     return obs, reward, terminated, truncated, info
    
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
        max_steps = 1000
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
    
#     # Print summary before telling EA
#         # best_idx = np.argmax(fitnesses)
#         # worst_idx = np.argmin(fitnesses)
#         # print(
#         #     f"[EA gen {gen}] "
#         #     f"best fitness = {fitnesses[best_idx]:.2f}, "
#         #     f"worst fitness = {fitnesses[worst_idx]:.2f}"
        # )

from joblib import Parallel, delayed             # (pip install joblib)
import multiprocessing as mp

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
        CMAES_opts["num_generations"] = 150
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