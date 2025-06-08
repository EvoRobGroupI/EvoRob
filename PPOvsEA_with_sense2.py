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
import io, math, random, xml.etree.ElementTree as ET
# Verify MuJoCo binding is available
try:
    import mujoco
except ImportError:
    raise RuntimeError("Could not import `mujoco`. Make sure MuJoCo Python bindings are installed.")
from mujoco import mj_name2id, mjtObj
from pathlib import Path
# -----------------------------------------------------------------------------
# Environment definition (unchanged)
# -----------------------------------------------------------------------------

TEMPLATE_XML = Path("CombinedSliderTurtle.xml")   # same dir as your script


# ----------------------------------------------------------------------
# 1. Course builder that starts from the *merged* XML
# ----------------------------------------------------------------------
class TurtleCourseMakerCombined:
    def __init__(self, jitter_range=(0.0, 1.5)):
        self.jitter_range = jitter_range
        self._base_tree   = ET.parse(TEMPLATE_XML)
        self._base_root   = self._base_tree.getroot()

        # cache the <geom> elements that belong to poles once
        self._pole_geoms = [
            g for g in self._base_root.findall(".//geom")
            if g.get("name", "").startswith("pole_")
        ]
        if not self._pole_geoms:
            raise RuntimeError("No <geom name='pole_…'> found in template!")

    # -------------------------------------------------------------- #
    def build_xml_string(self, jitter: float) -> str:
        """Return an XML *string* with pillar X-coords jittered by ±jitter."""
        # deep-copy the whole tree first (cheap)
        root = ET.fromstring(ET.tostring(self._base_root, encoding="unicode"))

        # find the corresponding geoms in the cloned tree
        pole_geoms = [
            g for g in root.findall(".//geom")
            if g.get("name", "").startswith("pole_")
        ]
        assert len(pole_geoms) == len(self._pole_geoms)

        for g in pole_geoms:
            pos = g.get("pos").split()
            x   = float(pos[0]) + random.uniform(-jitter, +jitter)
            pos[0] = f"{x:.2f}"
            g.set("pos", " ".join(pos))

        vis = root.find("visual") or ET.SubElement(root, "visual")
        glob = vis.find("global") or ET.SubElement(vis, "global")
        glob.set("offwidth",  "640")      # match what you ask for in render()
        glob.set("offheight", "640")      # anything ≥ requested height
        return ET.tostring(root, encoding="unicode")


# ----------------------------------------------------------------------
# 2.  Gymnasium MuJoCo environment
# ----------------------------------------------------------------------
class TurtleGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, maker=None, jitter_range=(0.0, 1.5), **ignored):
        super().__init__()
        self.maker = maker or TurtleCourseMakerCombined(jitter_range)

        # build one nominal model to size the spaces
        xml  = self.maker.build_xml_string(jitter=0.0)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data  = mujoco.MjData(self.model)

        nq, nv, nu = self.model.nq, self.model.nv, self.model.nu
        self.N_RAYS = 21
        self.observation_space = Box(-np.inf, np.inf,
                                     shape=(nq + nv + self.N_RAYS,),
                                     dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, shape=(nu,), dtype=np.float32)

        self._cache_ids()

    # -------------------------------------------------------------- #
    def _cache_ids(self):
        sid      = mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, "rf_-80")
        self.rf_adr = self.model.sensor_adr[sid]
        
    def render(self, width=640, height=480, camera="topdown"):
        if (self._renderer is None or
            self._renderer.width  != width or
            self._renderer.height != height):
            self._renderer = mujoco.Renderer(self.model, width, height)

        self._renderer.update_scene(self.data, camera=camera)
        frame = self._renderer.render()
        frame = np.flipud(frame).transpose(1, 0, 2).copy()   # (H,W,3) uint8
        return frame



    
    # -------------------------------------------------------------- #
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)

        jitter = np.random.uniform(*self.maker.jitter_range)
        xml    = self.maker.build_xml_string(jitter)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data  = mujoco.MjData(self.model)
        self._cache_ids()

        self._renderer = None          # ✨ <-- add this line
        return self._get_obs(), {}


    def _get_obs(self):
        q = self.data.qpos.ravel().astype(np.float32)
        v = self.data.qvel.ravel().astype(np.float32)
        ranges = self.data.sensordata[
            self.rf_adr : self.rf_adr + self.N_RAYS].astype(np.float32)
        ranges = np.clip(ranges, 0.0, 6.0) / 6.0
        return np.concatenate([q, v, ranges])

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
        self.data.ctrl[:] = np.clip(action, -1, 1)
        mujoco.mj_step(self.model, self.data)

        x_pos, y_pos, x_vel = self.data.qpos[0], self.data.qpos[1], self.data.qvel[0]
        yaw = self._get_yaw()
        yaw_penalty = -0.01 * abs(yaw)      # ≈ −1 per 100 deg
        reward = x_vel + 0.1*x_pos - 0.2*abs(y_pos) + yaw_penalty

        # reward      = x_vel + 0.1 * x_pos - 0.2 * abs(y_pos)

        terminated  = abs(y_pos) > 3 or x_pos < -1
        return self._get_obs(), reward, terminated, False, {}

    # def render(self, *_, **__):
    #     raise NotImplementedError("headless env")


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

    # def evaluate_individual(self, genotype):
    #     max_steps = 1000
    #     self.geno2pheno(genotype)
    #     obs = self.reset()

    #     steps_taken = 0

    #     total_reward = 0.0
    #     for _ in range(max_steps):
    #         action = self.controller.get_action(obs)
    #         obs, reward, done, _ = self.step(action)
    #         total_reward += reward
    #         steps_taken += 1
    #         if done:
                # break

        # print(f"[DEBUG] genotype rollout finished: steps_taken = {steps_taken}/{max_steps}")
    

        # return total_reward

    def evaluate_individual(self, genotype, k: int = 4) -> float:
        """
        Run the same genotype `k` times on freshly reset courses and
        return the *average* total reward.  k = 4 by default.
        """
        # 1) load the weights into the controller once
        self.geno2pheno(genotype)

        episode_rewards = []
        max_steps = 1_000

        for _ in range(k):
            obs, _ = self.env.reset()          # new jittered course
            total = 0.0

            for _ in range(max_steps):
                action   = self.controller.get_action(obs)
                obs, r, done, _ = self.step(action)
                total += r
                if done:
                    break

            episode_rewards.append(total)

        # arithmetic mean across k rollouts
        return float(np.mean(episode_rewards))
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

def run_EA(ea: CMAES, world: TurtleWorld, k: int = 4):
    for gen in range(ea.n_gen):
        population = ea.ask()
        fitnesses  = np.empty(ea.n_pop, dtype=np.float64)

        for i, indiv in enumerate(population):
            fitnesses[i] = world.evaluate_individual(indiv, k=k)

        ea.tell(population, fitnesses)

        best = fitnesses.max()
        mean = fitnesses.mean()
        std  = fitnesses.std(ddof=0)
        print(f"Gen {gen:3d} | best {best:8.1f} | mean {mean:7.1f} ± {std:6.1f}")

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
import imageio

def generate_ea_video(controller, video_name: str = "Turtle_EA.mp4"):
    env, frames = TurtleGymEnv(), []
    obs, _      = env.reset()

    for _ in range(1_000):
        action = controller.get_action(obs)
        obs, _, done, _, _ = env.step(action)
        frames.append(env.render())           # returns ready-to-encode uint8

        if done:
            obs, _ = env.reset()

    imageio.mimsave(
        video_name,
        frames,
        fps               = 30,
        macro_block_size  = None,      # suppresses FFMPEG “multiple of 16” warning
        codec             = "libx264"  # RGB→YUV conversion handled automatically
    )


# import imageio
# # import numpy as np                 # make sure this is imported once at top

# def generate_ea_video(controller, video_name="Turtle_EA.mp4"):
#     env    = TurtleGymEnv()         # no kwargs
#     obs, _ = env.reset()
#     frames = []

#     for _ in range(1_000):
#         # 1 ) act & step
#         action = controller.get_action(obs)
#         obs, _, done, _, _ = env.step(action)

#         # 2 ) render, flip (bottom-up → top-down), make contiguous
#         frame = env.render()                # uint8 (H, W, 3) with negative stride
#         frame = np.flipud(frame).copy()     # positive stride, encoder-friendly
#         frames.append(frame)

#         if done:
#             obs, _ = env.reset()

#     imageio.mimsave(video_name, frames, fps=30)


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
        print(frame.strides[0])
        frame = np.flipud(frame).copy() 
        # frame = (255.0 * frame).astype(np.uint8)
        frames.append(frame)
        if terminated or truncated:
            break

    import imageio
    imageio.mimsave(video_name, frames, fps=30,
    macro_block_size=None,   # avoids a warning for non-multiple-16 sizes
    format='FFMPEG',         # backend that understands float input
    codec='libx264rgb', )


# -----------------------------------------------------------------------------
# Main entry point with algorithm switch
# -----------------------------------------------------------------------------
def main():
    env = TurtleGymEnv()
    obs, _ = env.reset()
    frame  = env.render()          # should look perfect in e.g. matplotlib imshow
    print(frame.shape, frame.dtype)   # → (480, 640, 3) uint8
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
        CMAES_opts["num_generations"] = 5
        CMAES_opts["mutation_sigma"] = 0.5

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