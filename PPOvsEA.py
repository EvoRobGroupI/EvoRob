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

        frame_skip = 1
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

    def _get_obs(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
    
    # def _get_yaw(self):
    #     # MuJoCo’s base quaternion is in data.xquat[0] as [w, x, y, z]
    #     w, x, y, z = self.data.xquat[0]
    #     siny_cosp = 2.0 * (w * z + x * y)
    #     cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    #     return np.arctan2(siny_cosp, cosy_cosp)

    def step(self, action):

        # # Before simulating, record the “old” state
        # old_qpos = self.data.qpos.copy()
        # old_qvel = self.data.qvel.copy()


        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()

        # After simulating, compare qpos/qvel
        # new_qpos = self.data.qpos.copy()
        # new_qvel = self.data.qvel.copy()
        # print(
        #     "[DEBUG step] action:", action,
        #     "\n             old_qpos[:3]:", old_qpos[:3], "new_qpos[:3]:", new_qpos[:3],
        #     "\n             old_qvel[:3]:", old_qvel[:3], "new_qvel[:3]:", new_qvel[:3]
        # )

        # x_pos = new_qpos[0]
        # y_pos = new_qpos[1]
        # x_vel = new_qvel[0]

        x_pos = self.data.qpos[0]
        y_pos = self.data.qpos[1]
        x_vel = self.data.qvel[0]

        # yaw_rate = self.data.qvel[5]   # its within 5, max 3 really
        # yaw = self._get_yaw()
        # if abs(yaw) > 45: print(yaw)
        # print(yaw)

        reward = x_vel

        # reward = x_vel + x_pos# - 0.5 * abs(y_pos) - abs(yaw) # - 100 * abs(yaw_rate)
        # terminated = (x_vel < -0.05) or (abs(y_pos) > 3.0) #or abs(yaw) > 90 #or (abs(yaw_rate) > 2)

        terminated = False

        # terminated = (x_vel < -0.05) or (abs(y_pos) > 3.0) 
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
    #     x_pos = self.data.qpos[0]
    #     y_pos = self.data.qpos[1]
    #     x_vel = self.data.qvel[0]
    #     reward = x_vel + x_pos - 0.5*y_pos
    #     terminated = (x_vel < -0.05) or (abs(y_pos) > 1.0)
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
        max_steps = 500
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
def run_EA(ea, world):
    for gen in range(ea.n_gen):
        population = ea.ask()
        fitnesses = np.empty(ea.n_pop)
        for i, indiv in enumerate(population):
            fitnesses[i] = world.evaluate_individual(indiv)
        ea.tell(population, fitnesses)
    
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
        CMAES_opts["num_generations"] = 20
        CMAES_opts["mutation_sigma"] = 0.2

        population_size = 20
        results_dir = os.path.join(get_project_root(), "results", "TurtleWorld", "CMAES")
        os.makedirs(results_dir, exist_ok=True)

        ea = CMAES(population_size, n_parameters, CMAES_opts, results_dir)
        run_EA(ea, world)

        # Load best individual and generate video
        best_ind = np.load(os.path.join(results_dir, "39", "x_best.npy"))
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
