# from src.EA.ES import ES, ES_opts
from src.EA.CMAES import CMAES, CMAES_opts
from src.world.robot.controllers import MLP
from src.utils.Filesys import get_project_root
from src.world.World import World

from stable_baselines3.ppo import PPO
import torch
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv

import numpy as np
import os

# Verify MuJoCo binding is available
try:
    import mujoco
except ImportError:
    raise RuntimeError("Could not import `mujoco`. Make sure MuJoCo Python bindings are installed.")


class TurtleGymEnv(MujocoEnv, utils.EzPickle):
    """
    A Gymnasium/MuJoCo environment that loads 'CombinedSliderTurtle.xml' from the project root.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"]
    }

    def __init__(self, render_mode=None, camera_name = None):
        xml_path = os.path.join(get_project_root(), "CombinedSliderTurtle.xml")
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"Cannot find XML at {xml_path}")

        frame_skip = 5

        m = mujoco.MjModel.from_xml_path(xml_path)
        obs_dim = int(m.nq + m.nv)
        act_dim = int(m.nu)

        from gymnasium.spaces import Box
    
        xml_path = os.path.join(get_project_root(), "CombinedSliderTurtle.xml")
        ...

        # build the spaces first
        obs_high  = np.inf * np.ones(obs_dim, dtype=np.float64)
        act_high  = np.ones(act_dim, dtype=np.float32)
        observation_space = Box(low=-obs_high, high=obs_high, dtype=np.float64)
        action_space      = Box(low=-act_high,  high=act_high,  dtype=np.float32)

        #  call MujocoEnv without `action_space`
        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            camera_name=camera_name,
        )

        # overwrite the action-space you want
        self.action_space = action_space
        utils.EzPickle.__init__(self)


    def _get_obs(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        x_pos = self.data.qpos[0]
        y_pos = self.data.qpos[1]
        x_vel = self.data.qvel[0]
        reward = x_vel + x_pos - 0.5*y_pos
        terminated = (x_vel < -0.05) or (abs(y_pos) > 1.0)
        truncated = False
        info = {
            "x_pos": x_pos,
            "y_pos": y_pos,
            "x_vel": x_vel
        }
        return obs, reward, terminated, truncated, info

    def reset_model(self):
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        if hasattr(self.data, "act") and self.data.act is not None:
            self.data.act[:] = 0.0  # Reset all actuators to zero
        return self._get_obs()


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
        total_reward = 0.0
        for _ in range(max_steps):
            action = self.controller.get_action(obs)
            obs, reward, done, _ = self.step(action)
            total_reward += reward
            if done:
                break
        return total_reward


def run_EA(ea, world):
    for gen in range(ea.n_gen):
        population = ea.ask()
        fitnesses = np.empty(ea.n_pop)
        for i, indiv in enumerate(population):
            fitnesses[i] = world.evaluate_individual(indiv)
        ea.tell(population, fitnesses)


def generate_best_individual_video(controller, video_name: str = "Turtle_video.mp4"):
    env = TurtleGymEnv(render_mode="rgb_array", camera_name = "topdown")
    obs, _ = env.reset()
    frames = []
    max_steps = 500

    for _ in range(max_steps):
        action = controller.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)

        # Tell MuJoCo to use the “follow” camera (which tracks COM from (0,-2,0.5))
        frame = env.render()
        frames.append(frame)

        if terminated or truncated:
            break

    import imageio
    imageio.mimsave(video_name, frames, fps=30)



def main():
    world = TurtleWorld()
    n_parameters = world.n_params

    CMAES_opts["min"] = -1
    CMAES_opts["max"] = 1
    CMAES_opts["num_parents"] = 50
    CMAES_opts["num_generations"] = 100
    CMAES_opts["mutation_sigma"] = 0.2

    population_size = 20
    results_dir = os.path.join(get_project_root(), "results", "TurtleWorld", "CMAES")
    ea = CMAES(population_size, n_parameters, CMAES_opts, results_dir)

    run_EA(ea, world)

    best_ind = np.load(os.path.join(results_dir, "99", "x_best.npy"))
    world.controller.geno2pheno(best_ind)
    generate_best_individual_video(world.controller, "Best_Turtle_Evo.mp4")

    print("Evolution complete. Saved 'Best_Turtle_Evo.mp4'.")


if __name__ == "__main__":
    main()
