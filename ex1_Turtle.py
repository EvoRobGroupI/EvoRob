from src.EA.ES import ES, ES_opts
from src.EA.CMAES import CMAES, CMAES_opts
from src.world.robot.controllers import MLP
from src.utils.Filesys import get_project_root
from src.world.World import World

from stable_baselines3.ppo import PPO
import torch
import gymnasium as gym
import numpy as np
import os
from pathlib import Path
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.registration import register

import xml.etree.ElementTree as ET
import mujoco
import mujoco_viewer

ROOT_DIR = get_project_root()

ENV_KEY = os.getenv("ROBOT_ENV", "turtle")
XML_PATHS = {
    "turtle": Path("Turtle.xml"),
    "slider": Path("SliderEnv.xml"),
}

if ENV_KEY == "slider":
    tree = ET.parse(XML_PATHS["slider"])
    root = tree.getroot()
    worldbody = root.find("worldbody")
    for geom in worldbody.findall("geom"):
        if "pole" in geom.attrib.get("name", ""):
            worldbody.remove(geom)

    num_rows = 2
    num_cols = 20
    spacing_x = 0.7
    spacing_y = 0.7
    x_start = -spacing_x
    y_start = -spacing_y / 2

    for row in range(num_rows):
        for col in range(num_cols):
            x = x_start + col * spacing_x
            y = y_start + row * spacing_y
            pole_body = ET.SubElement(worldbody, "body")
            ET.SubElement(pole_body, "geom", {
                "type": "cylinder",
                "pos": f"{x:.2f} {y:.2f} 0.5",
                "size": "0.05 0.5",
                "rgba": "1 0.3 0.3 1",
                "name": f"pole_{row}_{col}"
            })

    inf_path = XML_PATHS["slider"].with_name("SliderEnv_inf.xml")
    tree.write(inf_path)

    with open("combined.xml", "w") as f:
        f.write(f"""
<mujoco model='slider_plus_turtle'>
  <include file='{inf_path}'/>
  <include file='{XML_PATHS['turtle']}'/>
</mujoco>
        """)
    XML_PATHS["slider"] = Path("combined.xml")

if ENV_KEY != "cheetah":
    class MujocoXMLWrapper(MujocoEnv):
        def __init__(self, xml, **kwargs):
            super().__init__(model_path=str(xml), frame_skip=5, **kwargs)

    env_id = f"{ENV_KEY}-xml-v0"
    if env_id not in gym.registry:
        register(id=env_id, entry_point=lambda **kw: MujocoXMLWrapper(XML_PATHS[ENV_KEY], **kw), max_episode_steps=1000)
    ENV_NAME = env_id
else:
    ENV_NAME = "HalfCheetah-v5"

class CheetahWorld(World):
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        action_space = self.env.action_space.shape[0]
        state_space = self.env.observation_space.shape[0]
        self.controller = MLP.NN_najaroController(state_space, action_space)
        self.dt = getattr(self.env, "dt", 0.01)
        self.n_params = self.controller.n_params

    def geno2pheno(self, genotype):
        self.controller.geno2pheno(genotype)
        return self.controller

    def evaluate_individual(self, genotype):
        trial_time = 50
        n_sim_steps = int(trial_time / self.dt)
        self.geno2pheno(genotype)
        rewards_list = []
        observations, info = self.env.reset()
        for _ in range(n_sim_steps):
            action = self.controller.get_action(observations)
            observations, rewards, terminated, truncated, info = self.env.step(action)
            rewards_list.append(rewards)
            if terminated or truncated:
                break
        return np.sum(rewards_list)

def run_EA(ea, world):
    env = gym.make(ENV_NAME)
    for gen in range(ea.n_gen):
        pop = ea.ask()
        fitnesses_gen = np.empty(ea.n_pop)
        env.reset()
        for index, genotype in enumerate(pop):
            fit_ind = world.evaluate_individual(genotype)
            fitnesses_gen[index] = fit_ind
        ea.tell(pop, fitnesses_gen)
    env.close()

def generate_best_individual_video(controller, video_name: str = "EvoRob_video.mp4"):
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    rewards_list = []
    observations, info = env.reset()
    frames = []
    for _ in range(1000):
        frames.append(env.render())
        action = controller.get_action(observations)
        observations, rewards, terminated, truncated, info = env.step(action)
        rewards_list.append(rewards)
        if terminated:
            break
    import imageio
    imageio.mimsave(video_name, frames, fps=30)
    env.close()

def main():
    world = CheetahWorld()
    n_parameters = world.n_params

    ES_opts["min"] = -1
    ES_opts["max"] = 1
    ES_opts["num_parents"] = 100
    ES_opts["num_generations"] = 100
    ES_opts["mutation_sigma"] = .5

    population_size = 50
    results_dir = os.path.join(ROOT_DIR, "results", ENV_KEY, "ES")
    ea = ES(population_size, n_parameters, ES_opts, results_dir)

    run_EA(ea, world)

    best_individual = np.load(os.path.join(results_dir, "99", "x_best.npy"))
    world.controller.geno2pheno(best_individual)
    generate_best_individual_video(world.controller, f"EA_best_{ENV_KEY}.mp4")

    env = gym.make(ENV_NAME)
    ppo = PPO("MlpPolicy", env, device=torch.device("cpu"))
    trial_time = 50
    n_sim_steps = int(trial_time / world.dt)
    n_total_steps = population_size * ES_opts["num_generations"] * n_sim_steps
    ppo.learn(total_timesteps=n_total_steps)
    ppo_controller = PPO_controller(ppo)

    rewards_list = []
    env = gym.make(ENV_NAME, render_mode="human")
    observations, info = env.reset()
    for _ in range(n_sim_steps):
        action = ppo_controller.get_action(observations)
        observations, rewards, terminated, truncated, info = env.step(action)
        rewards_list.append(rewards)

    generate_best_individual_video(ppo_controller, f"PPO_best_{ENV_KEY}.mp4")
    env.close()

class PPO_controller():
    def __init__(self, ppo: PPO):
        self.ppo = ppo
        self.state_space = ppo.observation_space
        self.action_space = ppo.action_space

    def get_action(self, state):
        state_tensor = torch.tensor(state[np.newaxis, :])
        action = self.ppo.policy(state_tensor)[0].squeeze().detach()
        return action.numpy()

if __name__ == "__main__":
    main()
