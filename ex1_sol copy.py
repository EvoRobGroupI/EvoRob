from src.EA.ES import ES, ES_opts
from src.EA.CMAES import CMAES, CMAES_opts
from src.world.robot.controllers import MLP
from src.utils.Filesys import get_project_root
from src.world.World import World

from stable_baselines3.ppo import PPO
import torch
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import spaces
import numpy as np
import os
import xml.etree.ElementTree as ET
import mujoco
import mujoco_viewer
import imageio

ROOT_DIR = get_project_root()
ENV_KEY = "SlidingTurtle"
XML_PATHS = {ENV_KEY: "SliderEnv_inf.xml"}


def create_pole_grid_xml():
    tree = ET.parse("SliderEnv.xml")
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("No <worldbody> element found in the XML.")

    # remove old poles
    for geom in list(worldbody.findall("geom")):
        if "pole" in geom.attrib.get("name", ""):
            worldbody.remove(geom)

    num_rows, num_cols = 2, 20
    spacing_x = spacing_y = 0.7
    x_start, y_start = -spacing_x, -spacing_y / 2

    for r in range(num_rows):
        for c in range(num_cols):
            x = x_start + c * spacing_x
            y = y_start + r * spacing_y
            body = ET.SubElement(worldbody, "body")
            ET.SubElement(body, "geom", {
                "type": "cylinder",
                "pos": f"{x:.2f} {y:.2f} 0.5",
                "size": "0.05 0.5",
                "rgba": "1 0.3 0.3 1",
                "name": f"pole_{r}_{c}"
            })

    tree.write("SliderEnv_inf.xml")

    # quick validation of the combined XMLs
    wrapper_xml = (
        "<mujoco model='slider_plus_turtle'>\n"
        "  <include file='SliderEnv_inf.xml'/>\n"
        "  <include file='Turtle.xml'/>\n"
        "</mujoco>"
    )
    mujoco.MjModel.from_xml_string(wrapper_xml)


create_pole_grid_xml()


class MujocoXMLWrapper(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, xml_path: str, **kwargs):
        model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(model)
        self.model = model
        obs_dim = model.nq + model.nv
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(model.nu,), dtype=np.float32)
        super().__init__(model_path=xml_path, frame_skip=5, observation_space=observation_space, action_space=action_space, **kwargs)

    # minimal observation: joint positions & velocities
    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def reset_model(self):
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs()

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, self.action_space.low, self.action_space.high)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = 0.0  # placeholder – customise as needed
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}


env_id = f"{ENV_KEY}-xml-v0"
if env_id not in gym.registry:
    register(id=env_id, entry_point=lambda **kw: MujocoXMLWrapper(XML_PATHS[ENV_KEY], **kw), max_episode_steps=1000)
ENV_NAME = env_id


class PPOController:
    def __init__(self, ppo: PPO):
        self.ppo = ppo

    def get_action(self, state):
        tensor_state = torch.tensor(state[None, :], dtype=torch.float32)
        action = self.ppo.policy(tensor_state)[0].squeeze().detach()
        return action.numpy()


class PillarWorld(World):
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        a_dim = self.env.action_space.shape[0]
        s_dim = self.env.observation_space.shape[0]
        self.controller = MLP.NNController(s_dim, a_dim)
        # fallback for timestep depending on gymnasium version
        self.dt = getattr(self.env.unwrapped.model.opt, "timestep", 0.01)
        self.n_params = self.controller.n_params

    def geno2pheno(self, genotype):
        self.controller.geno2pheno(genotype)
        return self.controller

    def evaluate_individual(self, genotype):
        trial_time = 50
        n_steps = int(trial_time / self.dt)
        self.geno2pheno(genotype)
        obs, _ = self.env.reset()
        total = 0.0
        for _ in range(n_steps):
            action = self.controller.get_action(obs)
            obs, rew, term, trunc, _ = self.env.step(action)
            total += rew
            if term or trunc:
                break
        return total


def run_EA(ea, world):
    for _ in range(ea.n_gen):
        pop = ea.ask()
        fitness = np.array([world.evaluate_individual(g) for g in pop])
        ea.tell(pop, fitness)


def generate_best_video(controller, filename: str = "EA_best.mp4"):
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    obs, _ = env.reset()
    frames, total = [], 0.0
    for _ in range(1000):
        frames.append(env.render())
        obs, rew, term, trunc, _ = env.step(controller.get_action(obs))
        total += rew
        if term or trunc:
            break
    print(total)
    imageio.mimsave(filename, frames, fps=30)
    env.close()


def main():
    world = PillarWorld()
    ES_opts.update({"min": -1, "max": 1, "num_parents": 100, "num_generations": 100, "mutation_sigma": 0.5})
    ea = ES(50, world.n_params, ES_opts, os.path.join(ROOT_DIR, "results", ENV_NAME, "CMAES"))
    run_EA(ea, world)
    best = np.load(os.path.join(ROOT_DIR, "results", ENV_NAME, "CMAES", "99", "x_best.npy"))
    world.controller.geno2pheno(best)
    generate_best_video(world.controller)


if __name__ == "__main__":
    main()
