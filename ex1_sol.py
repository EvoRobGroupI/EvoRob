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

""" Large programming projects are often modularised in different components. 
    In the upcoming exercise(s) we will (re)build an evolutionary pipeline for robot evolution in MuJoCo.

    Exercise1 gym-env: Gym environments are simulated 'Worlds' where an agent receives 'observations' and performs
    'actions'. In this exercise we will optimise a MultiLayerPerceptron (MLP) to control our agent.    
"""

ROOT_DIR = get_project_root()
ENV_NAME = 'HalfCheetah-v5'


class PPO_controller():
    def __init__(self, ppo: PPO):
        self.ppo = ppo
        self.state_space = ppo.observation_space
        self.action_space = ppo.action_space

    def get_action(self, state):
        state_tensor = torch.tensor(state[np.newaxis, :])
        action = self.ppo.policy(state_tensor)[0].squeeze().detach()
        return action.numpy()


class CheetahWorld(World):

    def __init__(self):
        self.env = gym.make(ENV_NAME)
        action_space = self.env.action_space.shape[0] # https://gymnasium.farama.org/environments/mujoco/half_cheetah/#action-space
        state_space = self.env.observation_space.shape[0] #https://gymnasium.farama.org/environments/mujoco/half_cheetah/#observation-space
        self.controller = MLP.NNController(state_space, action_space)
        self.dt = self.env.get_wrapper_attr('dt')
        self.n_params = self.controller.n_params

    def geno2pheno(self, genotype):
        self.controller.geno2pheno(genotype)
        return self.controller

    def evaluate_individual(self, genotype):
        trial_time = 50  # seconds in simulation
        n_sim_steps = int(trial_time / self.dt)

        self.geno2pheno(genotype)

        rewards_list = []
        observations, info = self.env.reset()
        for step in range(n_sim_steps):
            action = self.controller.get_action(observations)
            observations, rewards, terminated, truncated, info = self.env.step(action)
            rewards_list.append(rewards)
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


def generate_best_individual_video(controller, video_name: str = 'EvoRob1_video.mp4'):
    # TODO: Make a video of the best individual, and plot the fitness curve.
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    rewards_list = []
    observations, info = env.reset()
    frames = []
    for step in range(1000):
        frames.append(env.render())
        action = controller.get_action(observations)
        observations, rewards, terminated, truncated, info = env.step(action)
        rewards_list.append(rewards)
        if terminated:
            break
    print(np.sum(rewards_list))

    import imageio
    imageio.mimsave(video_name, frames, fps=30)  # Set frames per second (fps)
    env.close()


def main():
    # TODO: understanding the world
    world = CheetahWorld()
    n_parameters = world.n_params

    # TODO: improve the ES settings
    ES_opts["min"] = -1
    ES_opts["max"] = 1
    ES_opts["num_parents"] = 100
    ES_opts["num_generations"] = 100
    ES_opts["mutation_sigma"] = .5

    population_size = 50

    results_dir = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'CMAES')
    ea = ES(population_size, n_parameters, ES_opts, results_dir)

    run_EA(ea, world)

    # %% Make video of best behaviour
    best_individual = np.load(os.path.join(results_dir, "99", "x_best.npy"))
    world.controller.geno2pheno(best_individual)

    generate_best_individual_video(world.controller, 'EA_best.mp4')

    # %% Compare with PPO
    env = gym.make(ENV_NAME)
    ppo = PPO("MlpPolicy", env, device=torch.device('cpu'))
    trial_time = 50  # seconds in simulation
    n_sim_steps = int(trial_time / world.dt)
    n_total_steps = population_size * ES_opts["num_generations"] * n_sim_steps
    ppo.learn(total_timesteps=n_total_steps)
    ppo_controller = PPO_controller(ppo)

    rewards_list = []
    env = gym.make(ENV_NAME, render_mode='human')
    observations, info = env.reset()
    for step in range(n_sim_steps):
        action = ppo_controller.get_action(observations)
        observations, rewards, terminated, truncated, info = env.step(action)
        rewards_list.append(rewards)

    # Make video
    generate_best_individual_video(ppo_controller, 'PPO_best.mp4')

    print(np.sum(rewards_list))
    env.close()


if __name__ == '__main__':
    main()