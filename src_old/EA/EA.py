import os
from typing import Dict

import numpy as np

from src.utils.Filesys import search_file_list

EA_opts = {
    "min": -4,
    "max": 4,
    "num_generations": 100,
    "tournament_size": 16,
    "mutation_prob": 0.3,
    "crossover_prob": 0.1,
}

class EA(object):
    def __init__(self, n_pop, n_params, opts: Dict = EA_opts, output_dir="./results"):
        self.n_params = n_params
        self.n_pop = n_pop
        self.n_gen = opts["num_generations"]
        self.tournament_size = opts["tournament_size"]
        self.min = opts["min"]
        self.max = opts["max"]

        self.current_gen = 0
        self.F = opts["mutation_prob"]
        self.Cr = opts["crossover_prob"]

        # % bookkeeping
        self.directory_name = output_dir
        self.full_x = []
        self.full_fitness = []
        self.x_best_so_far = None
        self.f_best_so_far = -np.inf
        self.x = None
        self.f = None

    def ask(self):
        if self.current_gen == 0:
            new_population = self.initialise_x0()
        else:
            new_population = []
            # Generate new population through crossover and mutation
            for _ in range(self.n_pop // 2):  # Produce pairs of children
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                offspring1, offspring2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(offspring1))
                new_population.append(self.mutate(offspring2))
        new_population = np.clip(new_population, self.min, self.max)
        return new_population

    def tell(self, solutions, function_values, save_checkpoint=True):
        #% Some bookkeeping
        self.full_fitness.append(function_values)
        self.full_x.append(solutions)
        self.f = function_values
        self.x = solutions

        if np.max(function_values) > self.f_best_so_far:
            best_index = np.argmax(function_values)
            self.f_best_so_far = function_values[best_index]
            self.x_best_so_far = solutions[best_index]


        if self.current_gen % 5 == 0:
            print(f"Generation {self.current_gen}:\t{self.f_best_so_far}\n"
                  f"Mean fitness:\t{self.f.mean()} +- {self.f.std()}\n"
                  )

        if save_checkpoint:
            self.save_checkpoint()
        self.current_gen += 1

    def initialise_x0(self):
        population = np.random.uniform(self.min, self.max, (self.n_pop, self.n_params))
        return population

    def select_parent(self):
        """Tournament selection: choose a random individual and return it."""
        tournament = np.random.choice(self.n_pop, self.tournament_size)
        tournament_ind = np.argmax(self.f[tournament])
        return self.x[tournament[tournament_ind]]

    def crossover(self, parent1, parent2):
        """Single-point crossover."""
        if np.random.random() < self.Cr:
            try:
                point = np.random.randint(1, self.n_params-1)
            except:
                point = self.n_params - 1
            return np.array([parent1[:point], parent2[point:]]).squeeze(), np.array([parent2[:point], parent1[point:]]).squeeze()
        else:
            return parent1, parent2

    def mutate(self, individual):
        """Mutate an individual by flipping bits with a given mutation rate."""
        for i in range(self.n_params):
            if np.random.random() < self.F:
                individual[i] = individual[i] + np.random.uniform(-1, 1)
        return individual

    def save_checkpoint(self):
        curr_gen_path = os.path.join(self.directory_name, str(self.current_gen))
        os.makedirs(curr_gen_path, exist_ok=True)
        np.save(os.path.join(self.directory_name, 'full_f'), np.array(self.full_fitness))
        np.save(os.path.join(self.directory_name, 'full_x'), np.array(self.full_x))
        np.save(os.path.join(curr_gen_path, 'f_best'), np.array(self.f_best_so_far))
        np.save(os.path.join(curr_gen_path, 'x_best'), np.array(self.x_best_so_far))
        np.save(os.path.join(curr_gen_path, 'x'), np.array(self.x))
        np.save(os.path.join(curr_gen_path, 'f'), np.array(self.f))

    def load_checkpoint(self):
        dir_path = search_file_list(self.directory_name, 'f_best.npy')
        assert len(dir_path) > 0;
        "No files are here, check the directory_name!!"

        self.current_gen = int(dir_path[-1].split('/')[-2])
        curr_gen_path = os.path.join(self.directory_name, str(self.current_gen))
        print(f"Loading from: {curr_gen_path}")
        self.full_fitness = np.load(os.path.join(self.directory_name, 'full_f.npy'))
        self.full_x = np.load(os.path.join(self.directory_name, 'full_x.npy'))
        self.f_best_so_far = np.load(os.path.join(curr_gen_path, 'f_best.npy'))
        self.x_best_so_far = np.load(os.path.join(curr_gen_path, 'x_best.npy'))
        self.x = np.load(os.path.join(curr_gen_path, 'x.npy'))
        self.f = np.load(os.path.join(curr_gen_path, 'f.npy'))
