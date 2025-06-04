import os
from typing import Dict

import numpy as np
import cma

from src.utils.Filesys import search_file_list

CMAES_opts = {
    "min": -4,
    "max": 4,
    "num_generations": 100,
    "mutation_sigma": 0.3,
}

class CMAES():
    def __init__(self, n_pop, n_params, opts: Dict = CMAES_opts, output_dir='./results/CMAES'):
        self.n_params = n_params
        self.n_pop = n_pop
        self.n_gen = opts["num_generations"]
        self.min = opts["min"]
        self.max = opts["max"]

        self.current_gen = 0
        self.current_mean = self.initialise_x0(n_params)
        self.current_sigma = opts["mutation_sigma"]
        self.f_new = np.empty(self.n_pop)

        self.cmaes = self.load_cmaes() #TODO

        #% bookkeeping
        self.directory_name = output_dir
        self.full_x = []
        self.full_fitness = []
        self.x_best_so_far = None
        self.f_best_so_far = -np.inf
        self.x = [None]*self.n_pop
        self.f = [-np.inf]*self.n_pop

    def load_cmaes(self):
        #TODO
        lower_bounds = ...  # lower bounds per dimension !! check dimensions
        upper_bounds = ...  # upper bounds per dimension
        cmaes_params = {
            'popsize': ...,
            'bounds': (lower_bounds, upper_bounds),
        }
        return cma.CMAEvolutionStrategy(..., ..., inopts=cmaes_params)

    def ask(self):
        #TODO
        new_population = ...
        return new_population

    def tell(self, solutions, function_values, save_checkpoint=True):
        #TODO
        self.cmaes.tell(..., ...)


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

    def initialise_x0(self, num_parameters):
        #TODO
        mean_vector = ...
        return mean_vector

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

        self.cmaes = self.load_cmaes()
        for x, f in zip(self.full_x, self.full_fitness):
            self.cmaes.tell(x, f)
