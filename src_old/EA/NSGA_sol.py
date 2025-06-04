import copy
import os
from typing import Dict

import numpy as np

from src.utils.Filesys import search_file_list

NSGA_opts = {
    "min": -4,
    "max": 4,
    "num_parents": 16,
    "num_generations": 100,
    "mutation_prob": 0.3,
    "crossover_prob": 0.1,
}


class NSGAII_sol():
    def __init__(self, n_pop, n_params, opts: Dict = NSGA_opts, output_dir: str = "./results/NSGAII"):
        """
        Evolutionary Strategy [COMPLETE]

        :param n_pop: population size
        :param n_params: number of parameters
        :param opts: algorithm options
        :param output_dir: output directory Default = "./results/NSGAII"
        """
        # % EA options
        self.n_params = n_params
        self.n_pop = n_pop
        self.n_gen = opts["num_generations"]
        self.n_parents = opts["num_parents"]
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
        self.f_best_so_far = [[-np.inf]*2]
        self.x = None
        self.f = None


    def ask(self):
        if self.current_gen==0:
            new_population = self.initialise_x0()
        else:
            new_population = self.create_children(self.n_pop)
        new_population = np.clip(new_population, self.min, self.max)
        return new_population

    def tell(self, solutions, function_values, save_checkpoint=True):
        parents_population, parents_fitness = self.sort_and_select_parents(
            solutions, function_values, self.n_parents
        )


        #% Some bookkeeping
        self.full_fitness.append(function_values)
        self.full_x.append(solutions)
        self.x = parents_population
        self.f = parents_fitness

        aggregate_vals = np.sum(function_values, axis=1)
        if np.max(aggregate_vals) > np.sum(self.f_best_so_far):
            best_index = np.argmax(aggregate_vals)
            self.f_best_so_far = function_values[best_index]
            self.x_best_so_far = solutions[best_index]

        if self.current_gen % 5 == 0:
            print(f"Best fitness in generation {self.current_gen}: {self.f_best_so_far}\n"
                  f"Mean pop fitness: {self.f.mean()} +- {self.f.std()}\n"
                  )

        if save_checkpoint:
            self.save_checkpoint()
        self.current_gen += 1



    def initialise_x0(self):
        return np.random.uniform(low=self.min, high=self.max, size=(self.n_pop, self.n_params))

    def create_children(self, population_size):
        new_offspring = np.empty((population_size, self.n_params))
        for i in range(population_size):
            r0 = i
            while (r0 == i):
                r0 = np.floor(np.random.random() * self.n_pop).astype(int)
            r1 = r0
            while (r1 == r0 or r1 == i):
                r1 = np.floor(np.random.random() * self.n_pop).astype(int)
            r2 = r1
            while (r2 == r1 or r2 == r0 or r2 == i):
                r2 = np.floor(np.random.random() * self.n_pop).astype(int)

            jrand = np.floor(np.random.random() * population_size).astype(int)

            for j in range(self.n_params):
                if (np.random.random() <= self.Cr or j == jrand):
                    # Mutation
                    new_offspring[i][j] = copy.deepcopy(self.x[r0][j] + self.F * (self.x[r1][j] - self.x[r2][j]))
                else:
                    new_offspring[i][j] = copy.deepcopy(self.x[r0][j])

        mutated_population = np.clip(new_offspring, self.min, self.max)
        return mutated_population



    def sort_and_select_parents(self, solutions, function_values, n_parents):
        fronts, population_rank = self.fast_nondominated_sort(function_values)

        prob_weight = len(fronts) - np.array(population_rank)
        dist = prob_weight / np.sum(prob_weight)

        draw_ind = np.random.choice(np.arange(len(solutions)), n_parents,
                      p=dist)
        return solutions[draw_ind], function_values[draw_ind]


    def dominates(self, individual, other_individual):
        return all(x >= y for x, y in zip(individual, other_individual)) and any(
            x > y for x, y in zip(individual, other_individual)
        )

    def fast_nondominated_sort(self, fitness):
        domination_lists = [[] for _ in range(len(fitness))]
        domination_counts = [0 for _ in range(len(fitness))]
        population_rank = [0 for _ in range(len(fitness))]
        pareto_fronts = [[]]

        for individual_a in range(len(fitness)):
            for individual_b in range(len(fitness)):
                # does candidate 1 dominate candidate 2?
                if self.dominates(fitness[individual_a], fitness[individual_b]):
                    # append index of dominating solution
                    domination_lists[individual_a].append(individual_b)

                # does candidate 2 dominate candidate 1?
                elif self.dominates(fitness[individual_b], fitness[individual_a]):
                    #
                    domination_counts[individual_a] += 1

            # if solution dominates all
            if domination_counts[individual_a] == 0:
                # placeholder solution rank
                population_rank[individual_a] = 0

                # add solution to first Pareto front
                pareto_fronts[0].append(individual_a)

        # iterates until there are no more items appended in the last front
        i = 0
        while pareto_fronts[i]:
            # open next front
            next_front = []

            # iterate through all items in previous front
            for individual_a in pareto_fronts[i]:
                # check all other items which are dominated by this item
                for individual_b in domination_lists[individual_a]:
                    # reduce domination count
                    domination_counts[individual_b] -= 1

                    # every now nondominated item are append to next front
                    if domination_counts[individual_b] == 0:
                        # add solution rank
                        population_rank[individual_b] = i + 1
                        next_front.append(individual_b)

            i += 1

            pareto_fronts.append(next_front)

        # removes last empty front
        pareto_fronts.pop()

        return pareto_fronts, population_rank

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

        self.full_fitness = np.load(os.path.join(self.directory_name, 'full_f.npy'))
        self.full_x = np.load(os.path.join(self.directory_name, 'full_x.npy'))
        self.f_best_so_far = np.load(os.path.join(curr_gen_path, 'f_best.npy'))
        self.x_best_so_far = np.load(os.path.join(curr_gen_path, 'x_best.npy'))
        self.x = np.load(os.path.join(curr_gen_path, 'x.npy'))
        self.f = np.load(os.path.join(curr_gen_path, 'f.npy'))


