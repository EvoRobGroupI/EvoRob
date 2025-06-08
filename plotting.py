import matplotlib.pyplot as plt
import numpy as np
import os
from src.utils.Filesys import get_project_root

ROOT_DIR = get_project_root()
ENV_NAME = 'TurtleWorld'

results_no_sens = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'CMAES')
results_yes_sens= os.path.join(ROOT_DIR, 'results', ENV_NAME, 'CMAES')


fitnesses_es = np.load(os.path.join(results_no_sens, 'full_f.npy'))
fitnesses_cmaes = np.load(os.path.join(results_no_sens, 'full_f.npy'))

mean_f_es = np.mean(fitnesses_es, axis=1)
mean_f_cmaes = np.mean(fitnesses_cmaes, axis=1)

std_f_es = np.std(fitnesses_es, axis=1)
std_f_cmaes = np.std(fitnesses_cmaes, axis=1)

gens = np.arange(0, 50, 1)
plt.plot(gens, mean_f_es, color='k', label='ES')
plt.plot(gens, mean_f_cmaes, color='r', label='CMAES')
plt.fill_between(gens, mean_f_es - std_f_es, mean_f_es + std_f_es, color='k', alpha=0.5)
plt.fill_between(gens, mean_f_cmaes - std_f_cmaes, mean_f_cmaes + std_f_cmaes, color='r', alpha=0.5)
plt.legend(loc='best')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
# plt.savefig('my_world_f.pdf')