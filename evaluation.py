from copy import deepcopy
import numpy as np
from datetime import datetime, timedelta
from math import isclose
from pathlib import Path
from multiprocessing import Pool, cpu_count, set_start_method
from matplotlib import pyplot as plt

from demo_controller import player_controller

rng = np.random.default_rng(int(datetime.now().timestamp()))

import sys
import os

sys.path.insert(0, 'evoman')
from evoman.environment import Environment

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

n_hidden_neurons = 10

experiment_name = 'evaluation'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

import platform

# initializes simulation in multi evolution mode, for multiple static enemies.
env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  multiplemode="no",
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  logs="off"
                  )


# read in weights
def initialise_weights(group, method):
    # return 10 weight arrays
    dir_path = Path(f"handmade_results/{method}/{group}//best")
    weights = []
    for root, dirs, files in os.walk(dir_path):
        weights = [np.loadtxt(os.path.join(root, f), delimiter=',') for f in files if 'csv' in f]
    return weights


def evaluate(group, method, num_workers, pool):
    all_weights = initialise_weights(group, method)
    all_gains = []

    jobs = []
    d = int(len(all_weights) // num_workers)

    for i in range(num_workers):
        start = i * d
        end = ((i + 1) * d)
        if i == (num_workers - 1):
            jobs.append(pool.apply_async(evaluate_weights, (all_weights[start:], )))
        else:
            jobs.append(pool.apply_async(evaluate_weights, (all_weights[start:end],)))

    for job in jobs:
        all_gains.extend(job.get())

    plot_and_save(all_gains, group, method)


def plot_and_save(gains, group, method):
    path = Path(f"handmade_results/plots/{method}/{group}/")
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    np.savetxt(f'{path}/gains.csv', gains, delimiter=',')
    plt.boxplot(gains)
    plt.title(f'Group {group} with {method}')
    plt.xticks([])
    plt.ylabel("Individual Gain")

    plt.savefig(path.joinpath('ind_gain.pdf'))
    plt.savefig(path.joinpath('ind_gain.png'))


def evaluate_weights(weight_list):
    gains = []
    for weights in weight_list:
        gains.append(np.mean([evaluate_individual(weights) for _ in range(5)]))
    return gains


def evaluate_individual(weights):
    gains = []
    for enemy in [1, 2, 3, 4, 5, 6, 7, 8]:
        env.enemies = [enemy]
        f, p, e, t = env.play(pcont=weights)
        gain = p - e
        gains.append(gain)
    return np.mean(gains)


if __name__ == '__main__':
    if platform.system() == 'Darwin':
        set_start_method('spawn')  # Comment this if not on MACOS

    try:
        cpus = cpu_count() - int(sys.argv[1])
        group = sys.argv[2]
        method = " ".join(sys.argv[3:])
        pool = Pool(cpus)
        evaluate(group, method, cpus, pool)
    except Exception:
        raise ValueError("Requires args <less cpus> <group:(167|245)> <method:(Standard Mutation|Self Adaptive Mutation)>")

