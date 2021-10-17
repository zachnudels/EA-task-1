from copy import deepcopy
import numpy as np
from datetime import datetime, timedelta
from math import isclose
from pathlib import Path
from multiprocessing import Pool, cpu_count, set_start_method
from matplotlib import pyplot as plt
import pandas as pd

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
    dir_path = Path(f"handmade_results/{method}/{group}/best")
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

    max_gain = np.max(all_gains)
    best_ind = all_weights[np.argmax(all_gains)]

    plot_and_save(all_gains, group, method)

    return max_gain, best_ind


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


def evaluate_best(weights):
    # weights = np.loadtxt(path, delimiter=",")
    ppts = []
    epts = []
    for enemy in [1, 2, 3, 4, 5, 6, 7, 8]:
        print(f"Running Enemy {enemy}")
        ps = []
        es = []
        for _ in range(5):
            env.enemies = [enemy]
            f, p, e, t = env.play(pcont=weights)
            ps.append(p)
            es.append(e)
        ppts.append(np.mean(ps))
        epts.append(np.mean(es))

    path = Path(f"handmade_results/")
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({'Enemy Points': epts, 'Player Points': ppts})
    df.to_csv(path.joinpath("final_table.csv"))
    np.savetxt(path.joinpath("best.txt"), weights)


def plot_and_save(gains, group, method):
    path = Path(f"handmade_results/{method}/{group}/")
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    np.savetxt(f'{path}/gains.csv', gains, delimiter=',')
    plt.boxplot(gains)
    plt.title(f'Group {group} with {method}')
    plt.xticks([])
    plt.ylabel("Individual Gain")

    plt.savefig(path.joinpath('ind_gain.pdf'))
    plt.savefig(path.joinpath('ind_gain.png'))


if __name__ == '__main__':

    # df = evaluate_best('/Users/Zach/Google Drive/Studies/VU/6. Period 1 /Evolutionary Computing/Assignments/Assignment 1/evoman_framework/handmade_results/Standard Mutation/167/best/1634345001.639455.csv')
    # print(df)
    groups = ["167", "245"]
    methods = ["Standard Mutation", "Self-Adaptive Mutation"]
    if platform.system() == 'Darwin':
        set_start_method('spawn')  # Comment this if not on MACOS

    try:
        cpus = cpu_count() - int(sys.argv[1])
        # group = sys.argv[2]
        # method = " ".join(sys.argv[3:])
        pool = Pool(cpus)
        # evaluate(group, method, cpus, pool)
        best_ind = None
        current_gain = -10000
        for group in groups:
            for method in methods:
                print(f"=== Running group {group} for {method} ===")
                gain, ind = evaluate(group, method, cpus, pool)
                if gain > current_gain:
                    current_gain = gain
                    best_ind = ind
                print(f"Best gain found = {gain}")
                print()
        evaluate_best(best_ind)

    except Exception:
        raise ValueError("Requires args <less cpus> <group:(167|245)> <method:(Standard Mutation|Self Adaptive Mutation)>")

