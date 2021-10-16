from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import os
import sys


def read_data(group):
    mean_fit_per_gen = []
    max_fit_per_gen = []
    dir_path = Path(f"handmade_results/{group}/")
    for root, dirs, files in os.walk(dir_path.joinpath('means')):
        mean_fit_per_gen = [np.loadtxt(os.path.join(root, f), delimiter=',') for f in files if 'csv' in f]
    for root, dirs, files in os.walk(dir_path.joinpath('maxes')):
        max_fit_per_gen = [np.loadtxt(os.path.join(root, f), delimiter=',') for f in files if 'csv' in f]
    return np.array(mean_fit_per_gen), np.array(max_fit_per_gen)


def plot_graphs(group, method):
    mean_fit_per_gen, max_fit_per_gen = read_data(group)
    generations = mean_fit_per_gen.shape[1]

    avg_mean_fit_per_gen = np.mean(mean_fit_per_gen, axis=0)
    std_mean_fit_per_gen = np.std(mean_fit_per_gen, axis=0)
    avg_max_fit_per_gen = np.mean(max_fit_per_gen, axis=0)
    std_max_fit_per_gen = np.std(max_fit_per_gen, axis=0)

    plt.plot(range(generations), avg_mean_fit_per_gen, '-', label='Average mean')
    plt.fill_between(range(generations), avg_mean_fit_per_gen - std_mean_fit_per_gen,
                     avg_mean_fit_per_gen + std_mean_fit_per_gen, alpha=0.5)

    plt.plot(range(generations), avg_max_fit_per_gen, '-', label='Average max')
    plt.fill_between(range(generations), avg_max_fit_per_gen - std_max_fit_per_gen,
                     avg_max_fit_per_gen + std_max_fit_per_gen, alpha=0.5)

    dir_path = Path(f"handmade_results/plots/{method}/{group}")
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)

    plt.xlabel('Generations')
    plt.ylabel('Fitness measures')
    plt.title(f'Enemy group {group} with {method} EA')
    plt.legend()
    plt.savefig(f'{dir_path}/final_exp_plot.pdf')
    plt.savefig(f'{dir_path}/final_exp_plot.png')
    plt.close()


if __name__ == '__main__':
    try:
        group = sys.argv[1]
        method = " ".join(sys.argv[2:])
        plot_graphs(group, method)
    except Exception:
        raise ValueError("Requires args <group:(167|245)> <method:(Standard Mutation|Self Adaptive Mutation)>")
