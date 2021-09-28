import sys

sys.path.insert(0, 'evoman')
from pathlib import Path
from environment import Environment
import neat, multiprocessing, os, pickle
from rnn_controller import RNNController
from rnn_en_controller import EngineeredRNNController
from datetime import datetime
import platform

import numpy as np
from matplotlib import pyplot as plt
import random

experiment_name = 'individual_demo'
env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  enemymode="static",
                  level=2,
                  randomini="yes",
                  speed="fastest",
                  logs="off")

os.environ["SDL_VIDEODRIVER"] = "dummy"
TIME_CONST = 0.001
runs_per_net = 1


def eval_genome(controller, net, r_all):
    env.player_controller = controller
    fitnesses = []
    player = []
    enemy = []
    for runs in range(runs_per_net):
        net.reset()
        fitness, p, e, t = env.play(pcont=controller)
        fitnesses.append(fitness)
        player.append(p)
        enemy.append(e)
    i = fitnesses.index(min(fitnesses))
    if r_all:
        return fitnesses[i], player[i], enemy[i]
    else:
        return min(fitnesses)


def eval_genome_fs_neat(genome, config, r_all=False):
    net = neat.ctrnn.CTRNN.create(genome, config, TIME_CONST)
    controller = RNNController(net, TIME_CONST)
    return eval_genome(controller, net, r_all)


def eval_genome_feat_eng(genome, config, r_all=False):
    net = neat.ctrnn.CTRNN.create(genome, config, TIME_CONST)
    controller = EngineeredRNNController(net, TIME_CONST)
    return eval_genome(controller, net, r_all)


def evaluate_winners(winners, method, plot_dir):
    local_dir = os.path.dirname('evoman')
    config_path = os.path.join(local_dir, f"{method}.cfg")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path
                         )

    all_gains = []
    for winner in winners:
        print(f"Fitness = {winner.fitness}")
        gains = []
        for _ in range(5):
            p = 0
            e = 0
            if method == "ENGINEERED":
                _, p, e = eval_genome_feat_eng(winner, config, True)
            elif method == "FS_NEAT":
                _, p, e = eval_genome_fs_neat(winner, config, True)
            gains.append(p - e)
        all_gains.append(np.mean(gains))

    # make boxplot
    np.savetxt(f'{plot_dir}/gains.csv', all_gains, delimiter=',')
    plt.boxplot(all_gains)
    plt.savefig(f'{plot_dir}/ind_gain.pdf')
    plt.savefig(f'{plot_dir}/ind_gain.png')


def run_final_experiment(methods, enemies):
    generations = 100
    cpus = multiprocessing.cpu_count()
    runs = 10

    plot_dir = 'final_experiment_plots'
    plot_path = Path(plot_dir)
    if not plot_path.exists():
        plot_path.mkdir(parents=True, exist_ok=True)

    # run first EA
    for method in [methods]:
        # run for 3 individual enemies
        for enemy in [enemies]:
            path = Path(f"checkpoints/{enemy}/{method}/")
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

            spec_plot_dir = f'./{plot_dir}/{enemy}/{method}/'
            spec_plot_path = Path(spec_plot_dir)
            if not spec_plot_path.exists():
                spec_plot_path.mkdir(parents=True, exist_ok=True)

            # run 10 times with the same conditions and report average fitness per generation
            # durations = []
            mean_fit_per_gen = []
            max_fit_per_gen = []
            winners = []
            best_sizes = []
            for run in range(runs):
                # FOR TESTING SWITCH THE TWO LINES BELOW
                duration, means, maxes, winner, best_size = run_experiment(method, generations, cpus, enemy, run, path)
                # duration, means, maxes, winner = random.uniform(100, 500), np.random.uniform(low=10.0, high=60.0, size=(generations,)), np.random.uniform(low=60.0, high=100.0, size=(generations,)), random.uniform(0, 1)

                mean_fit_per_gen.append(means)
                max_fit_per_gen.append(maxes)
                winners.append(winner)
                best_sizes.append(best_size)

            avg_mean_fit_per_gen = np.mean(mean_fit_per_gen, axis=0)
            std_mean_fit_per_gen = np.std(mean_fit_per_gen, axis=0)
            avg_max_fit_per_gen = np.mean(max_fit_per_gen, axis=0)
            std_max_fit_per_gen = np.std(max_fit_per_gen, axis=0)


            # make a plot
            # plot average mean fitness per generation

            plt.plot(range(generations), avg_mean_fit_per_gen, '-', label='Average mean')
            plt.fill_between(range(generations), avg_mean_fit_per_gen - std_mean_fit_per_gen,
                             avg_mean_fit_per_gen + std_mean_fit_per_gen, alpha=0.5)

            plt.plot(range(generations), avg_max_fit_per_gen, '-', label='Average max')
            plt.fill_between(range(generations), avg_max_fit_per_gen - std_max_fit_per_gen,
                             avg_max_fit_per_gen + std_max_fit_per_gen, alpha=0.5)

            plt.xlabel('Generations')
            plt.ylabel('Fitness measures')
            plt.title(f'Enemy {enemy} against {method} EA')
            plt.legend()
            plt.savefig(f'{spec_plot_dir}/final_exp_plot.pdf')
            plt.savefig(f'{spec_plot_dir}/final_exp_plot.png')
            plt.close()
            # plt.show()

            np.savetxt(f'{spec_plot_dir}/10run_avg_mean_fitnesses.csv', avg_mean_fit_per_gen, delimiter=',')
            np.savetxt(f'{spec_plot_dir}/10run_avg_max_fitnesses.csv', avg_max_fit_per_gen, delimiter=',')
            np.savetxt(f'{spec_plot_dir}/10run_std_mean_fitnesses.csv', std_mean_fit_per_gen, delimiter=',')
            np.savetxt(f'{spec_plot_dir}/10run_std_max_fitnesses.csv', std_max_fit_per_gen, delimiter=',')
            np.savetxt(f'{spec_plot_dir}/10run_best_genome_size.csv', best_sizes, delimiter=',')

            evaluate_winners(winners, method, spec_plot_dir)


def run_experiment(method, generations, cpus, enemy, run, path):
    """
    Parameters:
        method:string ENGINEERED | FS_NEAT,
        generations:int number of generations to run,
        cpus:int number of cpus to use during run
        enemy:int enemy to run against
        run:int iteration
        path:Path working dir
    Returns:
        duration:datetime time of experiment,
        means:array[float] mean fitness of each generation,
        maxes:array[float] max fitness of each generation,
        winner:genome the winning genotype
    """

    local_dir = os.path.dirname('evoman')

    config_path = os.path.join(local_dir, f"{method}.cfg")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path
                         )

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    run_path = path.joinpath(Path(f"{run}/"))
    if not run_path.exists():
        run_path.mkdir(parents=True, exist_ok=True)

    pop.add_reporter(neat.Checkpointer(5, 300, str(run_path) + "/"))

    env.enemies = [enemy]
    pe = None

    if method == "ENGINEERED":
        pe = neat.ParallelEvaluator(cpus, eval_genome_feat_eng)
    elif method == "FS_NEAT":
        pe = neat.ParallelEvaluator(cpus, eval_genome_fs_neat)

    start = datetime.now()
    winner = pop.run(pe.evaluate, n=generations)
    end = datetime.now()
    means = stats.get_fitness_mean()
    maxes = stats.get_fitness_stat(max)
    best_genome = max(stats.best_genomes(len(stats.get_species_sizes())), key=lambda x: x.fitness)
    best_size = best_genome.size()

    return end - start, means, maxes, best_genome, best_size


if __name__ == '__main__':
    method = sys.argv[1]
    enemy = sys.argv[2]
    if platform.system() == 'Darwin':
        multiprocessing.set_start_method('spawn')  # Comment this if not on MACOS

    run_final_experiment(method, enemy)


