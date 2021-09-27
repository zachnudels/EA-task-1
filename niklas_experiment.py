import sys
sys.path.insert(0, 'evoman')

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
                      speed="fastest",
                      logs="off")


os.environ["SDL_VIDEODRIVER"] = "dummy"
TIME_CONST = 0.001
runs_per_net = 1


def eval_genome(controller, net):
    env.player_controller = controller
    fitnesses = []
    for runs in range(runs_per_net):
        net.reset()
        fitness, p, e, t = env.play(pcont=controller)
        fitnesses.append(fitness)

    return min(fitnesses)


def eval_genome_fs_neat(genome, config):
    net = neat.ctrnn.CTRNN.create(genome, config, TIME_CONST)
    controller = RNNController(net, TIME_CONST)
    return eval_genome(controller, net)


def eval_genome_feat_eng(genome, config):
    net = neat.ctrnn.CTRNN.create(genome, config, TIME_CONST)
    controller = EngineeredRNNController(net, TIME_CONST)
    return eval_genome(controller, net)

def evaluate_winners(winners, method, enemy, plot_dir):
    local_dir = os.path.dirname('evoman')
    config_path = os.path.join(local_dir, f"{method}.cfg")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path
                         )

    mean_fitnesses = []
    for winner in winners:
        fitnesses = []
        for run in range(5):
            # play game and get fitness
            fitness = -1
            if method == "ENGINEERED":
                fitness = eval_genome_feat_eng(winner, config)
            elif method == "FS_NEAT":
                fitness = eval_genome_fs_neat(winner, config)
            fitnesses.append(fitness)

        mean_fitnesses.append(np.mean(fitnesses))

    # make boxplot

    plt.boxplot(mean_fitnesses)      
    plt.savefig(f'{plot_dir}/ind_gain__{enemy}_{method}.pdf') 
    plt.savefig(f'{plot_dir}/ind_gain__{enemy}_{method}.png') 



def run_final_experiment():
    generations = 100
    cpus = multiprocessing.cpu_count()
    runs = 10

    plot_dir = 'final_experiment_plots'
    if (not os.path.exists(f'./{plot_dir}/')):
        os.mkdir(f'./{plot_dir}/')
    

    if platform.system() == 'Darwin':
        multiprocessing.set_start_method('spawn')  # Comment this if not on MACOS


    # run first EA
    for method in ["FS_NEAT"]:
        # run for 3 individual enemies
        for enemy in [2]:
            # run 10 times with the same conditions and report average fitness per generation
            durations = []
            mean_fit_per_gen = []
            max_fit_per_gen = []
            winners = []
            for run in range(runs):

                # FOR TESTING SWITCH THE TWO LINES BELOW
                duration, means, maxes, winner = run_experiment(method, generations, cpus, enemy, run)
                # duration, means, maxes, winner = random.uniform(100, 500), np.random.uniform(low=10.0, high=60.0, size=(generations,)), np.random.uniform(low=60.0, high=100.0, size=(generations,)), random.uniform(0, 1)

                durations.append(durations)
                mean_fit_per_gen.append(means)
                max_fit_per_gen.append(maxes)
                winners.append(winner)

            avg_mean_fit_per_gen = np.mean(mean_fit_per_gen, axis=0)
            std_mean_fit_per_gen = np.std(mean_fit_per_gen, axis=0)
            avg_max_fit_per_gen = np.mean(max_fit_per_gen, axis=0)
            std_max_fit_per_gen = np.std(max_fit_per_gen, axis=0)

            # make a plot
            # plot average mean fitness per generation
            
            plt.plot(range(generations), avg_mean_fit_per_gen, '-', label='Average mean')
            plt.fill_between(range(generations), avg_mean_fit_per_gen-std_mean_fit_per_gen, avg_mean_fit_per_gen+std_mean_fit_per_gen, alpha = 0.5)

            plt.plot(range(generations), avg_max_fit_per_gen, '-', label='Average max')
            plt.fill_between(range(generations), avg_max_fit_per_gen-std_max_fit_per_gen, avg_max_fit_per_gen+std_max_fit_per_gen, alpha = 0.5)

            # plt.plot(std_mean_fit_per_gen, '-', label='Std mean')
            # plt.plot(std_max_fit_per_gen, '-', label='Std max')


            plt.xlabel('Fitness measures')
            plt.ylabel('Generations')
            plt.title(f'Enemy {enemy} against {method} EA')
            plt.legend()
            plt.savefig(f'./{plot_dir}/final_exp_plot_{enemy}_{method}.pdf')
            plt.savefig(f'./{plot_dir}/final_exp_plot_{enemy}_{method}.png')
            # plt.show()

            evaluate_winners(winners, method, enemy, plot_dir)


def run_experiment(method, generations, cpus, enemy, run):
    """
    Parameters:
        method:string ENGINEERED | FS_NEAT,
        generations:int number of generations to run,
        cpus:int number of cpus to use during run
        enemy:int enemy to run against
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
    # pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(5, 300, f"{method}-{run}-"))

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

    return end-start, means, maxes, winner


if __name__ == '__main__':
    # m = "ENGINEERED"
    # n = 4
    # c = 2
    # run_experiment(m, n, c, 8)
    run_final_experiment()


