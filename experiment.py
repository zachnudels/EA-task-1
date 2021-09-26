import sys
sys.path.insert(0, 'evoman')

from environment import Environment
import neat, multiprocessing, os, pickle
from rnn_controller import RNNController
from rnn_en_controller import EngineeredRNNController
from datetime import datetime
import platform

import numpy as np

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


def run_final_experiment():
    generations = 100
    cpus = multiprocessing.cpu_count()
    runs = 10

    if platform.system() == 'Darwin':
        multiprocessing.set_start_method('spawn')  # Comment this if not on MACOS


    # run first EA
    for method in ["ENGINEERED", "FS_NEAT"]:
        # run for 3 individual enemies
        for enemy in [1, 3, 5]:
            # run 10 times with the same conditions and report average fitness per generation
            durations = []
            mean_fitness_per_generation = []
            max_fitness_per_generation = []
            best_solutions = []
            for run in range(runs):
                duration, means, maxes, winner = run_experiment(method, generations, cpus, enemy)
                durations.append(durations)
                mean_fitness_per_generation.append(means)
                max_fitness_per_generation.append(maxes)
                best_solutions.append(winner)

            # final_best_solution = np.argmax(best_solutions) # TODO this is only the logic - adjust to how the best solution is actually stored

            avg_mean_fitness_per_generation = np.mean(mean_fitness_per_generation)
            std_mean_fitness_per_generation = np.std(mean_fitness_per_generation)
            avg_max_fitness_per_generation = np.mean(max_fitness_per_generation)
            std_max_fitness_per_generation = np.std(max_fitness_per_generation)

        # make a plot


def run_experiment(method, generations, cpus, enemy):
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
    pop.add_reporter(neat.StdOutReporter(True))

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


