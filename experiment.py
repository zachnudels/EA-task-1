import sys
sys.path.insert(0, 'evoman')

from environment import Environment
import neat, multiprocessing, os, pickle
from rnn_controller import RNNController
from rnn_en_controller import EngineeredRNNController
from datetime import datetime

import numpy as np

experiment_name = 'individual_demo'
env = Environment(experiment_name=experiment_name,
                      playermode="ai",
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      logs="off")


os.environ["SDL_VIDEODRIVER"] = "dummy"


def eval_genome_fs_neat(genome, config):
    TIME_CONST = 0.001
    runs_per_net = 1
    net = neat.ctrnn.CTRNN.create(genome, config, TIME_CONST)
    controller = RNNController(net, TIME_CONST)
    env.player_controller = controller

    fitnesses = []
    for runs in range(runs_per_net):
        net.reset()
        fitness = 0
        f,p,e,t = env.play(pcont=controller)
        fitnesses.append(f)


    return min(fitnesses)

def eval_genome_feat_eng(genome, config):
    TIME_CONST = 0.001
    runs_per_net = 1
    net = neat.ctrnn.CTRNN.create(genome, config, TIME_CONST)
    controller = EngineeredRNNController(net, TIME_CONST)
    env.player_controller = controller

    fitnesses = []
    for runs in range(runs_per_net):
        net.reset()
        fitness = 0
        f,p,e,t = env.play(pcont=controller)
        fitnesses.append(f)

    return min(fitnesses)

def run_final_experiment():
    ## RUN FINAL EXPERIMENT

    # run first EA

    # run for 3 individual enemies

    for enemy in [1,3,5]:
        # run 10 times with the same conditions and report average fitness per generation

        mean_fitness_per_generation = []
        max_fitness_per_generation = []
        best_solutions = []
        for run in range(10):
            # run fitting
            pass

            # for each generation save the maximum fitness and the mean fitness of the population

            # save best solution

        final_best_solution = np.argmax(best_solutions) # TODO this is only the logic - adjust to how the best solution is actually stored

        avg_mean_fitness_per_generation = np.mean(mean_fitness_per_generation)
        std_mean_fitness_per_generation = np.std(mean_fitness_per_generation)
        avg_max_fitness_per_generation = np.mean(max_fitness_per_generation)
        std_max_fitness_per_generation = np.std(max_fitness_per_generation)

    # make a plot 



if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # Comment this if not on MACOS

    """
    PARAMTER CHOICES
    """
    # Method
    # method = "FS_NEAT
    method = "ENGINEERED"

    # Generations
    n = 50

    # Config
    config_path = 'engineered.cfg'

    # Number of CPUs
    # cpus = multiprocessing.cpu_count()
    cpus = 2



    local_dir = os.path.dirname('evoman')
    config_path = os.path.join(local_dir, config_path)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)


    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    if method == "ENGINEERED":
        pe = neat.ParallelEvaluator(cpus, eval_genome_feat_eng)
    elif method == "FS_NEAT":
        pe = neat.ParallelEvaluator(cpus, eval_genome_fs_neat)

    start = datetime.now()
    winner = pop.run(pe.evaluate, n=n)
    end = datetime.now()

    print(f"Duration: {end-start}")
