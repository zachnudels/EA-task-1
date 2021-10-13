import sys

sys.path.insert(0, 'evoman')
from pathlib import Path
from evoman.environment import Environment
import neat, multiprocessing, os, pickle
from neat import ParallelEvaluator
from rnn_controller import RNNController
from rnn_en_controller import EngineeredRNNController
from datetime import datetime
import platform

import numpy as np
from matplotlib import pyplot as plt
import random

from multiprocessing import Pool


def multi_fitness(values):
    return np.mean(values) / 2 + min(values)

TIME_CONST = 0.001
runs_per_net = 1

os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'individual_demo'
env = Environment(experiment_name=experiment_name,
                    playermode="ai",
                    enemymode="static",
                    level=2,
                    randomini="no",
                    speed="fastest",
                    multiplemode="yes",
                    logs="off")

class CustomParallelEvaluator(ParallelEvaluator):
    def __init__(self, num_workers, eval_function, timeout=None, enemies=None):
        super().__init__(num_workers, eval_function, timeout)
        if enemies is None:
            enemies = list([1])
        self.enemies = enemies

    def evaluate(self, genomes, config):
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config, self.enemies)))

        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in list(zip(jobs, genomes)):
            genome.fitness = job.get(timeout=self.timeout)



def eval_genome(controller, net, r_all, enemies):
    env.player_controller = controller
    env.enemies = enemies
    env.cons_multi = multi_fitness
    fitnesses = []
    player = []
    enemy = []
    # print(f"ENEMIES: {env.enemies}")
    for runs in range(runs_per_net):
        net.reset()
        # print("START PLAY")
        fitness, p, e, t = env.play(pcont=controller)
        # print("FINISHED PLAY")
        fitnesses.append(fitness)
        player.append(p)
        enemy.append(e)
    i = fitnesses.index(min(fitnesses))
    if r_all:
        return fitnesses[i], player[i], enemy[i]
    else:
        return min(fitnesses)


def eval_genome_fs_neat(genome, config, enemies, r_all=False):
    net = neat.ctrnn.CTRNN.create(genome, config, TIME_CONST)
    controller = RNNController(net, TIME_CONST)
    return eval_genome(controller, net, r_all, enemies)


def eval_genome_feat_eng(genome, config, enemies, r_all=False):
    net = neat.ctrnn.CTRNN.create(genome, config, TIME_CONST)
    controller = EngineeredRNNController(net, TIME_CONST)
    return eval_genome(controller, net, r_all, enemies)


def evaluate_winners(winners, method, plot_dir):
    local_dir = os.path.dirname('evoman')
    config_path = os.path.join(local_dir, f"{method}_best.cfg")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path
                         )

    all_gains = []
    for winner in winners:
        gains = []
        for _ in range(5):
            p = 0
            e = 0
            if method == "ENGINEERED":
                _, p, e = eval_genome_feat_eng(winner, config, True, enemies=[1,2,3,4,5,6,7,8])
            elif method == "FS_NEAT":
                _, p, e = eval_genome_fs_neat(winner, config, True, enemies=[1,2,3,4,5,6,7,8])
            gains.append(p - e)
        all_gains.append(np.mean(gains))

    # make boxplot
    np.savetxt(f'{plot_dir}/gains.csv', all_gains, delimiter=',')
    plt.boxplot(all_gains)
    plt.savefig(f'{plot_dir}/ind_gain.pdf')
    plt.savefig(f'{plot_dir}/ind_gain.png')


def run_final_experiment(methods, groups):
    generations = 100
    cpus = multiprocessing.cpu_count() - 1
    runs = 10

    plot_dir = 'results_generalist'
    plot_path = Path(plot_dir)
    if not plot_path.exists():
        plot_path.mkdir(parents=True, exist_ok=True)

    # run first EA
    for method in [methods]:
        # run for 3 individual enemies
        for group in groups:
            # path = Path(f"checkpoints/{group}/{method}/")
            # if not path.exists():
            #     path.mkdir(parents=True, exist_ok=True)

            spec_plot_dir = f'./{plot_dir}/{group}/{method}/'
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
                run_path = spec_plot_path.joinpath(Path(f"{run}/"))
                if not run_path.exists():
                    run_path.mkdir(parents=True, exist_ok=True)

                # FOR TESTING SWITCH THE TWO LINES BELOW
                duration, means, maxes, winner, best_size = run_experiment(method=method, cpus=cpus, generations=generations, group=group, run_path=run_path)
                # duration, means, maxes, winner = random.uniform(100, 500), np.random.uniform(low=10.0, high=60.0, size=(generations,)), np.random.uniform(low=60.0, high=100.0, size=(generations,)), random.uniform(0, 1)

                mean_fit_per_gen.append(means)
                max_fit_per_gen.append(maxes)
                winners.append(winner)
                best_sizes.append(best_size)

                np.savetxt(f'{run_path}mean_fit_per_gen.csv', mean_fit_per_gen, delimiter=',')
                np.savetxt(f'{run_path}max_fit_per_gen.csv', max_fit_per_gen, delimiter=',')
                np.savetxt(f'{run_path}winners.csv', winners, delimiter=',')
                np.savetxt(f'{run_path}best_sizes.csv', best_sizes, delimiter=',')



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
            plt.title(f'group {group} against {method} EA')
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


def run_experiment(method, cpus, generations, run_path, group=None, config=None):
    """
    Parameters:
        method:string ENGINEERED | FS_NEAT,
        generations:int number of generations to run,
        cpus:int number of cpus to use during run
        group:List[int] enemies to run against
        run:int iteration
        path:Path working dir
    Returns:
        duration:datetime time of experiment,
        means:array[float] mean fitness of each generation,
        maxes:array[float] max fitness of each generation,
        winner:genome the winning genotype
    """

    local_dir = os.path.dirname('evoman')

    if cpus is None:
        cpus = multiprocessing.cpu_count()

    if config is None:  # use default
        config_path = os.path.join(local_dir, f"{method}_best.cfg")
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path
                             )

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    # pop.add_reporter(neat.Checkpointer(5, 300, str(run_path) + "/"))

    # env.update_parameter('enemies',group)
    pe = None

    if method == "ENGINEERED":
        pe = CustomParallelEvaluator(cpus, eval_genome_feat_eng, enemies=group)
    elif method == "FS_NEAT":
        pe = CustomParallelEvaluator(cpus, eval_genome_fs_neat, enemies=group)

    start = datetime.now()
    end = datetime.now()
    winner = pop.run(pe.evaluate, n=generations)
    means = stats.get_fitness_mean()
    maxes = stats.get_fitness_stat(max)
    best_genome = max(stats.best_genomes(len(stats.get_species_sizes())), key=lambda x: x.fitness)

    return end - start, means, maxes, best_genome, best_genome.size(), winner


if __name__ == '__main__':
    method = sys.argv[1]
    group_1 = [int(a) for a in sys.argv[2]]
    # group_2 = [int(a) for a in sys.argv[3]]

    if platform.system() == 'Darwin':
        multiprocessing.set_start_method('spawn')  # Comment this if not on MACOS

    run_final_experiment(method, [group_1]) #,group_2


