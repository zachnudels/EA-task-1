from copy import deepcopy
import numpy as np
from datetime import datetime, timedelta
from math import isclose
from pathlib import Path
from multiprocessing import Pool, cpu_count, set_start_method

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

experiment_name = 'multi_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

import platform

# initializes simulation in multi evolution mode, for multiple static enemies.
env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  logs="off"
                  )


class Individual:
    def __init__(self, weights=None):
        if isinstance(weights, list):
            self.weights = np.array(weights)
        elif isinstance(weights, int):
            self.weights = rng.uniform(-1, 1, weights)
        else:
            raise ValueError("Weights must either be list or int")

        self.child = False
        self.fitness = None

    def limit_weights(self, lower, upper):
        for i in range(len(self.weights)):
            if self.weights[i] > upper:
                self.weights[i] = upper
            elif self.weights[i] < lower:
                self.weights[i] = lower

    def __len__(self):
        return len(self.weights)

    def __iter__(self):
        yield from self.weights

    def __str__(self):
        return ",".join([str(x) for x in self.weights])

    def __le__(self, other):
        return self.fitness <= other.fitness

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __ge__(self, other):
        return self.fitness >= other.fitness


def selection(population):
    random_pop = rng.choice(population, len(population))
    return [(random_pop[i], random_pop[i + 1]) for i in range(0, len(random_pop) - 1, 2)]


def mutate(population, prop, generations, current_generation):
    for i in range(len(population)):
        if rng.uniform(0, 1) < prop:
            new_individual = deepcopy(population[i])
            # randomly alter weights
            mu = 0
            sigma = 0.5
            if not new_individual.child:
                sigma = (generations - current_generation) / generations

            for j in range(len(new_individual.weights)):
                new_individual.weights[j] += rng.normal(mu, sigma)
                new_individual.child = True
            # add to population
            new_individual.limit_weights(-1, 1)
            population.append(new_individual)
    return population


def self_adaptive_mutation(population, prop, sigmas, tau, tau_p):
    for i in range(len(population)):
        if rng.uniform(0, 1) < prop:
            new_individual = deepcopy(population[i])
            global_change = tau_p * rng.normal(0, 1)
            for j in range(len(new_individual.weights)):
                sigmas[j] = sigmas[j] * np.exp(global_change + (tau * rng.normal(0, 1)))
                if sigmas[j] <= 0:
                    sigmas[j] = 1e-5
                new_individual.weights[j] = new_individual.weights[j] + sigmas[j] * rng.normal(0, 1)
            new_individual.limit_weights(-1, 1)
            population.append(new_individual)
    return population, sigmas


def recombination(parent_list):  # mating
    offsprings = []
    for parent_1, parent_2 in parent_list:
        # take convex combination
        weight_1 = rng.uniform(0, 1)
        weight_2 = 1 - weight_1

        if len(parent_1) != len(parent_2):
            raise Exception("Parent length doesn't match")

        offspring_a = Individual([weight_1 * p1 + weight_2 * p2 for p1, p2 in zip(parent_1.weights, parent_2.weights)])
        offspring_a.child = True
        offsprings.append(offspring_a)

        offspring_b = Individual([weight_2 * p1 + weight_1 * p2 for p1, p2 in zip(parent_1.weights, parent_2.weights)])
        offspring_b.child = True
        offsprings.append(offspring_b)

    return offsprings


def final_selection(population, pop_size):
    return sorted(population, reverse=True, key=lambda x: x.fitness)[:pop_size]


def eval_ind(individuals, enemies, multi_fitness):
    env.enemies = enemies
    env.cons_multi = multi_fitness
    fitnesses = []
    for individual in individuals:
        if individual.fitness is None:
            f, p, e, t = env.play(pcont=individual.weights)
            individual.fitness = f
            fitnesses.append(f)
        else:
            fitnesses.append(individual.fitness)
    # print(f"Population subset: {len(individuals)}, Number of fitnesses: {len(fitnesses)}")
    return fitnesses


def evaluate(pool, population, num_workers, enemies, multi_fitness):
    jobs = []
    d = int(len(population) // num_workers)
    fitnesses = []
    for i in range(num_workers):
        start = i * d
        end = ((i + 1) * d)
        # print(f"Sending {start}:{end}. Population size is {len(population)}")
        if i == (num_workers - 1):
            jobs.append(pool.apply_async(eval_ind, (population[start:], enemies, multi_fitness)))
        else:
            jobs.append(pool.apply_async(eval_ind, (population[start:end], enemies, multi_fitness)))

    for job in jobs:
        fitnesses.extend(job.get(timeout=None))

#    print(len(fitnesses))
    for i in range(len(population)):
        population[i].fitness = fitnesses[i]

    return population


def resample_population(population, population_size, num_weights):
    population = sorted(population, reverse=True, key=lambda x: x.fitness)[:int(population_size // 2)]
    for _ in range(int(population_size // 2)):
        population.append(Individual(num_weights))
    return population


def evolve(population_size,
           num_generations,
           num_weights,
           mutate_prop,
           resample_gen,
           pool,
           enemies,
           multi_fitness,
           self_adaptive=False):
    print("Initializing Population")
    #  randomly initialise pop

    means = []
    maxes = []
    stagnant = False
    total_time = timedelta(0)

    sigmas = [rng.uniform(0, 1) for _ in range(num_weights)]
    tau = 1 / np.sqrt(2 * num_weights)
    tau_p = 1 / np.sqrt(2 * np.sqrt(num_weights))

    population = [Individual(num_weights) for _ in range(population_size)]
    print(f"\n\n==== Generation 0/{num_generations} =====\n")
    start = datetime.now()
    population = evaluate(pool, population, num_workers, enemies, multi_fitness)
    end = datetime.now()
    time = end - start
    total_time += time

    mean = np.mean([x.fitness for x in population])
    maxx = np.max([x.fitness for x in population])

    print(f"Mean fitness: {mean}")
    print(f"Max fitness: {maxx}")
    print(f"Duration: {time} (average: {time})")
    means.append(mean)
    maxes.append(maxx)

    for generation in range(num_generations):
        print(f"\n\n==== Generation {generation + 1}/{num_generations} =====\n")
        resampled = False
        start = datetime.now()
        if stagnant:
            resampled = True
            stagnant = False
            population = resample_population(population, population_size, num_weights)
            population = evaluate(pool, population, num_workers, enemies, multi_fitness)
        elif generation != 0 and generation % resample_gen == 0:
            resampled = True
            population = resample_population(population, population_size, num_weights)
            population = evaluate(pool, population, num_workers, enemies, multi_fitness)

        parents = selection(population)
        offspring = recombination(parents)
        population.extend(offspring)
        if self_adaptive:
            population, sigmas = self_adaptive_mutation(population, mutate_prop, sigmas, tau, tau_p)
        else:
            population = mutate(population, mutate_prop, num_generations, generation)
        population = evaluate(pool, population, num_workers, enemies, multi_fitness)
        population = final_selection(population, population_size)

        mean = np.mean([x.fitness for x in population])
        maxx = np.max([x.fitness for x in population])
        if isclose(mean, maxx, rel_tol=1e-05):
            stagnant = True

        end = datetime.now()
        time = end - start
        total_time += time

        if resampled:
            print("Reset half of population")
        if stagnant:
            print("Population stagnant. Resetting in next generation.")
        print(f"Mean fitness: {mean}")
        print(f"Max fitness: {maxx}")
        print(f"Duration: {time} (average: {total_time / (generation + 2)})")
        means.append(mean)
        maxes.append(maxx)

    print(f"Total duration was {total_time}")

    return population[np.argmax(population)], means, maxes


def multi_fitness(values):
    return np.mean(values) / 2 + np.min(values)


def save_results(means, maxes, best_genome, means_path, maxes_path, best_path):
    dt = datetime.now().timestamp()
    np.savetxt(f'{means_path}/{dt}.csv', np.array(means), delimiter=',')
    np.savetxt(f'{maxes_path}/{dt}.csv', np.array(maxes), delimiter=',')
    np.savetxt(f'{best_path}/{dt}.csv', np.array(best_genome.weights), delimiter=',')


if __name__ == '__main__':

    if platform.system() == 'Darwin':
        set_start_method('spawn')  # Comment this if not on MACOS

    _enemies = [1, 6, 7]

    enemy_string = "".join([str(x) for x in _enemies])
    path = Path(f"handmade_results/{enemy_string}/")
    means_path = path.joinpath(f"means/")
    maxes_path = path.joinpath(f"maxes/")
    best_path = path.joinpath(f"best/")
    if not means_path.exists():
        means_path.mkdir(parents=True, exist_ok=True)

    if not maxes_path.exists():
        maxes_path.mkdir(parents=True, exist_ok=True)

    if not best_path.exists():
        best_path.mkdir(parents=True, exist_ok=True)


    # number of weights for multilayer with 10 hidden neurons.
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    num_workers = cpu_count()
    pool = Pool(num_workers)
    _best, _means, _maxes = evolve(population_size=50,
                                   num_generations=50,
                                   num_weights=n_vars,
                                   mutate_prop=0.5,
                                   resample_gen=20,
                                   pool=pool,
                                   enemies=_enemies,
                                   multi_fitness=multi_fitness,
                                   self_adaptive=False
                                   )
    save_results(_means, _maxes, _best, means_path, maxes_path, best_path)
