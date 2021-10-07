import random
from copy import deepcopy
import numpy as np
from datetime import datetime
from math import isclose

from engineered_controller import EngineeredController

rng = np.random.default_rng(int(datetime.now().timestamp()))

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

    def __len__(self):
        return len(self.weights)

    def __iter__(self):
        yield from self.weights

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
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
    random_pop = random.sample(population, len(population))
    return [(random_pop[i], random_pop[i+1]) for i in range(0, len(random_pop)-1, 2)]


def mutate(parents, offspring, new_inds, generations, current_generation):
    population = parents
    for child in offspring:
        population.append(child)

    for individual in population:
        if random.random() < (new_inds/len(population)):
            new_individual = deepcopy(individual)
            # randomly alter weights
            for weight in individual:
                mu = 0
                if new_individual.child:
                    sigma = 0.5
                else:
                    sigma = (generations - current_generation) / generations
                weight + random.gauss(mu, sigma)
                new_individual.child = True
            # add to population
            population.append(new_individual)

    return population


def recombination(parent_list): # mating
    offsprings = []
    for parent_1, parent_2 in parent_list:
        # take convex combination
        weight_1 = random.uniform(0, 1)
        weight_2 = 1 - weight_1

        if len(parent_1) != len(parent_2):
            raise Exception("Parent length doesn't match")

        offspring = Individual([weight_1 * p1 + weight_2 * p2 for p1, p2 in zip(parent_1.weights, parent_2.weights)])
        offspring.child = True

        offsprings.append(offspring)

    parents = []
    for pair in parent_list:
        parents.append(pair[0])
        parents.append(pair[1])

    return parents, offsprings


def final_selection(population, pop_size = 10):
    return sorted(population, reverse=True, key=lambda x: x.fitness)[:pop_size]


def evaluate(population, environment):
    # This assumes that an individual will never have its weights changed
    # TODO CHECK THIS ASSUMPTION!
    for individual in population:
        if individual.fitness is None:
            f, p, e, t = environment.play(pcont=individual.weights)
            individual.fitness = f
    return population


def resample_population(population, population_size, num_weights):
    population = sorted(population, reverse=True, key=lambda x: x.fitness)[:int(population_size // 2)]
    for _ in range(int(population_size // 2)):
        population.append(Individual(num_weights))
    return population


def evolve(population_size, num_generations, num_weights, mutate_new_inds, environment, resample_gen):
    print("Initializing Population")
    #  randomly initialise pop
    population = [Individual(num_weights) for _ in range(population_size)]
    population = evaluate(population, environment)
    means = []
    maxes = []
    stagnant = False

    for generation in range(num_generations):
        print(f"\n\n==== Generation {generation} =====\n")
        resampled = False
        start = datetime.now()
        if stagnant:
            resampled = True
            stagnant = False
            population = resample_population(population, population_size, num_weights)
            population = evaluate(population, environment)
        elif generation != 0 and generation % resample_gen == 0:
            resampled = True
            population = resample_population(population, population_size, num_weights)
            population = evaluate(population, environment)

        parents = selection(population)
        parents, offspring = recombination(parents)
        population = mutate(parents, offspring, mutate_new_inds, num_generations, generation)
        population = evaluate(population, environment)
        population = final_selection(population, population_size)

        mean = np.mean([x.fitness for x in population])
        max = np.max([x.fitness for x in population])
        if isclose(mean, max):
            stagnant = True

        end = datetime.now()

        if resampled:
            print("Reset half of population")
        if stagnant:
            print("Population stagnant. Resetting in next generation.")
        print(f"Mean fitness: {mean}")
        print(f"Max fitness: {max}")
        print(f"Duration: {end-start}")
        means.append(mean)
        maxes.append(max)

    return population[np.argmax(population)], means, maxes


if __name__ == '__main__':
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

    # initializes simulation in multi evolution mode, for multiple static enemies.
    env = Environment(experiment_name=experiment_name,
                      enemies=[1, 4, 6, 7],
                      multiplemode="yes",
                      playermode="ai",
                      player_controller=EngineeredController(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      logs="off"
                      )

    # number of weights for multilayer with 10 hidden neurons.
    n_vars = (9 + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    best, means, maxes = evolve(population_size=10,
                                num_generations=150,
                                num_weights=n_vars,
                                mutate_new_inds=5,
                                environment=env,
                                resample_gen=20
                                )
    # print(best)
    print(means)
    print(maxes)
