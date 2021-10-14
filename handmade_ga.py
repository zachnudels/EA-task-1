from copy import deepcopy
import numpy as np
from datetime import datetime, timedelta
from math import isclose
from pathlib import Path

from demo_controller import player_controller

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

    def __repr__(self):
        return self.weights

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
    return [(random_pop[i], random_pop[i+1]) for i in range(0, len(random_pop)-1, 2)]


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


def evolve(population_size, num_generations, num_weights, mutate_prop, environment, resample_gen, self_adaptive=False):
    print("Initializing Population")
    #  randomly initialise pop

    means = []
    maxes = []
    stagnant = False
    total_time = timedelta(0)

    sigmas = [rng.uniform(0, 1) for _ in range(num_weights)]
    tau = 1/np.sqrt(2*num_weights)
    tau_p = 1/np.sqrt(2*np.sqrt(num_weights))

    population = [Individual(num_weights) for _ in range(population_size)]
    print(f"\n\n==== Generation 0/{num_generations} =====\n")
    start = datetime.now()
    population = evaluate(population, environment)
    end = datetime.now()
    time = end - start
    total_time += time

    mean = np.mean([x.fitness for x in population])
    max = np.max([x.fitness for x in population])

    print(f"Mean fitness: {mean}")
    print(f"Max fitness: {max}")
    print(f"Duration: {time} (average: {time})")
    means.append(mean)
    maxes.append(max)

    for generation in range(num_generations):
        print(f"\n\n==== Generation {generation+1}/{num_generations} =====\n")
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
        offspring = recombination(parents)
        population.extend(offspring)
        if self_adaptive:
            population, sigmas = self_adaptive_mutation(population, mutate_prop, sigmas, tau, tau_p)
        else:
            population = mutate(population, mutate_prop, num_generations, generation)
        population = evaluate(population, environment)
        population = final_selection(population, population_size)

        mean = np.mean([x.fitness for x in population])
        max = np.max([x.fitness for x in population])
        if isclose(mean, max, rel_tol=1e-05):
            stagnant = True

        end = datetime.now()
        time = end - start
        total_time += time

        if resampled:
            print("Reset half of population")
        if stagnant:
            print("Population stagnant. Resetting in next generation.")
        print(f"Mean fitness: {mean}")
        print(f"Max fitness: {max}")
        print(f"Duration: {time} (average: {total_time/(generation+2)})")
        means.append(mean)
        maxes.append(max)

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
    enemies = [2, 4, 5]

    enemy_string = "".join([str(x) for x in enemies])
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
                      enemies=enemies,
                      multiplemode="yes",
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      logs="off"
                      )

    env.cons_multi = multi_fitness

    # number of weights for multilayer with 10 hidden neurons.
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    best, means, maxes = evolve(population_size=50,
                                num_generations=100,
                                num_weights=n_vars,
                                mutate_prop=0.5,
                                environment=env,
                                resample_gen=20,
                                self_adaptive=True
                                )
    save_results(means, maxes, best, means_path, maxes_path, best_path)
