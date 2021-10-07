import random
from copy import deepcopy

class Individual():

    def __init__(self, nr_weights = 10):
        self.weights = [0 for _ in range(nr_weights)]
        self.fitness = None
        self.is_child = False


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

        offspring = [weight_1 * p1 + weight_2 * p2 for p1, p2 in zip(parent_1.weights, parent_2.weights)]

        offsprings.append(offspring)

    return offsprings


def final_selection(population, pop_size = 10):
    return sorted(population, reverse=True, key=lambda x: x.fitness)[:pop_size]



if __name__ == '__main__':
    pop = [Individual() for x in range(20)]
    # print(selection(pop))
    print(final_selection(pop))