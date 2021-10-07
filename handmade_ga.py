import random
from copy import deepcopy


class Individual:
    def __init__(self, weights=None, num_weights=None):
        if weights is None:
            if num_weights is None:
                raise ValueError("If no weights specified, must specify number of random weights to initialize")
            self.weights = [random.random() for _ in range(num_weights)]
        else:
            self.weights = weights

        self.child = False

    def __len__(self):
        return len(self.weights)

    def __iter__(self):
        yield from self.weights




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

        offspring = Individual([weight_1 * p1 + weight_2 * p2 for p1, p2 in zip(parent_1.weights, parent_2.weights)])
        offspring.child = True

        offsprings.append(offspring)

    parents = []
    for pair in parent_list:
        parents.append(pair[0])
        parents.append(pair[1])

    return parents, offsprings


def evolve(population_size, num_generations, num_weights, mutate_new_inds):
    # randomly initialise pop
    population = [Individual(num_weights=num_weights) for _ in range(population_size)]

    for generation in range(num_generations):
        parents = selection(population)
        parents, offspring = recombination(parents)
        population = mutate(parents, offspring, mutate_new_inds, num_generations, generation)

def final_selection(population, pop_size = 10):
    return sorted(population, reverse=True, key=lambda x: x.fitness)[:pop_size]



if __name__ == '__main__':
    evolve(10, 1, 10, 5)
