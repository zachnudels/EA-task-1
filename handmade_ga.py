import random
from copy import deepcopy


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





if __name__ == '__main__':
    pop = [1,2,3,4,5,6, 7]
    print(selection(pop))