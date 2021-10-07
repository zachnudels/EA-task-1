import random


def selection(population):
    random_pop = random.sample(population, len(population))
    return [(random_pop[i], random_pop[i+1]) for i in range(0, len(random_pop)-1, 2)]


if __name__ == '__main__':
    pop = [1,2,3,4,5,6, 7]
    print(selection(pop))