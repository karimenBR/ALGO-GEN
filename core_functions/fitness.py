def compute_cost(individual, instance):
    return sum(abs(g - d) for g, d in zip(individual.chromosome, instance.data))


def fitness(individual, instance):
    individual.cost = compute_cost(individual, instance)
    individual.fitness = 1 / (1 + individual.cost)
    return individual.fitness
