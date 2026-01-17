import random

def roulette_selection(population):
    total = sum(ind.fitness for ind in population)
    pick = random.uniform(0, total)
    current = 0
    for ind in population:
        current += ind.fitness
        if current >= pick:
            return ind

def tournament_selection(population, k=3):
    selected = random.sample(population, k)
    return max(selected, key=lambda ind: ind.fitness)
