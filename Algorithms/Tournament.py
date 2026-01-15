import random


import random

def tournament_selection(population, scores, k=3):
    selection_ix = random.randint(0, len(population) - 1)

    for _ in range(k - 1):
        ix = random.randint(0, len(population) - 1)
        if scores[ix] > scores[selection_ix]:
            selection_ix = ix

    return population[selection_ix]

import random

def tournament_selection_population(population, scores, k=3):
    new_population = []

    pop_size = len(population)

    for _ in range(pop_size):
        selection_ix = random.randint(0, pop_size - 1)

        for _ in range(k - 1):
            ix = random.randint(0, pop_size - 1)
            if scores[ix] > scores[selection_ix]:
                selection_ix = ix

        new_population.append(population[selection_ix])

    return new_population


if __name__ == "__main__":
    population = ['Individual_A', 'Individual_B', 'Individual_C', 'Individual_D']
    scores = [10, 25, 5, 40]

    selected = tournament_selection_population(population, scores,3)
    print("Selected individual:", selected)
