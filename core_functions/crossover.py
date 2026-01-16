import random

def one_point(parent1, parent2):
    assert len(parent1) == len(parent2), "Chromosomes must have same length"

    point = random.randint(1, len(parent1) - 1)

    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    return child1, child2


def uniform(parent1, parent2):
    assert len(parent1) == len(parent2), "Chromosomes must have same length"

    child1, child2 = [], []

    for g1, g2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(g1)
            child2.append(g2)
        else:
            child1.append(g2)
            child2.append(g1)

    return child1, child2

if __name__ == "__main__":
    p1 = [1, 2, 3, 4, 5]
    p2 = [5, 4, 3, 2, 1]

    c1, c2 = one_point(p1, p2)
    print("One-point:", c1, c2)

    c1, c2 = uniform(p1, p2)
    print("Uniform:", c1, c2)
