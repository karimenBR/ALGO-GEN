import random

def swap(chromosome):
    child = chromosome.copy()
    i, j = random.sample(range(len(child)), 2)
    child[i], child[j] = child[j], child[i]
    return child

def inversion(chromosome):
    child = chromosome.copy()
    i, j = sorted(random.sample(range(len(child)), 2))
    child[i:j] = reversed(child[i:j])
    return child

def random_reset(chromosome, min_val=0, max_val=9):
    child = chromosome.copy()
    i = random.randint(0, len(child) - 1)
    child[i] = random.randint(min_val, max_val)
    return child

if __name__ == "__main__":
    chrom = [1, 2, 3, 4, 5]

    print("Swap:", swap(chrom))
    print("Inversion:", inversion(chrom))
    print("Random reset:", random_reset(chrom))
