
class Instance:
    def __init__(self, size, data):
        self.size = size
        self.data = data

def load_instance(path):
    with open(path, "r") as f:
        lines = f.read().strip().splitlines()
        size = int(lines[0])
        data = list(map(int, lines[1].split()))
    return Instance(size, data)

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.cost = None
        self.fitness = None
