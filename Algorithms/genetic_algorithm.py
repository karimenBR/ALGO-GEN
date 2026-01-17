import random
import time
import statistics
from core_functions.instance import load_instance, Individual
from core_functions.fitness import fitness
from core_functions.crossover import one_point, uniform
from core_functions.mutation import swap, inversion, random_reset
from core_functions.Selection import roulette_selection, tournament_selection


class GeneticAlgorithm:
    def __init__(self, instance, pop_size=100, crossover_rate=0.8,
                 mutation_rate=0.2, max_generations=1000,
                 crossover_op='one_point', mutation_op='swap',
                 selection_op='tournament', tournament_k=3):
        """
        Initialise l'algorithme génétique

        Args:
            instance: Instance du problème
            pop_size: Taille de la population
            crossover_rate: Probabilité de croisement
            mutation_rate: Probabilité de mutation
            max_generations: Nombre maximal de générations
            crossover_op: Type de croisement ('one_point' ou 'uniform')
            mutation_op: Type de mutation ('swap', 'inversion', 'random_reset')
            selection_op: Type de sélection ('roulette' ou 'tournament')
            tournament_k: Taille du tournoi (pour tournament_selection)
        """
        self.instance = instance
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.tournament_k = tournament_k

        # Sélection des opérateurs
        self.crossover_op = one_point if crossover_op == 'one_point' else uniform

        if mutation_op == 'swap':
            self.mutation_op = swap
        elif mutation_op == 'inversion':
            self.mutation_op = inversion
        else:
            self.mutation_op = lambda chrom: random_reset(chrom, 0, max(instance.data))

        self.selection_op = selection_op

        # Historique pour l'analyse
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_cost_history = []
        self.diversity_history = []

    def initialize_population(self):
        """Crée une population initiale aléatoire"""
        population = []
        for _ in range(self.pop_size):
            chromosome = [random.randint(0, max(self.instance.data))
                          for _ in range(self.instance.size)]
            ind = Individual(chromosome)
            fitness(ind, self.instance)
            population.append(ind)
        return population

    def select_parent(self, population):
        """Sélectionne un parent selon l'opérateur choisi"""
        if self.selection_op == 'roulette':
            return roulette_selection(population)
        else:
            return tournament_selection(population, self.tournament_k)

    def calculate_diversity(self, population):
        """Calcule la diversité génétique de la population"""
        if len(population) < 2:
            return 0

        differences = []
        sample_size = min(10, len(population) - 1)

        for i in range(min(20, len(population))):
            for j in range(i + 1, min(i + sample_size, len(population))):
                diff = sum(1 for g1, g2 in zip(population[i].chromosome,
                                               population[j].chromosome) if g1 != g2)
                differences.append(diff / len(population[i].chromosome))

        return statistics.mean(differences) if differences else 0

    def run(self, verbose=True):
        """Exécute l'algorithme génétique"""
        start_time = time.time()

        # Initialisation
        population = self.initialize_population()
        best_individual = max(population, key=lambda ind: ind.fitness)

        generation = 0
        stagnation_counter = 0

        if verbose:
            print(f"Génération 0: Meilleur coût = {best_individual.cost}, "
                  f"Fitness = {best_individual.fitness:.6f}")

        # Boucle principale
        while generation < self.max_generations:
            generation += 1
            new_population = []

            # Élitisme : conserver le meilleur individu
            new_population.append(best_individual)

            # Génération de la nouvelle population
            while len(new_population) < self.pop_size:
                # Sélection
                parent1 = self.select_parent(population)
                parent2 = self.select_parent(population)

                # Croisement
                if random.random() < self.crossover_rate:
                    child1_chrom, child2_chrom = self.crossover_op(
                        parent1.chromosome, parent2.chromosome)
                else:
                    child1_chrom = parent1.chromosome.copy()
                    child2_chrom = parent2.chromosome.copy()

                # Mutation
                if random.random() < self.mutation_rate:
                    child1_chrom = self.mutation_op(child1_chrom)
                if random.random() < self.mutation_rate:
                    child2_chrom = self.mutation_op(child2_chrom)

                # Création des nouveaux individus
                child1 = Individual(child1_chrom)
                child2 = Individual(child2_chrom)

                # Évaluation
                fitness(child1, self.instance)
                fitness(child2, self.instance)

                new_population.extend([child1, child2])

            # Limiter à la taille de population
            population = new_population[:self.pop_size]

            # Mise à jour du meilleur individu
            current_best = max(population, key=lambda ind: ind.fitness)
            if current_best.fitness > best_individual.fitness:
                best_individual = current_best
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Enregistrement des statistiques
            avg_fitness = statistics.mean(ind.fitness for ind in population)
            diversity = self.calculate_diversity(population)

            self.best_fitness_history.append(best_individual.fitness)
            self.avg_fitness_history.append(avg_fitness)
            self.best_cost_history.append(best_individual.cost)
            self.diversity_history.append(diversity)

            # Affichage périodique
            if verbose and generation % 100 == 0:
                print(f"Génération {generation}: Coût = {best_individual.cost}, "
                      f"Fitness = {best_individual.fitness:.6f}, "
                      f"Diversité = {diversity:.4f}")

            # Critère d'arrêt : stagnation
            if stagnation_counter > 200:
                if verbose:
                    print(f"Arrêt par stagnation à la génération {generation}")
                break

        execution_time = time.time() - start_time

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Algorithme terminé en {execution_time:.2f} secondes")
            print(f"Nombre de générations : {generation}")
            print(f"Meilleure solution : {best_individual.chromosome}")
            print(f"Coût : {best_individual.cost}")
            print(f"Fitness : {best_individual.fitness:.6f}")
            print(f"{'=' * 60}\n")

        return {
            'best_individual': best_individual,
            'best_chromosome': best_individual.chromosome,
            'best_cost': best_individual.cost,
            'best_fitness': best_individual.fitness,
            'execution_time': execution_time,
            'generations': generation,
            'history': {
                'best_fitness': self.best_fitness_history,
                'avg_fitness': self.avg_fitness_history,
                'best_cost': self.best_cost_history,
                'diversity': self.diversity_history
            }
        }


# Exemple d'utilisation
if __name__ == "__main__":
    # Charger une instance
    instance = load_instance("data/instance1.txt")

    # Créer et exécuter l'AG
    ga = GeneticAlgorithm(
        instance,
        pop_size=100,
        crossover_rate=0.8,
        mutation_rate=0.2,
        max_generations=500,
        crossover_op='one_point',
        mutation_op='swap',
        selection_op='tournament',
        tournament_k=3
    )

    result = ga.run(verbose=True)