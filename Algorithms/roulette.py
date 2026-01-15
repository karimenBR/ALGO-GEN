#this a wheel selection algorithme
import random


def roulette_selection(population, fitness_scores, num_parents):
    if any(f < 0 for f in fitness_scores):
        raise ValueError("Fitness scores must be non-negative for roulette selection.")

    selected_parents = random.choices(
        population=population,
        weights=fitness_scores,
        k=num_parents
    )
    return selected_parents


# Example usage:
if __name__ == "__main__":
    current_population = ['Individual_A', 'Individual_B', 'Individual_C', 'Individual_D']
    fitnesses = [10, 25, 5, 40]  # Higher is better
    print("Roulette selection")
    print("population", current_population)
    # Select 2 parents
    parents = roulette_selection(current_population, fitnesses, 2)
    print(f"Selected parents: {parents}")
