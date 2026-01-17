import os
import statistics
import json
from pathlib import Path
from genetic_algorithm import GeneticAlgorithm
from core_functions.instance import load_instance


def evaluate_single_config(instance_path, config, runs=5, verbose=False):
    """
    Évalue une configuration sur une instance donnée

    Args:
        instance_path: Chemin vers l'instance
        config: Dictionnaire de paramètres
        runs: Nombre d'exécutions
        verbose: Affichage détaillé

    Returns:
        Dictionnaire avec les résultats agrégés
    """
    instance = load_instance(instance_path)
    instance_name = os.path.basename(instance_path)

    costs = []
    times = []
    generations = []

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Instance: {instance_name}")
        print(f"Configuration: {config['name']}")
        print(f"{'=' * 60}")

    for run in range(runs):
        ga = GeneticAlgorithm(
            instance,
            pop_size=config.get('pop_size', 100),
            crossover_rate=config.get('crossover_rate', 0.8),
            mutation_rate=config.get('mutation_rate', 0.2),
            max_generations=config.get('max_generations', 1000),
            crossover_op=config.get('crossover_op', 'one_point'),
            mutation_op=config.get('mutation_op', 'swap'),
            selection_op=config.get('selection_op', 'tournament'),
            tournament_k=config.get('tournament_k', 3)
        )

        result = ga.run(verbose=False)
        costs.append(result['best_cost'])
        times.append(result['execution_time'])
        generations.append(result['generations'])

        if verbose:
            print(f"  Run {run + 1}: Coût = {result['best_cost']}, "
                  f"Temps = {result['execution_time']:.2f}s, "
                  f"Générations = {result['generations']}")

    avg_cost = statistics.mean(costs)
    std_cost = statistics.stdev(costs) if len(costs) > 1 else 0
    avg_time = statistics.mean(times)
    avg_gen = statistics.mean(generations)

    if verbose:
        print(f"\n  Résultats agrégés:")
        print(f"    Coût moyen: {avg_cost:.2f} ± {std_cost:.2f}")
        print(f"    Meilleur: {min(costs)}, Pire: {max(costs)}")
        print(f"    Temps moyen: {avg_time:.2f}s")
        print(f"    Générations moyennes: {avg_gen:.0f}")

    return {
        'instance': instance_name,
        'config': config['name'],
        'avg_cost': avg_cost,
        'std_cost': std_cost,
        'min_cost': min(costs),
        'max_cost': max(costs),
        'avg_time': avg_time,
        'avg_generations': avg_gen,
        'all_costs': costs,
        'all_times': times
    }


def evaluate_all_instances(data_folder, configurations, runs=5,
                           known_optimums=None, verbose=True):
    """
    Évalue toutes les configurations sur toutes les instances

    Args:
        data_folder: Dossier contenant les instances
        configurations: Liste de configurations à tester
        runs: Nombre d'exécutions par configuration/instance
        known_optimums: Dictionnaire {instance_name: optimum_value}
        verbose: Affichage détaillé

    Returns:
        Liste des résultats et rapport statistique
    """
    instance_files = sorted([f for f in os.listdir(data_folder)
                             if f.endswith('.txt')])

    if not instance_files:
        print(f"Aucune instance trouvée dans {data_folder}")
        return [], {}

    all_results = []

    print(f"\n{'#' * 60}")
    print(f"# ÉVALUATION DES PERFORMANCES")
    print(f"# Nombre d'instances: {len(instance_files)}")
    print(f"# Nombre de configurations: {len(configurations)}")
    print(f"# Exécutions par test: {runs}")
    print(f"{'#' * 60}\n")

    for config in configurations:
        config_results = []

        for instance_file in instance_files:
            instance_path = os.path.join(data_folder, instance_file)

            result = evaluate_single_config(
                instance_path,
                config,
                runs=runs,
                verbose=verbose
            )

            # Calculer l'écart par rapport à l'optimum si connu
            if known_optimums and instance_file in known_optimums:
                optimum = known_optimums[instance_file]
                gap = ((result['avg_cost'] - optimum) / optimum) * 100
                result['gap_to_optimum'] = gap
                result['known_optimum'] = optimum

            config_results.append(result)
            all_results.append(result)

        # Statistiques par configuration
        print_config_summary(config['name'], config_results, known_optimums)

    # Rapport global
    report = generate_global_report(all_results, configurations, known_optimums)

    return all_results, report


def print_config_summary(config_name, results, known_optimums=None):
    """Affiche un résumé pour une configuration"""
    print(f"\n{'=' * 60}")
    print(f"RÉSUMÉ - {config_name}")
    print(f"{'=' * 60}")

    avg_costs = [r['avg_cost'] for r in results]
    avg_times = [r['avg_time'] for r in results]

    print(f"  Coût moyen global: {statistics.mean(avg_costs):.2f}")
    print(f"  Écart-type des coûts: {statistics.stdev(avg_costs):.2f}")
    print(f"  Temps moyen d'exécution: {statistics.mean(avg_times):.2f}s")

    if known_optimums:
        gaps = [r['gap_to_optimum'] for r in results if 'gap_to_optimum' in r]
        if gaps:
            print(f"  Écart moyen à l'optimum: {statistics.mean(gaps):.2f}%")
            print(f"  Écart-type des écarts: {statistics.stdev(gaps):.2f}%")


def generate_global_report(all_results, configurations, known_optimums=None):
    """Génère un rapport statistique global"""
    report = {
        'by_configuration': {},
        'by_instance': {},
        'global_stats': {}
    }

    # Statistiques par configuration
    for config in configurations:
        config_name = config['name']
        config_results = [r for r in all_results if r['config'] == config_name]

        avg_costs = [r['avg_cost'] for r in config_results]
        avg_times = [r['avg_time'] for r in config_results]

        report['by_configuration'][config_name] = {
            'avg_cost': statistics.mean(avg_costs),
            'std_cost': statistics.stdev(avg_costs) if len(avg_costs) > 1 else 0,
            'min_cost': min(avg_costs),
            'max_cost': max(avg_costs),
            'avg_time': statistics.mean(avg_times)
        }

        if known_optimums:
            gaps = [r['gap_to_optimum'] for r in config_results
                    if 'gap_to_optimum' in r]
            if gaps:
                report['by_configuration'][config_name]['avg_gap'] = statistics.mean(gaps)
                report['by_configuration'][config_name]['std_gap'] = statistics.stdev(gaps)

    # Statistiques par instance
    instances = list(set(r['instance'] for r in all_results))
    for instance in instances:
        instance_results = [r for r in all_results if r['instance'] == instance]
        costs = [r['avg_cost'] for r in instance_results]

        report['by_instance'][instance] = {
            'best_config': min(instance_results, key=lambda x: x['avg_cost'])['config'],
            'best_cost': min(costs),
            'worst_cost': max(costs)
        }

    return report


def save_results(results, report, output_file='results.json'):
    """Sauvegarde les résultats en JSON"""
    output = {
        'detailed_results': results,
        'report': report
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nRésultats sauvegardés dans {output_file}")


# Exemple d'utilisation
if __name__ == "__main__":
    # Définir les configurations à tester
    configurations = [
        {
            'name': 'Config 1: One-point + Swap + Tournament',
            'pop_size': 100,
            'crossover_rate': 0.8,
            'mutation_rate': 0.2,
            'max_generations': 500,
            'crossover_op': 'one_point',
            'mutation_op': 'swap',
            'selection_op': 'tournament',
            'tournament_k': 3
        },
        {
            'name': 'Config 2: Uniform + Inversion + Tournament',
            'pop_size': 100,
            'crossover_rate': 0.8,
            'mutation_rate': 0.2,
            'max_generations': 500,
            'crossover_op': 'uniform',
            'mutation_op': 'inversion',
            'selection_op': 'tournament',
            'tournament_k': 3
        },
        {
            'name': 'Config 3: One-point + Swap + Roulette',
            'pop_size': 100,
            'crossover_rate': 0.8,
            'mutation_rate': 0.2,
            'max_generations': 500,
            'crossover_op': 'one_point',
            'mutation_op': 'swap',
            'selection_op': 'roulette'
        },
        {
            'name': 'Config 4: High Mutation',
            'pop_size': 100,
            'crossover_rate': 0.7,
            'mutation_rate': 0.4,
            'max_generations': 500,
            'crossover_op': 'one_point',
            'mutation_op': 'swap',
            'selection_op': 'tournament',
            'tournament_k': 3
        },
        {
            'name': 'Config 5: Large Population',
            'pop_size': 200,
            'crossover_rate': 0.8,
            'mutation_rate': 0.2,
            'max_generations': 300,
            'crossover_op': 'one_point',
            'mutation_op': 'swap',
            'selection_op': 'tournament',
            'tournament_k': 5
        }
    ]

    # Optimums connus (à remplir selon vos instances)
    known_optimums = {
        'instance1.txt': 100,  # Exemple, à remplacer
        'instance2.txt': 150,  # Exemple, à remplacer
    }

    # Évaluation
    results, report = evaluate_all_instances(
        'data',
        configurations,
        runs=5,
        known_optimums=None,  # Mettre known_optimums si vous les avez
        verbose=True
    )

    # Sauvegarder les résultats
    save_results(results, report, 'evaluation_results.json')

    # Afficher le meilleur
    print("\n" + "=" * 60)
    print("MEILLEURE CONFIGURATION PAR INSTANCE:")
    print("=" * 60)
    for instance, data in report['by_instance'].items():
        print(f"{instance}: {data['best_config']} (Coût: {data['best_cost']:.2f})")