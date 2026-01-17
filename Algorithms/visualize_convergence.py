import matplotlib.pyplot as plt
import json
import os
from genetic_algorithm import GeneticAlgorithm
from core_functions.instance import load_instance


def plot_single_run(result, title="Convergence de l'Algorithme Génétique",
                    save_path=None):
    """
    Visualise la convergence d'une seule exécution

    Args:
        result: Résultat retourné par ga.run()
        title: Titre du graphique
        save_path: Chemin pour sauvegarder (None = affichage)
    """
    history = result['history']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 1. Évolution du coût
    ax1 = axes[0, 0]
    ax1.plot(history['best_cost'], 'b-', linewidth=2, label='Meilleur coût')
    ax1.set_xlabel('Génération')
    ax1.set_ylabel('Coût')
    ax1.set_title('Évolution du Meilleur Coût')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Évolution du fitness
    ax2 = axes[0, 1]
    ax2.plot(history['best_fitness'], 'g-', linewidth=2, label='Meilleur fitness')
    ax2.plot(history['avg_fitness'], 'r--', linewidth=1.5, label='Fitness moyen')
    ax2.set_xlabel('Génération')
    ax2.set_ylabel('Fitness')
    ax2.set_title('Évolution du Fitness')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Diversité génétique
    ax3 = axes[1, 0]
    ax3.plot(history['diversity'], 'm-', linewidth=2)
    ax3.set_xlabel('Génération')
    ax3.set_ylabel('Diversité')
    ax3.set_title('Diversité Génétique')
    ax3.grid(True, alpha=0.3)

    # 4. Statistiques de l'exécution
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""
    STATISTIQUES DE L'EXÉCUTION
    {'=' * 40}

    Meilleur coût trouvé: {result['best_cost']}
    Meilleur fitness: {result['best_fitness']:.6f}

    Temps d'exécution: {result['execution_time']:.2f} s
    Nombre de générations: {result['generations']}

    Coût initial: {history['best_cost'][0]}
    Amélioration: {history['best_cost'][0] - result['best_cost']}
    Taux d'amélioration: {((history['best_cost'][0] - result['best_cost']) / history['best_cost'][0] * 100):.2f}%
    """
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graphique sauvegardé dans {save_path}")
    else:
        plt.show()

    plt.close()


def compare_configurations(instance_path, configurations, save_path=None):
    """
    Compare visuellement plusieurs configurations sur une instance

    Args:
        instance_path: Chemin vers l'instance
        configurations: Liste de dictionnaires de configuration
        save_path: Chemin pour sauvegarder (None = affichage)
    """
    instance = load_instance(instance_path)
    instance_name = os.path.basename(instance_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Comparaison des Configurations - {instance_name}',
                 fontsize=16, fontweight='bold')

    results = []
    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown', 'pink', 'gray']

    print(f"\nExécution des configurations sur {instance_name}...")

    for i, config in enumerate(configurations):
        print(f"  Configuration {i + 1}/{len(configurations)}: {config['name']}")

        ga = GeneticAlgorithm(
            instance,
            pop_size=config.get('pop_size', 100),
            crossover_rate=config.get('crossover_rate', 0.8),
            mutation_rate=config.get('mutation_rate', 0.2),
            max_generations=config.get('max_generations', 500),
            crossover_op=config.get('crossover_op', 'one_point'),
            mutation_op=config.get('mutation_op', 'swap'),
            selection_op=config.get('selection_op', 'tournament'),
            tournament_k=config.get('tournament_k', 3)
        )

        result = ga.run(verbose=False)
        results.append(result)

        color = colors[i % len(colors)]
        label = config['name'][:30]  # Limiter la longueur du label

        # Coût
        axes[0, 0].plot(result['history']['best_cost'],
                        color=color, linewidth=1.5, label=label)

        # Fitness
        axes[0, 1].plot(result['history']['best_fitness'],
                        color=color, linewidth=1.5, label=label)

        # Diversité
        axes[1, 0].plot(result['history']['diversity'],
                        color=color, linewidth=1.5, label=label)

    # Configuration des sous-graphiques
    axes[0, 0].set_xlabel('Génération')
    axes[0, 0].set_ylabel('Coût')
    axes[0, 0].set_title('Évolution du Coût')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_xlabel('Génération')
    axes[0, 1].set_ylabel('Fitness')
    axes[0, 1].set_title('Évolution du Fitness')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].set_xlabel('Génération')
    axes[1, 0].set_ylabel('Diversité')
    axes[1, 0].set_title('Diversité Génétique')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    # Tableau comparatif
    axes[1, 1].axis('off')

    # Créer un tableau de comparaison
    table_data = []
    for i, (config, result) in enumerate(zip(configurations, results)):
        table_data.append([
            f"Config {i + 1}",
            f"{result['best_cost']}",
            f"{result['execution_time']:.2f}s",
            f"{result['generations']}"
        ])

    table = axes[1, 1].table(cellText=table_data,
                             colLabels=['Config', 'Coût', 'Temps', 'Gén.'],
                             cellLoc='center',
                             loc='center',
                             colWidths=[0.3, 0.2, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Colorier l'en-tête
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Colorier la meilleure ligne
    best_idx = min(range(len(results)),
                   key=lambda i: results[i]['best_cost'])
    for i in range(4):
        table[(best_idx + 1, i)].set_facecolor('#FFD700')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nGraphique de comparaison sauvegardé dans {save_path}")
    else:
        plt.show()

    plt.close()


def plot_results_from_json(json_path, output_folder='plots'):
    """
    Crée des graphiques à partir d'un fichier JSON de résultats

    Args:
        json_path: Chemin vers le fichier JSON
        output_folder: Dossier de sortie pour les graphiques
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    os.makedirs(output_folder, exist_ok=True)

    results = data['detailed_results']
    report = data['report']

    # Graphique 1: Comparaison des coûts moyens par configuration
    configs = list(report['by_configuration'].keys())
    avg_costs = [report['by_configuration'][c]['avg_cost'] for c in configs]
    std_costs = [report['by_configuration'][c]['std_cost'] for c in configs]

    plt.figure(figsize=(12, 6))
    x_pos = range(len(configs))
    plt.bar(x_pos, avg_costs, yerr=std_costs, alpha=0.7, capsize=5)
    plt.xlabel('Configuration')
    plt.ylabel('Coût Moyen')
    plt.title('Comparaison des Coûts Moyens par Configuration')
    plt.xticks(x_pos, [f"Config {i + 1}" for i in range(len(configs))],
               rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'cost_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Graphique 2: Temps d'exécution par configuration
    avg_times = [report['by_configuration'][c]['avg_time'] for c in configs]

    plt.figure(figsize=(12, 6))
    plt.bar(x_pos, avg_times, alpha=0.7, color='orange')
    plt.xlabel('Configuration')
    plt.ylabel('Temps Moyen (s)')
    plt.title('Comparaison des Temps d\'Exécution')
    plt.xticks(x_pos, [f"Config {i + 1}" for i in range(len(configs))],
               rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'time_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Graphique 3: Écarts à l'optimum (si disponible)
    if 'avg_gap' in report['by_configuration'][configs[0]]:
        avg_gaps = [report['by_configuration'][c].get('avg_gap', 0)
                    for c in configs]

        plt.figure(figsize=(12, 6))
        plt.bar(x_pos, avg_gaps, alpha=0.7, color='green')
        plt.xlabel('Configuration')
        plt.ylabel('Écart à l\'Optimum (%)')
        plt.title('Écart Moyen à l\'Optimum Connu')
        plt.xticks(x_pos, [f"Config {i + 1}" for i in range(len(configs))],
                   rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'gap_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\nGraphiques sauvegardés dans le dossier '{output_folder}'")


# Exemple d'utilisation
if __name__ == "__main__":
    # Test 1: Visualiser une seule exécution
    print("Test 1: Exécution simple et visualisation")
    instance = load_instance("data/instance1.txt")

    ga = GeneticAlgorithm(
        instance,
        pop_size=100,
        crossover_rate=0.8,
        mutation_rate=0.2,
        max_generations=500,
        crossover_op='one_point',
        mutation_op='swap',
        selection_op='tournament'
    )

    result = ga.run(verbose=True)
    plot_single_run(result,
                    title="AG - One-point + Swap + Tournament",
                    save_path="convergence_simple.png")

    # Test 2: Comparer plusieurs configurations
    print("\n" + "=" * 60)
    print("Test 2: Comparaison de configurations")

    configurations = [
        {
            'name': 'One-point + Swap',
            'crossover_op': 'one_point',
            'mutation_op': 'swap',
            'selection_op': 'tournament'
        },
        {
            'name': 'Uniform + Inversion',
            'crossover_op': 'uniform',
            'mutation_op': 'inversion',
            'selection_op': 'tournament'
        },
        {
            'name': 'One-point + High Mutation',
            'crossover_op': 'one_point',
            'mutation_op': 'swap',
            'mutation_rate': 0.4,
            'selection_op': 'tournament'
        }
    ]

    compare_configurations("data/instance1.txt",
                           configurations,
                           save_path="comparison_configs.png")