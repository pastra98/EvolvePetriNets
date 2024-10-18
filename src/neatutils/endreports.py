import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
from scipy.stats import gaussian_kde

import json
import gc
import pickle
import gzip
import os

import pandas as pd
import numpy as np

from collections import defaultdict
from neat.genome import GeneticNet

FSIZE = (10, 5)


def save_report(ga_info: dict, savedir: str) -> None:
    """saves some plots in the specified dir
    """
    matplotlib.use('Agg')

    # save run report
    run_report(ga_info, savedir=savedir)

    # get the full history df, create the dataframes and save them
    full_history = ga_info["history"]
    os.makedirs(f"{savedir}/data")

    pop_df = get_population_df(full_history)
    pop_df.to_feather(f"{savedir}/data/population.feather")
    gen_info_df = get_gen_info_df(full_history)
    gen_info_df.to_feather(f"{savedir}/data/gen_info.feather")

    use_species = "species" in full_history[1]
    if use_species:
        species_df = get_species_df(full_history)
        species_df.to_feather(f"{savedir}/data/species.feather")
        save_species_leaders(ga_info["species_leaders"], savedir)
        # plot the species
        species_cmap = get_species_color_map(species_df)
        species_stackplot(species_df, species_cmap, savedir)
        plot_species_evolution(species_df, pop_df, gen_info_df, species_cmap, savedir)
        plot_avg_species_fit(species_df, species_cmap, savedir)
        # ridgeline_plot(pop_df, species_cmap, savedir)
    
    # save the improvements, species leaders & best genome, delete ga info after
    save_improvements(ga_info["improvements"], savedir)
    save_genome_gviz(ga_info["best_genome"], savedir, name_prefix="best_genome")
    pickle_genome(ga_info["best_genome"], "best_genome", savedir)

    # make the plots
    time_stackplot(gen_info_df, savedir)
    fitness_plot(gen_info_df, use_species, savedir)
    components_plot(gen_info_df, savedir)
    unique_components(pop_df, savedir)
    metrics_plot(pop_df, savedir)
    mutation_stats = mutation_effects_plot(pop_df, savedir)
    mutation_stats.to_feather(f"{savedir}/data/mutation_stats.feather")

    # analysis of best genome mutation lineage
    best_genomes = filter_best_genomes(gen_info_df, pop_df)
    best_genome_lineage(best_genomes, savedir)
    best_genome_mutation_analysis(best_genomes, savedir)

    # pickle component tracker
    save_component_dict(ga_info["component_dict"], f"{savedir}/data/component_dict.pkl.gz")

    # close all plots and free memory
    plt.close("all")
    gc.collect()


def run_report(ga_info, savedir: str) -> None:
    """Short txt file with overview info about run
    """
    savedir = savedir.rstrip("/reports")
    with open(f"{savedir}/report.txt", "w") as f:
        f.write(f"Best fitness:\n{ga_info['best_genome'].fitness}\n")
        f.write(f"Best metrics:\n")
        json.dump(ga_info['best_genome'].fitness_metrics, f, indent=4)
        f.write(f"\nTotal components discovered:\n{ga_info['total_components']}\n")
        f.write(f"Duration of run:\n{ga_info['time']}")

# ---------- CONVERT TO DATAFRAMES

def get_species_df(full_history: dict):
    l = []
    for gen, info_d in full_history.items():
        for s in info_d["species"]:
            l.append(s | {"gen": gen})
    return pd.DataFrame(l)


def get_population_df(full_history: dict):
    l = []
    for gen, info_d in full_history.items():
        for g in info_d["population"]:
            l.append(g | {"gen": gen})
    df = pd.DataFrame(l)
    # expand the fitness metrics to columns
    metrics = pd.json_normalize(df['fitness_metrics']).add_prefix("metric_")
    return pd.concat([df.drop(columns=['fitness_metrics']), metrics], axis=1)


def get_gen_info_df(full_history: dict):
    l = []
    excludes = ["species", "population"] # exclude the lists
    for gen, info_d in full_history.items():
        filtered_info_d = {k: v for k, v in info_d.items() if k not in excludes}
        l.append(filtered_info_d | {"gen": gen})
    df = pd.DataFrame(l)
    df.set_index("gen")
    return df

# ---------- SERIALIZATION

def save_genome_gviz(genome: GeneticNet, savedir: str, name_prefix=""):
    """Save gviz render of specified genome
    """
    with open(f"{savedir}/{name_prefix}_id-{genome.id}.pdf", "wb") as f:
        f.write(genome.get_gviz().pipe(format="pdf"))


def save_improvements(improvements: str, savedir: str):
    """Save gviz render of every fitness improvement
    """
    os.makedirs(f"{savedir}/improvements")
    for gen, g in improvements.items():
        save_genome_gviz(g, f"{savedir}/improvements", name_prefix=f"improvement_gen-{gen}")


def save_species_leaders(leaders: list[GeneticNet], savedir: str):
    """Save gviz render of leader of every species
    """
    os.makedirs(f"{savedir}/leaders")
    for g in leaders:
        save_genome_gviz(g, f"{savedir}/leaders", name_prefix=f"species_{g.species_id}")


def pickle_genome(genome: GeneticNet, name: str, savedir: str):
    """Pickle the full GeneticNet to disk
    """
    genome.clear_cache()
    with open(f"{savedir}/{name}.pkl", "wb") as f:
        pickle.dump(genome, f)


def save_component_dict(component_dict: dict, fpath: str):
    """Pickle the component dictionary with max compression
    """
    with gzip.open(fpath, 'wb') as f:
        pickle.dump(component_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

# ---------- DF FILTERING HELPER FUNC

def filter_best_genomes(gen_info_df, pop_df):
    """Filters out the best genomes that resulted in fitness improvement from pop_df
    """
    def extract_best_genomes(gen_info_df, pop_df):
        merged_df = pd.merge(gen_info_df[['gen', 'best_genome']], pop_df, left_on=['gen', 'best_genome'], right_on=['gen', 'id'], how='inner')
        return merged_df

    def filter_improved_generations(gen_info_df):
        gen_info_df = gen_info_df.sort_values(by='gen').reset_index(drop=True)
        gen_info_df['previous_best_fitness'] = gen_info_df['best_genome_fitness'].shift(1)
        improved_gen_info_df = gen_info_df[gen_info_df['best_genome_fitness'] > gen_info_df['previous_best_fitness']]
        return improved_gen_info_df

    best_genomes_df = extract_best_genomes(gen_info_df, pop_df)
    improved_gen_info_df = filter_improved_generations(gen_info_df)
    return best_genomes_df[best_genomes_df['gen'].isin(improved_gen_info_df['gen'])]

# ---------- PLOTTING FUNCTIONS
# ----- PERFORMANCE PLOTS

def time_stackplot(gen_info_df: pd.DataFrame, savedir: str):
    """Stackplot of evaluation and pop update times
    """
    times_df = pd.DataFrame(gen_info_df["times"].tolist(), index=gen_info_df.index)
    plt.figure(figsize=FSIZE)
    plt.stackplot(
        times_df.index,
        times_df["evaluate_curr_generation"],
        times_df["pop_update"],
        labels=["Evaluate Current Generation", "Population Update"]
        )
    plt.legend(frameon=False)
    plt.title("Time Spent Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Time")
    plt.savefig(f"{savedir}/time_plot.pdf")

# ----- COMPONENT PLOTS

def components_plot(gen_info_df: pd.DataFrame, savedir: str):
    """Plot the total number of components over time
    """
    plt.figure(figsize=FSIZE)
    plt.plot(gen_info_df["num_total_components"])
    plt.title("Total components")
    plt.xlabel("Generation")
    plt.ylabel("num components")
    plt.savefig(f"{savedir}/num_components.pdf")


def unique_components(pop_df: pd.DataFrame, savedir: str):
    """Plot the number of unique components in every generation to visualize
    convergence
    """
    # Group by generation and count unique components
    unique_components_per_gen = pop_df.groupby('gen')['my_components'].apply(
        lambda x: len(set(component for sublist in x for component in sublist))
        )
    plot_df = pd.DataFrame({
        'Generation': unique_components_per_gen.index,
        'UniqueComponents': unique_components_per_gen.values
    })
    # Plot the unique components
    plt.figure(figsize=(10, 6))
    plt.plot(plot_df['Generation'], plot_df['UniqueComponents'], linestyle='-')
    plt.title('Number of Unique Components per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Number of Unique Components')
    plt.savefig(f"{savedir}/unique_components.pdf")


def ridgeline_plot(
        pop_df: pd.DataFrame,
        cmap: dict,
        savedir: str,
        n_points=200,
        overlap=0.5,
        alpha=0.6,
        top_n=20,
        ):
    """Ridgeline plot to show distribution of components among different species
    """
    # get component counts by species and total
    exploded_df = pop_df.explode('my_components')
    species_counts = exploded_df.groupby(['species_id', 'my_components']).size().unstack(fill_value=0)
    overall_counts = species_counts.sum().sort_values(ascending=False)
    # Select top components, get plotting data
    top_components = overall_counts.nlargest(top_n).index
    species_data = []
    species_labels = []
    for species_id in species_counts.index:
        species_data.append(species_counts.loc[species_id, top_components])
        species_labels.append(f"Species {species_id[:8]}")  # Using first 8 characters of UUID
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    n = len(species_data)
    max_value = max([max(d) for d in species_data])
    x = np.linspace(0, max_value, n_points)
    for i, (y, label) in enumerate(zip(species_data, species_labels)):
        y = np.array(y)
        kde = gaussian_kde(y)
        y_kde = kde(x)
        y_kde = y_kde / y_kde.max() * overlap
        ax.plot(x, y_kde + i, color='black', lw=0.5)
        if cmap and species_counts.index[i] in cmap:
            color = cmap[species_counts.index[i]]
        else:
            color = plt.cm.viridis(i / n)
        ax.fill_between(x, i, y_kde + i, alpha=alpha, color=color)
    # set up plot
    ax.set_yticks(range(n))
    ax.set_xticks([])
    ax.set_yticklabels(species_labels)
    ax.set_title('Distribution of Components Among Species')
    plt.tight_layout()
    plt.savefig(f"{savedir}/component_dist_plot.pdf", bbox_inches='tight')

# ----- METRICS/FITNESS PLOTS

def fitness_plot(gen_info_df: pd.DataFrame, use_species: bool, savedir: str):
    """Plot the best, best species and avg fitness of population
    """
    plt.figure(figsize=FSIZE)
    if use_species:
        plt.plot(gen_info_df[["best_genome_fitness", "best_species_avg_fitness", "avg_pop_fitness"]])
        plt.legend(["Best Genome Fitness", "Best Species Average Fitness", "Average Population Fitness"], frameon=False)
    else:
        plt.plot(gen_info_df[["best_genome_fitness", "avg_pop_fitness"]])
        plt.legend(["Best Genome Fitness", "Average Population Fitness"], frameon=False)
    plt.title("Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.savefig(f"{savedir}/fitness_plot.pdf")


def metrics_plot(pop_df: pd.DataFrame, savedir: str):
    """Combined plots of metrics for best genome and populaiton avg
    """
    def plot_metrics(df, title):
        plt.figure(figsize=FSIZE)
        for col in df.columns:
            plt.plot(df.index, df[col], label=col)
        plt.title(title)
        plt.xlabel('Generation')
        plt.ylabel('Metric Value')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(f"{savedir}/{title.split()[0]}_metrics.pdf", bbox_inches='tight')
    # find metrics columns, aggregate them over generations, plot best and avg
    metrics = [col for col in pop_df.columns if col.startswith("metric_")]
    aggregated_metrics = pop_df.groupby('gen')[metrics].agg(['max', 'mean'])
    df_best = aggregated_metrics.xs('max', level=1, axis=1)
    df_avg = aggregated_metrics.xs('mean', level=1, axis=1)
    plot_metrics(df_best, 'Best Genome Metrics Over Generations')
    plot_metrics(df_avg, 'Average Population Metrics Over Generations')

# ----- MUTATION PLOTS

def mutation_effects_plot(pop_df, savedir: str, max_gen: int = None):
    """
    Generate visualizations for mutation effects using pandas operations:
    1. Boxplot of fitness impacts with relative frequency on secondary axis
    2. Line chart of mutation impact per generation (up to max_gen if specified)

    Args:
    max_gen (int, optional): Maximum generation to include in the line plot. If None, include all generations.

    Returns:
    mutations effect summary df
    """
    # Calculate fitness deltas to parents
    df_with_parents = pop_df[pop_df["gen"] > 1].copy()
    df_with_parents = df_with_parents[df_with_parents["my_mutation"] != ""]  # Exclude empty strings
    fitness_dict = pop_df.set_index('id')['fitness'].to_dict()
    df_with_parents['fitness_difference'] = df_with_parents.apply(
        lambda row: row['fitness'] - fitness_dict[row['parent_id']], axis=1
    )
    
    # Group by mutation and calculate statistics
    mutation_stats = df_with_parents.groupby('my_mutation').agg({
        'fitness_difference': [
            'count',
            'min',
            lambda x: x.quantile(0.25),
            'median',
            lambda x: x.quantile(0.75),
            'max',
            'mean'
        ],
        'gen': 'unique'
    })
    mutation_stats.columns = ['frequency', 'min', '25%', 'median', '75%', 'max', 'mean', 'generations']
    mutation_stats = mutation_stats.sort_index()
    
    # Calculate relative frequencies
    total_mutations = mutation_stats['frequency'].sum()
    mutation_stats['relative_frequency'] = mutation_stats['frequency'] / total_mutations

    # Create boxplot with relative frequencies on secondary axis
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Mutation')
    ax1.set_ylabel('Fitness Impact Distribution', color=color)
    ax1.axhline(y=0, color='lightgray', linestyle='--')
    bp = ax1.boxplot(
        [group['fitness_difference'].values for name, group in df_with_parents.groupby('my_mutation')],
        patch_artist=True, meanline=True, showmeans=True, showfliers=False
    )
    for box in bp['boxes']:
        box.set(color=color, linewidth=2)
        box.set(facecolor='lightblue')
    ax1.set_xticks(range(1, len(mutation_stats) + 1))
    ax1.set_xticklabels(mutation_stats.index, rotation=45, ha='right')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Relative Frequency', color=color)
    ax2.plot(range(1, len(mutation_stats) + 1), mutation_stats['relative_frequency'], color=color, marker='o', linestyle='-')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title('Fitness Impact Distribution and Relative Frequency of Each Mutation')
    fig.tight_layout()
    fig.savefig(f"{savedir}/mutation_analysis.pdf", dpi=300)
    plt.close(fig)
    
    # Create line chart for mutations over time
    fig, ax = plt.subplots(figsize=(12, 6))
    for mutation, data in df_with_parents.groupby('my_mutation'):
        impact_by_gen = data.groupby('gen')['fitness_difference'].apply(list).reset_index()
        if max_gen is not None:
            impact_by_gen = impact_by_gen[impact_by_gen['gen'] <= max_gen]
        ax.plot(impact_by_gen['gen'], impact_by_gen['fitness_difference'].apply(np.mean), label=mutation)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Average Fitness Impact')
    title = 'Average Impact of Mutations Over Generations'
    if max_gen is not None:
        title += f' (up to gen {max_gen})'
        ax.set_xlim(right=max_gen)
    ax.set_title(title)
    
    # Add legend with adjusted layout
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(f"{savedir}/mutations_over_time.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Save summary statistics
    mutation_stats.drop('generations', axis=1).to_markdown(f"{savedir}/mutation_effects.txt")
    return mutation_stats


def best_genome_lineage(best_genomes_df: pd.DataFrame, savedir=str):
    """Scatterplot of the mutation history of the best genome
    """
    gens = best_genomes_df['gen']
    fitnesses = best_genomes_df['fitness']
    mutations = best_genomes_df['my_mutation']

    # Generate a color and marker for each mutation type
    unique_mutations = mutations.unique()
    colors = plt.colormaps['tab10'](range(len(unique_mutations)))
    markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'X', 'P', 'h']  # Select a few markers
    mutation_style_map = {mutation: (colors[i], markers[i % len(markers)]) for i, mutation in enumerate(unique_mutations)}
    
    plt.figure(figsize=FSIZE)
    
    # Plot points for each generation based on mutation type
    for mutation in unique_mutations:
        mutation_mask = (mutations == mutation)
        plt.scatter(
            gens[mutation_mask],
            fitnesses[mutation_mask],
            color=mutation_style_map[mutation][0],
            marker=mutation_style_map[mutation][1],
            s=30,  # Size of the marker
            label=f'Mutation: {mutation}'
        )
    
    plt.legend(loc='lower right', fontsize='medium')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title('Best Genome: Fitness Progression with Mutation Types', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{savedir}/best_mutation_lineage.pdf")


def best_genome_mutation_analysis(best_genomes_df: pd.DataFrame, savedir: str):
    """Similar to mutation impacts, but this time only the mutations that actually
    improved the fitness of the best genome
    """
    # filter out mutation frequencies and mean fitness
    mutations = best_genomes_df['my_mutation']
    mutation_counts = mutations.value_counts().sort_index()
    mean_fitness_impact = best_genomes_df.groupby(mutations)['fitness'].mean().sort_index()
    mutation_types = mutation_counts.index

    fig, ax1 = plt.subplots(figsize=FSIZE)
    # X-axis positions for the bars
    x = np.arange(len(mutation_types))
    width = 0.4 # Bar width

    # Plot frequency of mutations on the left y-axis
    ax1.bar(x - width/2, mutation_counts, width, color='b', label='Mutation Count')
    ax1.set_ylabel('Frequency')
    ax1.set_xlabel('Mutation Type')
    # set ticks
    ax1.set_xticks(x)
    ax1.set_xticklabels(mutation_types, rotation=45, ha='right', fontsize=10)
    ax1.tick_params(axis='y')

    # Create second y-axis for fitness impact
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, mean_fitness_impact, width, color='r', label='Mean Fitness Impact')
    ax2.set_ylabel('Mean Fitness Impact')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('Best Genome: Mutation Count and Mean Fitness Impact by Mutation Type', fontsize=14)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{savedir}/best_mutation_impacts.pdf")

# ----- SPECIES PLOTS

def get_species_color_map(species_df: pd.DataFrame):
    """Generates a colormap to be used by stackplot and species tree visualization
    """
    all_species = species_df["name"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_species)))
    return {species: colors[i] for i, species in enumerate(all_species)}


def species_stackplot(species_df: pd.DataFrame, cmap: dict, savedir: str):
    """Stackplot of species member counts with a separate legend figure
    """
    filtered = species_df[species_df["obliterate"] == False]
    grouped = filtered.groupby(["gen", "name"])["num_members"].sum().unstack(fill_value=0)
    # Create main plot
    fig, ax = plt.subplots(figsize=(10, 5))
    # Use the provided color map in stackplot
    stack = ax.stackplot(grouped.index, grouped.T, colors=[cmap[species] for species in grouped.columns])
    ax.set_title("Species Sizes Over Time")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Number of Members")
    # Create separate legend figure
    figlegend = plt.figure(figsize=(10, 2))
    labels = [label.split("-")[0] for label in grouped.columns]
    # Use the provided color map for legend
    proxy_artists = [plt.Rectangle((0, 0), 1, 1, fc=cmap[species]) for species in grouped.columns]
    figlegend.legend(proxy_artists, labels, loc='center', ncol=5, frameon=False)
    # Save both figures
    fig.savefig(f"{savedir}/species_plot.pdf", bbox_inches='tight')
    figlegend.savefig(f"{savedir}/species_plot_legend.pdf", bbox_inches='tight')


def plot_species_evolution(
        species_df: pd.DataFrame,
        pop_df: pd.DataFrame,
        gen_info_df: pd.DataFrame,
        cmap: dict,
        savedir: str,
        maxwidth=200,
        figsize=(20, 12)
        ):
    """A tree visualization of which species branched from which, how many members
    they had, and when they had the best genome.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # create the datastructure species_tree needed for plotting
    species_tree = defaultdict(lambda: {"history": [], "forks": set(), "forked_from": None})
    for gen in species_df["gen"].unique():
        gen_species = species_df[species_df["gen"] == gen]
        # find the species of the best genome
        best_g_id = gen_info_df[gen_info_df["gen"] == gen]["best_genome"].iloc[0]
        best_g_species = pop_df[pop_df["id"] == best_g_id]["species_id"].iloc[0]
        # loop through every species and save it's history
        for _, species in gen_species.iterrows():
            species_id = species["name"]
            num_members = species["num_members"]
            # history contains generation, num_members and bool if it contains best_g
            species_tree[species_id]["history"].append((gen, num_members, species_id==best_g_species))
            if species["age"] == 1 and gen > 1:
                representative = pop_df[(pop_df["id"] == species["representative_id"]) & (pop_df["gen"] == gen)].iloc[0]
                parent_species = pop_df[(pop_df["id"] == representative["parent_id"]) & (pop_df["gen"] == gen - 1)].iloc[0]["species_id"]
                species_tree[species_id]["forked_from"] = parent_species
                species_tree[parent_species]["forks"].add(species["name"])
    species_tree = dict(species_tree)
    
    # get the total number of species and population size
    total_species = len(species_tree)
    popsize = len(pop_df[pop_df["gen"]==1])

    # Calculate species offsets, used for making nice tree structure vis
    def calculate_offsets(tree):
        offsets = {}
        current_offset = 1

        def dfs(species):
            nonlocal current_offset
            if species not in offsets:
                offsets[species] = current_offset
                current_offset += 1
            children = sorted([child for child, data in tree.items() if data.get('forked_from') == species])
            for child in children:
                dfs(child)

        # Find all root species (species with no 'forked_from')
        roots = [node for node, data in tree.items() if not data['forked_from']]
        roots.sort()
        # Perform DFS starting from each root
        for root in roots:
            dfs(root)
        return offsets

    offsets = calculate_offsets(species_tree)
    # Plot species lines
    legend_elements = []
    for species_id, data in species_tree.items():
        # in the case of species spawned in the last round, skip them to avoid bugs
        if len(data['history']) <= 1: continue
        segments = []
        widths = []
        y = offsets[species_id]
        for i in range(len(data['history']) - 1):
            x1, num_members, is_best = data['history'][i]
            x2, _, _ = data['history'][i + 1]
            segments.append([(x1, y), (x2, y)])
            widths.append(num_members)
            if is_best:
                ax.scatter(x1, y, color='gold', marker='d', s=100, edgecolor='black', linewidth=1, zorder=2)
        # add first branching segment
        if data['forked_from']:
            first_p = segments[0][0]
            first_seg = [(first_p[0]-1, offsets[data['forked_from']]), first_p]
            segments.insert(0, first_seg)
            widths.insert(0, maxwidth/(total_species*2))
        # Calculate widths, create LineCollection
        widths = np.array(widths) / popsize * maxwidth
        lc = LineCollection(segments, linewidths=widths, colors=cmap[species_id], zorder=1)
        ax.add_collection(lc)
        legend_elements.append(plt.Line2D([0], [0], color=cmap[species_id], lw=2, label=f'Species {species_id[:8]}...'))

    # add vertical lines for easier readability
    for x in range(0, int(ax.get_xlim()[1]), 50):
        ax.axvline(x=x, color='grey', linestyle='--', linewidth=1, zorder=0)

    # Set up plot along with labels
    ax.autoscale()
    ax.tick_params(axis='x', labelsize=18)
    ax.set_xlabel('Generation', fontsize=24)
    ax.set_ylabel('Species', fontsize=24)
    ax.set_title('Evolutionary Tree', fontsize=36)
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{savedir}/species_tree.pdf", bbox_inches='tight')
    # Separate plot for the legend
    figlegend, ax_legend = plt.subplots(figsize=(10, 1))
    ax_legend.axis('off')
    legend = ax_legend.legend(handles=legend_elements, loc='center', ncol=len(legend_elements)/2, fontsize=12)
    figlegend.tight_layout()
    figlegend.savefig(f"{savedir}/species_tree_legend.pdf", bbox_inches='tight')
    

def plot_avg_species_fit(species_df: pd.DataFrame, cmap: dict, savedir: str, use_adjusted=True):
    """Plots the average fitness of all species, defaults to adjusted fitness
    """
    plt.figure(figsize=FSIZE)
    for species_name in cmap:
        filtered_df = species_df[species_df["name"]==species_name]
        if use_adjusted:
            plt.plot(filtered_df["gen"], filtered_df["avg_fitness_adjusted"], color=cmap[species_name])
        else:
            plt.plot(filtered_df["gen"], filtered_df["avg_fitness"], color=cmap[species_name])

    plt.title(f"Average {'adjusted ' if use_adjusted else ''}fitness of species")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.savefig(f"{savedir}/avg_species_fitness.pdf")
