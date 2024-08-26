import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from statistics import fmean
from math import ceil
from neat.genome import GeneticNet
import json

import gc
import pickle
import gzip
import os

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

    use_species = "species" in full_history[1]
    if use_species:
        species_df = get_species_df(full_history)
        species_df.to_feather(f"{savedir}/data/species.feather")
        save_species_leaders(ga_info["species_leaders"], savedir)
        species_plot(species_df, savedir)

    pop_df = get_population_df(full_history)
    pop_df.to_feather(f"{savedir}/data/population.feather")
    gen_info_df = get_gen_info_df(full_history)
    gen_info_df.to_feather(f"{savedir}/data/gen_info.feather")
    

    # save the improvements, species leaders & best genome, delete ga info after
    save_improvements(ga_info["improvements"], savedir)
    save_genome_gviz(ga_info["best_genome"], savedir, name_prefix="best_genome")
    pickle_genome(ga_info["best_genome"], "best_genome", savedir)

    # make the plots
    time_stackplot(gen_info_df, savedir)
    fitness_plot(gen_info_df, use_species, savedir)
    components_plot(gen_info_df, savedir)
    mutation_effects_plot(pop_df, savedir)
    metrics_plot(pop_df, savedir)

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
        for key in excludes:
            info_d.pop(key, None)
        l.append(info_d | {"gen": gen})
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


def species_plot(species_df: pd.DataFrame, savedir: str):
    """Stackplot of species member counts with a separate legend figure
    """
    filtered = species_df[species_df["obliterate"] == False]
    grouped = filtered.groupby(["gen", "name"])["num_members"].sum().unstack(fill_value=0)
    # Create main plot
    fig, ax = plt.subplots(figsize=(10, 5))
    stack = ax.stackplot(grouped.index, grouped.T)
    ax.set_title("Species Sizes Over Time")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Number of Members")
    # Create separate legend figure
    figlegend = plt.figure(figsize=(10, 2))
    labels = [label.split("-")[0] for label in grouped.columns]
    proxy_artists = [plt.Rectangle((0, 0), 1, 1, fc=poly.get_facecolor()[0]) for poly in stack]
    figlegend.legend(proxy_artists, labels, loc='center', ncol=5, frameon=False)
    # Save both figures
    fig.savefig(f"{savedir}/species_plot.pdf", bbox_inches='tight')
    figlegend.savefig(f"{savedir}/species_plot_legend.pdf", bbox_inches='tight')


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


def components_plot(gen_info_df: pd.DataFrame, savedir: str):
    """Plot the total number of components over time
    """
    plt.figure(figsize=FSIZE)
    plt.plot(gen_info_df["num_total_components"])
    plt.title("Total components")
    plt.xlabel("Generation")
    plt.ylabel("num components")
    plt.savefig(f"{savedir}/components_plot.pdf")


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


def mutation_effects_plot(pop_df, savedir: str):
    df_with_parents = pop_df[pop_df["gen"] > 1]
    fitness_dict = pop_df.set_index('id')['fitness'].to_dict()
    df_with_parents.loc[:, 'fitness_difference'] = df_with_parents.apply(lambda row: row['fitness'] - fitness_dict[row['parent_id']], axis=1)
    mutation_effects = {}
    mutation_frequency = {}

    for _, row in df_with_parents.iterrows():
        mutations = row['my_mutations']
        fitness_diff = row['fitness_difference']
        for mutation in mutations:
            if mutation not in mutation_effects:
                mutation_effects[mutation] = []
            mutation_effects[mutation].append(fitness_diff)
            mutation_frequency[mutation] = mutation_frequency.get(mutation, 0) + 1

    mutation_effects = {m: mutation_effects[m] for m in sorted(mutation_effects)}
    total_mutations = sum(mutation_frequency.values())
    relative_frequencies = {mutation: count / total_mutations for mutation, count in mutation_frequency.items()}

    mutations = mutation_effects.keys()
    data_for_boxplot = mutation_effects.values()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Mutation')
    ax1.set_ylabel('Fitness Impact Distribution', color=color)
    ax1.axhline(y=0, color='lightgray', linestyle='--')

    bp = ax1.boxplot(data_for_boxplot, patch_artist=True, meanline=True, showmeans=True)

    for box in bp['boxes']:
        box.set(color=color, linewidth=2)
        box.set(facecolor='lightblue')

    ax1.set_xticks(np.arange(1, len(mutations) + 1))
    ax1.set_xticklabels(mutations, rotation=45, ha='right')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Relative Frequency', color=color)
    ax2.plot(np.arange(1, len(mutations) + 1), [relative_frequencies[mutation] for mutation in mutations], color=color, marker='o', linestyle='-')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Fitness Impact Distribution and Relative Frequency of Each Mutation')
    fig.tight_layout()

    # save a dataframe as well
    summary_df = pd.DataFrame(columns=['Min', '25%', 'Median', '75%', 'Max', 'Mean', 'Frequency'])
    for mutation, effects in mutation_effects.items():
        df_temp = pd.DataFrame(effects, columns=['Effects'])
        summary = df_temp['Effects'].describe(percentiles=[.25, .5, .75])
        summary_df.loc[mutation] = [
            summary['min'], summary['25%'], summary['50%'], summary['75%'], summary['max'],
            summary['mean'], summary['count']
        ]

    fig.savefig(f"{savedir}/mutation_analysis.pdf", dpi=300)
    summary_df.to_markdown(f"{savedir}/mutation_effects.txt")


def best_genome_lineage(best_genomes_df: pd.DataFrame, savedir=str):
    """Scatterplot of the mutation history of the best genome
    """
    gens = best_genomes_df['gen']
    fitnesses = best_genomes_df['fitness']
    mutations = best_genomes_df['my_mutations'].apply(lambda x: x[0])

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
    
    plt.legend(loc='upper left', fontsize='medium')
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
    mutations = best_genomes_df['my_mutations'].apply(lambda x: x[0])
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