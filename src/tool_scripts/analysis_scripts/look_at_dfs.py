# %%
"""
Purpose of this script is to have a space where I can work with run data, and 
test implementing new plots before they are integrated into the endreports module
Nothing that is in here should not be implemented in endreports at some point,
except for the tests at the bottom of the file
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from statistics import fmean
from math import ceil
import pickle, gzip
from importlib import reload
import neatutils.endreports as er

def load_component_dict(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)

pop_df = pd.read_feather("E:/migrate_o/github_repos/EvolvePetriNets/results/results_used_in_analysis/compare_search_width/truncation/2_08-26-2024_14-32-50/data/population.feather")
gen_info_df = pd.read_feather("E:/migrate_o/github_repos/EvolvePetriNets/results/results_used_in_analysis/compare_search_width/truncation/2_08-26-2024_14-32-50/data/gen_info.feather")
species_df = pd.read_feather("E:/migrate_o/github_repos/EvolvePetriNets/results/data/issue_still_there_question_08-08-2024_15-48-51/whatever/2_08-08-2024_15-48-57/data/species.feather")

component_dict = load_component_dict("E:/migrate_o/github_repos/EvolvePetriNets/results/results_used_in_analysis/compare_search_width/truncation/2_08-26-2024_14-32-50/data/component_dict.pkl.gz")

savedir = "E:/migrate_o/github_repos/EvolvePetriNets/results/results_used_in_analysis/compare_search_width/truncation/2_08-26-2024_14-32-50/data"

FSIZE = (10, 5)
"""
################################################################################
Working with results data to test implementing endreports
################################################################################
"""
# %%
# lineage plot
# pop_df
gen_info_df


def best_genomes(gen_info_df, pop_df):
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


bg = best_genomes(gen_info_df, pop_df)


def plot_lineage(best_genomes_df, savedir=str):
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
    
    plt.legend(loc='bottom right', fontsize='medium')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title('Best Genome: Fitness Progression with Mutation Types', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{savedir}/best_mutation_lineage.pdf")

# Example usage:
plot_lineage(bg, savedir)

# %%

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

best_genome_mutation_analysis(bg, savedir)


# %%
# species plot
import matplotlib.pyplot as plt
import pandas as pd

# Group by gen and species_id, sum num_members, then do stackplot
grouped = species_df.groupby(['gen', 'name'])['num_members'].sum().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(15, 5))
ax.stackplot(grouped.index, grouped.T, labels=grouped.columns)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.title('Species Sizes Over Time')
plt.xlabel('Generation')
plt.ylabel('Number of Members')
plt.savefig(f"{savedir}/species_plot.pdf")

# %%
gen_info_df.columns
# %%
import matplotlib.pyplot as plt
import pandas as pd

def time_stackplot(gen_info_df, savedir: str) -> None:
    times_df = pd.DataFrame(gen_info_df["times"].tolist(), index=gen_info_df.index)
    plt.figure(figsize=FSIZE)
    plt.stackplot(
        times_df.index,
        times_df["evaluate_curr_generation"],
        times_df["pop_update"],
        labels=[
            "Evaluate Current Generation", "Population Update"
        ])
    plt.legend()
    plt.title('Time Spent Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Time')
    plt.savefig(f"{savedir}/time_plot.pdf")

# help(ax.stackplot)
time_stackplot(gen_info_df, savedir)

# %%
plt.close("all")

def fitness_plot(gen_info_df, savedir: str):
    """Plot
    """
    plt.figure(figsize=FSIZE)
    plt.plot(gen_info_df[["best_genome_fitness", "best_species_avg_fitness", "avg_pop_fitness"]])
    plt.legend(["Best Genome Fitness", "Best Species Average Fitness", "Average Population Fitness"])
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

fitness_plot(gen_info_df, savedir)

# %%
import matplotlib.pyplot as plt

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


metrics_plot(pop_df, savedir)

# %%

def species_plot(species_df: pd.DataFrame, savedir: str):
    """Stackplot of species member counts with a separate legend figure
    """
    grouped = species_df.groupby(["gen", "name"])["num_members"].sum().unstack(fill_value=0)
    # Create main plot
    fig, ax = plt.subplots(figsize=(10, 5))
    stack = ax.stackplot(grouped.index, grouped.T)
    ax.set_title("Species Sizes Over Time")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Number of Members")
    # Create separate legend figure
    figlegend = plt.figure(figsize=(10, 2))
    labels = [label.split("-")[0] for label in grouped.columns]
    # Create proxy artists for the legend
    proxy_artists = [plt.Rectangle((0, 0), 1, 1, fc=poly.get_facecolor()[0]) for poly in stack]
    figlegend.legend(proxy_artists, labels, loc='center', ncol=5, frameon=False)
    # Save both figures
    fig.savefig(f"{savedir}/species_plot.pdf", bbox_inches='tight')
    figlegend.savefig(f"{savedir}/species_plot_legend.pdf", bbox_inches='tight')

species_plot(species_df, savedir)

# %%
"""
################################################################################
Scattered analysis things for trying to figure why best genome fitness went down
################################################################################
"""

gen_info_df = pd.read_feather("E:/migrate_o/github_repos/EvolvePetriNets/results/data/testing_new_roulette_08-02-2024_11-56-43/whatever/1_08-02-2024_11-57-03/feather/gen_info.feather")
pop_df = pd.read_feather("E:/migrate_o/github_repos/EvolvePetriNets/results/data/testing_new_roulette_08-02-2024_11-56-43/whatever/1_08-02-2024_11-57-03/feather/population.feather")

savedir = "E:/migrate_o/github_repos/EvolvePetriNets/results/data/testing_new_roulette_08-02-2024_11-56-43/whatever/1_08-02-2024_11-57-03"
# %%
pop_df[pop_df["parent_id"]=="1e7ad65a-e429-4483-ae1b-73706ceb998d"]
pop_df[pop_df["id"]=="1e7ad65a-e429-4483-ae1b-73706ceb998d"]

# %%
from neatutils import endreports as er
reload(er)

er.mutation_effects_plot(pop_df, savedir)

# df_with_parents = pop_df.dropna(subset=['parent_id']).copy()
# fitness_dict = pop_df.set_index('id')['fitness'].to_dict()
# df_with_parents

# %%
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

# Assuming filtered is already defined
filtered = pop_df[pop_df["gen"] == 300]

components = filtered["my_components"]
species_ids = filtered["species_id"]

# Flatten the list of lists into a single list and keep track of species
all_components = []
all_species = []
for sublist, species in zip(components, species_ids):
    all_components.extend(sublist)
    all_species.extend([species] * len(sublist))

# Count the occurrences of each component
component_counts = Counter(all_components)

# Get the 10 most common components
most_common_components = [component for component, count in component_counts.most_common(10)]

# Filter out the 10 most common components
filtered_components = [component for component in all_components if component not in most_common_components]
filtered_species = [species for component, species in zip(all_components, all_species) if component not in most_common_components]

# Create a DataFrame for plotting
plot_df = pd.DataFrame({
    'Component': filtered_components,
    'Species': filtered_species
})

# Plot the overlayed histogram using seaborn
plt.figure(figsize=(10, 6))
# sns.histplot(data=plot_df, x='Component', hue='Species', multiple='layer', bins=100, palette='tab10')
sns.histplot(data=plot_df, x='Component', hue='Species', multiple='stack', bins=100, palette='tab10')
plt.xlabel('')
plt.xticks([])
plt.title("Components by Species histogram")
plt.show()

# %%
# Group by generation and count unique components
unique_components_per_gen = pop_df.groupby('gen')['my_components'].apply(lambda x: len(set(component for sublist in x for component in sublist)))

# Create a DataFrame for plotting
plot_df = pd.DataFrame({
    'Generation': unique_components_per_gen.index,
    'UniqueComponents': unique_components_per_gen.values
})

# Plot the line plot using seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(data=plot_df, x='Generation', y='UniqueComponents')
plt.xlabel('Generation')
plt.ylabel('Number of Unique Components')
plt.title('Number of Unique Components per Generation')
plt.show()

gen_info_df.columns

# Calculate the difference between the best fitness of the current and previous generation
fitness_diff = gen_info_df["best_genome_fitness"].diff()

# Find the generations where the best fitness is lower than that of the previous generation
gens_with_lower_fitness = gen_info_df.index[fitness_diff < 0]

print(gens_with_lower_fitness)
# %%
gen_info_df.columns
changed = [(i, i+1) for i in gens_with_lower_fitness]

for change, after in changed:
    print()
    print(change, after)
    print("changed", gen_info_df[gen_info_df["gen"]==change]["best_species"].iloc[0])
    print("after", gen_info_df[gen_info_df["gen"]==after]["best_species"].iloc[0])
    print("prev best")
    print(gen_info_df[gen_info_df["gen"]==change]["best_genome_fitness"].iloc[0])
    print(pop_df[pop_df["gen"]==change]["fitness"].max())
    print("new best")
    print(gen_info_df[gen_info_df["gen"]==after]["best_genome_fitness"].iloc[0])
    print(pop_df[pop_df["gen"]==after]["fitness"].max())

    best_id1 = gen_info_df[gen_info_df["gen"]==change]["best_genome"].iloc[0]
    best_id2 = gen_info_df[gen_info_df["gen"]==after]["best_genome"].iloc[0]
    print("prev leader species")
    print(pop_df[pop_df["id"]==best_id1]["species_id"].iloc[0])
    print("curr leader species")
    print(pop_df[pop_df["id"]==best_id2]["species_id"].iloc[0])
    prev_best = species_df[(species_df["gen"] == change) & (species_df["name"] == pop_df[pop_df["id"]==best_id1]["species_id"].iloc[0])]
    old_best = species_df[(species_df["gen"] == after) & (species_df["name"] == pop_df[pop_df["id"]==best_id1]["species_id"].iloc[0])]
    print(pop_df[pop_df["id"]==old_best["leader_id"].iloc[0]]["fitness"].iloc[0])
    print("old best ever fit")
    print(old_best["best_ever_fitness"].iloc[0])
    print("old best num members")
    print(old_best["num_members"].iloc[0])
    # print(old_best["curr_mutation_rate"].iloc[0])
    # print(old_best["num_gens_no_improvement"].iloc[0])
    all_of_old_spec = pop_df[(pop_df["gen"]==after) & (pop_df["species_id"]==old_best["name"].iloc[0])]
    print(all_of_old_spec["my_mutations"].value_counts())
    print(prev_best["num_to_spawn"].iloc[0])

    print(80*"/")
# %%
# species_df.columns
species_df["num_to_spawn"].value_counts()

# %%
pop_df = pd.read_feather("E:/migrate_o/github_repos/EvolvePetriNets/results/results_used_in_analysis/why_fit_decline/data/population.feather")
gen_info_df = pd.read_feather("E:/migrate_o/github_repos/EvolvePetriNets/results/results_used_in_analysis/why_fit_decline/data/gen_info.feather")

bg = er.filter_best_genomes(gen_info_df, pop_df)
bg1 = bg[bg["gen"] == 69]

bg_id2 = gen_info_df[gen_info_df["gen"] == 70]["best_genome"].iloc[0]
bg2 = pop_df[pop_df["id"] == bg_id2]

bg1
# bg2
# %%
pop_df[pop_df["parent_id"] == bg1["id"].iloc[0]]
bg1_clone = pop_df[pop_df["id"] == "1438793c-4de0-4f04-951e-45bd473cf607"]

bg1_clone["my_components"].iloc[0] == bg1["my_components"].iloc[0]
bg1_clone

# %%
bg1_parent = pop_df[pop_df["id"] == "02a44ab7-f1b7-4680-948c-d542e8fc452f"]
bg1_parent
# %%
# reconstructing the genome
conns =  [
    ["start", "register request"], ["start", "reject request"], ["register request", "4"],
    ["register request", "1"], ["register request", "3"], ["register request", "5"],
    ["1", "reject request"], ["1", "examine thoroughly"], ["1", "examine casually"],
    ["examine casually", "2"], ["examine casually", "3"], ["2", "reinitiate request"],
    ["3", "check ticket"], ["examine thoroughly", "4"], ["reinitiate request", "4"],
    ["reinitiate request", "5"], ["4", "check ticket"], ["5", "decide"],
    ["check ticket", "6"], ["check ticket", "7"], ["check ticket", "8"],
    ["6", "reject request"], ["6", "decide"], ["7", "decide"], ["8", "decide"],
    ["decide", "1"], ["reject request", "end"],
]
from neat import genome
reload(genome)
from neat.initial_population import get_log_footprints
from tool_scripts.useful_functions import \
    load_genome, show_genome, get_aligned_traces, get_log_variants, log, reset_ga, \
    eval_and_print_metrics

tl = list(log["footprints"]["activities"])

pids = ["1", "2", "3", "4", "5", "6", "7", "8"]
places = {i: genome.GPlace(i) for i in pids}


rg = genome.GeneticNet({}, places, {}, "", tl)
for conn in conns:
    if conn[0] in pids:
        rg.place_trans_arc(conn[0], conn[1])
    else:
        rg.trans_place_arc(conn[0], conn[1])

rg.evaluate_fitness(log)

# %%
####### WORKING DIRECTLY WITH PICKLED BEST GENOMES ##########
from pprint import pprint
reload(genome)

pop_df = pd.read_feather("E:/migrate_o/github_repos/EvolvePetriNets/results/results_used_in_analysis/why_fit_decline_allpickled/data/population.feather")
gen_info_df = pd.read_feather("E:/migrate_o/github_repos/EvolvePetriNets/results/results_used_in_analysis/why_fit_decline_allpickled/data/gen_info.feather")

bg = load_genome("E:/migrate_o/github_repos/EvolvePetriNets/results/results_used_in_analysis/why_fit_decline_allpickled/improvements/improvement_gen-204.pkl")
bg.get_gviz()

bg_kid = pop_df[pop_df["parent_id"] == bg.id]
pop_df[pop_df["id"] == bg.id]

# bg.evaluate_fitness(log)["metrics"]
bg.evaluate_fitness(log)

bg_clone = bg.clone()
bg_clone.evaluate_fitness(log)

print("bg")
print(bg.fitness)
pprint(bg.fitness_metrics)
pprint(bg.transitions)

print("clone")
print(bg_clone.fitness)
pprint(bg_clone.fitness_metrics)
pprint(bg_clone.transitions)

# bg.transitions == bg_clone.transitions
# %%

# for a in bg_clone.arcs.values():
#     if "pay compensation" in [a.target_id, a.source_id]:
#         print(a)
    
bg.get_connected() == bg_clone.get_connected()

bg.build_fc_petri(log).transitions
print()
bg_clone.build_fc_petri(log).transitions
bg_clone.get_connected()