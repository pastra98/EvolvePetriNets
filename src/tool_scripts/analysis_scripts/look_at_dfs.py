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

# data_fp = "E:/migrate_o/github_repos/EvolvePetriNets/results/data/test_data_speciation_09-30-2024_19-45-31/whatever/2_09-30-2024_19-45-39/data"
# data_fp = "E:/migrate_o/github_repos/EvolvePetriNets/results/data/test_data_speciation_09-30-2024_19-45-31/whatever/1_09-30-2024_19-45-39/data"
data_fp = "E:/migrate_o/github_repos/EvolvePetriNets/results/data/test_data_speciation_09-30-2024_19-45-31/whatever/4_09-30-2024_19-45-39/data"

gen_info_df = pd.read_feather(data_fp + "/gen_info.feather")
pop_df = pd.read_feather(data_fp + "/population.feather")
species_df = pd.read_feather(data_fp + "/species.feather")
component_dict = load_component_dict(data_fp + "/component_dict.pkl.gz")

savedir = data_fp

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

# %%
################################################################################
#################### SPECIES TREE ##############################################
################################################################################
# gen_info_df.columns
# pop_df.columns
species_df

# %%
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def plot_species_evolution(
        species_df: pd.DataFrame,
        pop_df: pd.DataFrame,
        gen_info_df: pd.DataFrame,
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
    offsets = {}
    current_offset = 1
    def dfs(species):
        nonlocal current_offset
        if species not in offsets:
            offsets[species] = current_offset
            current_offset += 1
        children = [child for child, data in species_tree.items() if data.get('forked_from') == species]
        for child in sorted(children):
            dfs(child)
    # Find the root species and start DFS
    root = next(node for node, data in species_tree.items() if not data['forked_from'])
    dfs(root)

    # Generate a unique color for each species
    colors = plt.cm.rainbow(np.linspace(0, 1, total_species))
    color_map = {species: colors[i] for i, species in enumerate(offsets.keys())}

    # Plot species lines
    legend_elements = []
    for species_id, data in species_tree.items():
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
        lc = LineCollection(segments, linewidths=widths, colors=color_map[species_id], zorder=1)
        ax.add_collection(lc)
        legend_elements.append(plt.Line2D([0], [0], color=color_map[species_id], lw=2, label=f'Species {species_id[:8]}...'))

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
    

plot_species_evolution(species_df, pop_df, gen_info_df, savedir=savedir)
