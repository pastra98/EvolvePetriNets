# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from statistics import fmean
from math import ceil
import pickle, gzip

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

def create_species_tree(species_df, pop_df, gen_info_df):
    species_tree = defaultdict(lambda: {"history": [], "forks": set(), "forked_from": None})
    
    for gen in species_df["gen"].unique():
        gen_species = species_df[species_df["gen"] == gen]
        # find the species of the best genome
        best_g_id = gen_info_df[gen_info_df["gen"] == gen]["best_genome"].iloc[0]
        best_g_species = pop_df[pop_df["id"] == best_g_id]["species_id"].iloc[0]
        
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
    
    return dict(species_tree)


def calculate_total_forks(species_id, species_tree):
    total = len(species_tree[species_id]['forks'])
    for fork in species_tree[species_id]['forks']:
        total += calculate_total_forks(fork, species_tree)
    return total

def plot_species_evolution(
        species_tree,
        maxwidth=200,
        popsize=500,
        figsize=(20, 12)
        ):
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate the total number of species and maximum generation
    total_species = len(species_tree)
    
    # calculate species offsets
    def calculate_offsets(tree):
        offsets = {}
        current_offset = 1

        def dfs(species):
            nonlocal current_offset
            if species not in offsets:
                offsets[species] = current_offset
                current_offset += 1
            
            children = [child for child, data in tree.items() if data.get('forked_from') == species]
            for child in sorted(children):
                dfs(child)

        # Find the root species
        root = next(node for node, data in species_tree.items() if not data['forked_from'])
        dfs(root)

        return offsets

    offsets = calculate_offsets(species_tree)

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
        
        # Calculate widths
        widths = np.array(widths) / popsize * maxwidth

        # Create LineCollection
        lc = LineCollection(segments, linewidths=widths, colors=color_map[species_id], zorder=1)
        ax.add_collection(lc)
        legend_elements.append(plt.Line2D([0], [0], color=color_map[species_id], lw=2, label=f'Species {species_id[:8]}...'))

    # add vertical lines
    for x in range(0, int(ax.get_xlim()[1]), 50):
        ax.axvline(x=x, color='grey', linestyle='--', linewidth=1, zorder=0)

    # Set up plot along with labels
    ax.autoscale()
    ax.set_xlabel('Generation', fontsize=24)
    ax.set_ylabel('Species', fontsize=24)
    ax.set_title('Evolutionary Tree', fontsize=36)
    ax.set_yticks([])
    # ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    plt.tight_layout()
    plt.show()

    # Separate plot for the legend
    fig_legend, ax_legend = plt.subplots(figsize=(10, 1))
    ax_legend.axis('off')
    legend = ax_legend.legend(handles=legend_elements, loc='center', ncol=len(legend_elements)/2, fontsize=12)
    fig_legend.tight_layout()
    plt.show()
    

# This should also be wrapped in one function
species_tree = create_species_tree(species_df, pop_df, gen_info_df)
plot_species_evolution(species_tree)
