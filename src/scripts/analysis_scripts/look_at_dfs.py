# %%
"""
Purpose of this script is to have a space where I can work with run data, and 
test implementing new plots before they are integrated into the endreports module
Eventually this file should not mirror anything that is already implemented in
endreports, i.e. it serves as temporary testing ground and holds unused/abandoned
visualizations
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

data_fp = "C:/Users/pauls/Documents/GitHubRepos/EvolvePetriNets/results/data/avg_fit_plot_new_pop_df_10-08-2024_14-07-58/whatever/4_10-08-2024_14-08-02/data"
# data_fp = "C:/Users/pauls/Documents/GitHubRepos/EvolvePetriNets/results/data/intra_s_truncation_s_cutoff_005_10-08-2024_13-36-59/whatever/4_10-08-2024_13-37-05/data"

gen_info_df = pd.read_feather(data_fp + "/gen_info.feather")
pop_df = pd.read_feather(data_fp + "/population.feather")
species_df = pd.read_feather(data_fp + "/species.feather")
component_dict = load_component_dict(data_fp + "/component_dict.pkl.gz")

savedir = data_fp

FSIZE = (10, 5)

# %%
"""
################################################################################
####################### NOT IMPLEMENTED IN ENDREPORTS ##########################
################################################################################
--------------------------------------------------------------------------------
--- Stackplot that shows all components and to which species they belong
--- not in use because the ridgeline plot works better
--------------------------------------------------------------------------------
"""
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
"""
--------------------------------------------------------------------------------
--- Spawn-rank histogram
--------------------------------------------------------------------------------
"""

def analyze_offspring_distribution(pop_df, popsize=500):
    # Initialize rank_spawns dictionary
    max_gen = pop_df['gen'].max()

    # this does not work for species, because they might not exist in a given generation

    rank_spawns = {i: 0 for i in range(1, popsize+1)}
    
    for gen in range(2, max_gen + 1):
        # Get previous generation's genomes
        prev_gen = pop_df[pop_df['gen'] == gen - 1]
        
        # Sort previous generation by fitness and create rank dictionary
        prev_gen_sorted = prev_gen.sort_values('fitness', ascending=False)
        previous_parents = dict(zip(prev_gen_sorted['id'], range(1, len(prev_gen_sorted) + 1)))
        
        # Get current generation
        curr_gen = pop_df[pop_df['gen'] == gen]
        
        # Count offspring for each parent
        offspring_counts = curr_gen['parent_id'].value_counts()
        
        # Map parent ranks to offspring counts
        for parent_id, count in offspring_counts.items():
            if parent_id in previous_parents:
                rank = previous_parents[parent_id]
                if rank <= 500:
                    rank_spawns[rank] += count
    
    return rank_spawns

def plot_offspring_distribution(rank_spawns):
    ranks = list(rank_spawns.keys())
    spawn_counts = list(rank_spawns.values())
    
    plt.figure(figsize=(12, 6))
    plt.bar(ranks, spawn_counts, width=1)
    plt.xlabel('Fitness Rank of Parent')
    plt.ylabel('Number of Offspring')
    plt.title('Distribution of Offspring by Parent Fitness Rank')
    plt.xlim(0, 500)
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    # plt.tight_layout()
    plt.show()

# Assuming pop_df is already loaded
print("roulette")
data_fp = "C:/Users/pauls/Documents/GitHubRepos/EvolvePetriNets/results/data/roulette_selection_10-08-2024_15-46-36/whatever/4_10-08-2024_15-46-42/data"
pop_df = pd.read_feather(data_fp + "/population.feather")
rank_spawns = analyze_offspring_distribution(pop_df)
plot_offspring_distribution(rank_spawns)

print("truncation")
data_fp = "C:/Users/pauls/Documents/GitHubRepos/EvolvePetriNets/results/data/truncation_10-08-2024_15-57-16/whatever/4_10-08-2024_15-57-22/data"
pop_df = pd.read_feather(data_fp + "/population.feather")
rank_spawns = analyze_offspring_distribution(pop_df)
plot_offspring_distribution(rank_spawns)

print("speciation")
data_fp = "C:/Users/pauls/Documents/GitHubRepos/EvolvePetriNets/results/data/16core_tokenreplay_OOP_10-06-2024_17-09-58/whatever/6_10-06-2024_17-10-02/data"
pop_df = pd.read_feather(data_fp + "/population.feather")
rank_spawns = analyze_offspring_distribution(pop_df)
plot_offspring_distribution(rank_spawns)



for s_id in pop_df["species_id"].unique():
    if type(s_id) == str:
        print(s_id)
        rank_spawns = analyze_offspring_distribution(pop_df[pop_df["species_id"]==s_id])
        plot_offspring_distribution(rank_spawns)

# %%
"""
--------------------------------------------------------------------------------
--- Scatterplot of fitnes of entire population
--------------------------------------------------------------------------------
"""
def show_fitness_scatter(pop_df: pd.DataFrame, title="", color_crossover=True):
    if color_crossover:
        # Create a color column based on the condition
        colors = ['red' if "crossover" in mutation else 'grey' for mutation in pop_df["my_mutations"]]
        plt.scatter(pop_df["gen"], pop_df["fitness"], s=10, color=colors, alpha=0.5)
    else:
        plt.scatter(pop_df["gen"], pop_df["fitness"], s=10, color='grey', alpha=0.5)

    plt.title(f'Fitness vs Generation: {title}')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()

s_id = pop_df["species_id"].unique()[0]
df_f = pop_df[(pop_df["species_id"]==s_id) & (pop_df["fitness"]>0)]
show_fitness_scatter(df_f, title=f"Species {s_id[:8]}")

# roulette
data_fp = "C:/Users/pauls/Documents/GitHubRepos/EvolvePetriNets/results/data/roulette_selection_10-08-2024_15-46-36/whatever/3_10-08-2024_15-46-42/data"
pop_df = pd.read_feather(data_fp + "/population.feather")
plt.scatter(pop_df["gen"], pop_df["fitness"], s=10, color='grey', alpha=0.5)
show_fitness_scatter(df_f, title="roulette")

# truncation
data_fp = "C:/Users/pauls/Documents/GitHubRepos/EvolvePetriNets/results/data/truncation_10-08-2024_15-57-16/whatever/4_10-08-2024_15-57-22/data"
pop_df = pd.read_feather(data_fp + "/population.feather")
plt.scatter(pop_df["gen"], pop_df["fitness"], s=10, color='grey', alpha=0.5)
show_fitness_scatter(df_f, title="truncation")

# speciation
data_fp = "C:/Users/pauls/Documents/GitHubRepos/EvolvePetriNets/results/data/speciation_10-08-2024_16-08-07/whatever/4_10-08-2024_16-08-14/data"
pop_df = pd.read_feather(data_fp + "/population.feather")
plt.scatter(pop_df["gen"], pop_df["fitness"], s=10, color='grey', alpha=0.5)
show_fitness_scatter(df_f, title="speciation")

# %%
"""
################################################################################
####################### TEMPORARY TEST STUFF ###################################
################################################################################
--------------------------------------------------------------------------------
--- Why does the speciation best species avg fitness fluctuate so wildly & also
--- affect the overall population fitness?
--------------------------------------------------------------------------------
"""

gen_info_df

decreases = []
for i in range(1, len(gen_info_df)):
    current_fitness = gen_info_df.loc[i, 'best_species_avg_fitness']
    previous_fitness = gen_info_df.loc[i - 1, 'best_species_avg_fitness']
    
    if current_fitness < previous_fitness:
        delta = previous_fitness - current_fitness
        generation = gen_info_df.loc[i, 'gen']
        decreases.append((generation, delta))

# Convert the list to a DataFrame
decreases_df = pd.DataFrame(decreases, columns=['gen', 'delta'])

# Sort the DataFrame by delta in descending order
sorted_decreases_df = decreases_df.sort_values(by='delta', ascending=False)

print(sorted_decreases_df)
# %%
def show_species_piechart(species_df, gen):
    filtered = species_df[species_df["gen"]==gen][["name", "num_members"]]
    plt.figure(figsize=(10, 7))
    plt.pie(filtered["num_members"], labels=filtered["name"].str[:8], autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Species Members for Generation {gen}')
    plt.axis('equal')
    plt.show()

show_species_piechart(species_df, 182)

# %%
"""
--------------------------------------------------------------------------------
--- Looking at the results summary df
--------------------------------------------------------------------------------
"""
import pandas as pd

summary_df_fp = "C:/Users/pauls/Documents/GitHubRepos/EvolvePetriNets/results/data/superfuckingbig_10-16-2024_21-03-49/final_report_df.feather"
# summary_df_fp = "C:/Users/pauls/Documents/GitHubRepos/EvolvePetriNets/results/data/test_import_params_10-16-2024_19-55-43/final_report_df.feather"
summary_df = pd.read_feather(summary_df_fp)

summary_df["max_fitness"].plot(kind="hist", bins=50)
