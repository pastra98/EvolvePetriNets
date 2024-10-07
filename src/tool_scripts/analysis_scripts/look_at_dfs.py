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

data_fp = "C:/Users/pauls/Documents/GitHubRepos/EvolvePetriNets/results/data/16core_tokenreplay_OOP_10-06-2024_17-09-58/whatever/5_10-06-2024_17-10-02/data"
# data_fp = "E:/migrate_o/github_repos/EvolvePetriNets/results/data/test_data_speciation_09-30-2024_19-45-31/whatever/1_09-30-2024_19-45-39/data"
# data_fp = "E:/migrate_o/github_repos/EvolvePetriNets/results/data/test_data_speciation_09-30-2024_19-45-31/whatever/4_09-30-2024_19-45-39/data"
# data_fp = "E:/migrate_o/github_repos/EvolvePetriNets/results/data/merge_worked_10-02-2024_13-28-17/whatever/4_10-02-2024_13-28-26/data"

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
