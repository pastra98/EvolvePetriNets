# %% [markdown]
# # todos
# 

# ## general todos
"""
[ ] eventually fix the paths for anyone not running this from src directory (remove ...)
[x] map setup numbers to better names for all plots
[ ] traverse setup folders and concat dfs
"""

# ## Plots I want this to work on
"""
[ ] Time plot?
[ ] Fitness_plot
[ ] Total components?
[ ] Unique components per generation
[ ] Mutation boxplot effects
[ ] fitness Metrics plots
[ ] Mutation lineplots
[ ] Average metrics?
[ ] Number species
[ ] Spawn rank histogram
[ ] best genome mutation analysis
[x] Scatterplot
"""

# ## Do not forget about single-run plots, they should also be in here
"""
[ ] best genome lineage
[ ] species leaders??
[ ] species stackplot
[ ] species tree
[ ] species ridgeline
[ ] species average fitnesses
"""

# ## Other analysis implemented someplace else
"""
[ ] ... check the onenote
"""

# ## not implemented yet
"""
[ ] ... check the onenote
"""

# ## Those can only be copied from a run
"""
[ ] improvements (combine them into something)
[ ] species leaders??
"""

# %% [markdown]
# # Analysis plots
# very wip

# %%
import polars as pl
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


# %% [markdown]
# ## Fitness ~ Components Scatter

# %%

def get_mapped_setupname_df(df, setup_map):
    renamed_df = df.copy()
    renamed_df["setupname"] = renamed_df["setupname"].apply(
        lambda x: setup_map.get(int(x.split('_')[1]), x)
        )
    return renamed_df

def create_scatter_plot(df, setup_map=None):

    # If a setup_map is provided, rename the setupname column values
    if setup_map:
        df = get_mapped_setupname_df(df, setup_map)

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    # Get unique setupnames for colors
    setupnames = df['setupname'].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(setupnames)))
    # Plot each setupname with a different color
    for setupname, color in zip(setupnames, colors):
        mask = df['setupname'] == setupname
        setup_data = df.loc[mask]
        
        # Scatter plot for individual points
        ax.scatter(setup_data['num_components'], 
                   setup_data['max_fitness'],
                   c=[color], 
                   label=setupname,
                   alpha=0.7)
        # Calculate and plot mean without adding to legend
        mean_components = setup_data['num_components'].mean()
        mean_fitness = setup_data['max_fitness'].mean()
        ax.scatter(mean_components, mean_fitness, 
                   c=[color], marker='X', s=200, edgecolors='black', linewidth=2)
    # Set labels and title
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Max Fitness')
    ax.set_title('Max Fitness vs Number of Components (with Setup Means)')
    # Add legend
    ax.legend()
    # Add a text annotation explaining the 'X' markers
    ax.text(0.95, 0.05, "'X' markers represent setup means", 
            transform=ax.transAxes, ha='right', va='bottom', 
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
    # Show the plot
    plt.tight_layout()
    plt.show()


df = pd.read_feather("../analysis/data/setup_reports_test/data/final_report_df.feather")
setup_map={1: "speciation", 2: "roulette", 3: "truncation"}
create_scatter_plot(df, setup_map)

# %%
# total components


# %%
# unique components


# %%
# # Plotting the mean fitness diff
import matplotlib.pyplot as plt

# Group by generation and calculate the mean fitness difference
def plot_mean_fitness_diff(pop_df: pl.DataFrame):
    # calculate the mean fitness difference
    avg_mean_diff = pop_df.filter(
        pl.col("gen") > 1
        ).group_by("gen").agg(
            pl.col("fitness_difference").mean().alias("mean_fitness_difference")
            ).sort("gen")
    # Plot the mean fitness difference over generations
    plt.figure(figsize=(10, 6))
    plt.plot(avg_mean_diff["gen"], avg_mean_diff["mean_fitness_difference"])
    plt.xlabel("Generation")
    plt.ylabel("Mean Fitness Difference")
    plt.title("Mean Fitness Difference Over Generations")
    plt.grid(True)
    plt.show()

# TODO: make this work with multiple runs

plot_mean_fitness_diff(pop_df)