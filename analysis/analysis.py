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
import matplotlib.pyplot as plt
from importlib import reload
import neatutils.endreports as er

import scripts.helper_scripts.setup_analysis as sa


# %% [markdown]
# ## Fitness ~ Components Scatter

# %%
# testing setup analysis
# fitness & components scatterplot

summary_df = pl.read_ipc("../analysis/data/setup_reports_test/execution_data/final_report_df.feather")
sa.components_fitness_scatter(summary_df)

# %%

res = sa.exec_results_crawler("../analysis/data/testing_truncation")

# TODO: generalized plotting func
search = {
    "spawn_cutoff_10%": {"spawn_cutoff": 0.1},
    "spawn_cutoff_25%": {"spawn_cutoff": 0.25},
    "spawn_cutoff_50%": {"spawn_cutoff": 0.50},
    "spawn_cutoff_75%": {"spawn_cutoff": 0.75},
}

# sa.search_and_aggregate_param_results(res, {"all": {}}) # this would aggregate all dataframes

plt_layout = [["spawn_cutoff_10%", "spawn_cutoff_25%", "spawn_cutoff_50%", "spawn_cutoff_75%"]]

data_sources = sa.search_and_aggregate_param_results(res, search)

# y_ax = "num_total_components"
# optional
# x_ax = "gen"
# title = "Title"
# subplt_titles = []

sa.plot_data(plt_layout, data_sources, "num_total_components")
sa.plot_data(plt_layout, data_sources, "best_genome_fitness")
sa.plot_data(plt_layout, data_sources, "best_genome_fitness")

print(res["setups"][1]["gen_info_agg"].columns)



# %%

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
test_pop_df = pl.read_ipc("../analysis/data/testing_truncation/execution_data/setup_1/1_10-21-2024_21-17-27/data/population.feather")
plot_mean_fitness_diff(test_pop_df)