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
[x] Fitness_plot
    [ ] comparing best vs. avg vs best species (needs to be only done once)
[x] Total components?
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

# ## Interesting one-off analysis
"""
[ ] Mutation lineage for a selected run
[ ] Component similarity drift
[ ] Fitness of components plotting
        - Look at the components in the fittest genome at the end, plot their t-values
        obtained from the component dict over the generations
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
from importlib import reload
import neatutils.endreports as er
import scripts.helper_scripts.setup_analysis as sa


# %% [markdown]
# ## Fitness ~ Components Scatter

summary_df = pl.read_ipc("../analysis/data/setup_reports_test/execution_data/final_report_df.feather")
sa.components_fitness_scatter(summary_df)

# %%
reload(sa)



res = sa.exec_results_crawler("../analysis/data/testing_truncation")

search = {
    "spawn_cutoff_10%": {"spawn_cutoff": 0.1},
    "spawn_cutoff_25%": {"spawn_cutoff": 0.25},
    "spawn_cutoff_50%": {"spawn_cutoff": 0.50},
    "spawn_cutoff_75%": {"spawn_cutoff": 0.75},
}


plt_layout = [["spawn_cutoff_10%", "spawn_cutoff_25%", "spawn_cutoff_50%", "spawn_cutoff_75%"]]
data_sources = sa.search_and_aggregate_param_results(res, search)

sa.generalized_lineplot(plt_layout, data_sources, "num_total_components")
sa.generalized_lineplot(plt_layout, data_sources, "best_genome_fitness")
sa.generalized_lineplot(plt_layout, data_sources, "avg_pop_fitness")
sa.generalized_lineplot(plt_layout, data_sources, "num_unique_components")

sa.generalized_barplot(plt_layout, data_sources, "num_total_components")

print(res["setups"][1]["gen_info_agg"].columns)



# %%

# %%
# unique components