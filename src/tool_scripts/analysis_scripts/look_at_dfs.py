# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from statistics import fmean
from math import ceil

gen_info_df = pd.read_feather("E:/migrate_o/github_repos/EvolvePetriNets/results/data/1000g_10runs_still_not_working_improvement_at_end_run_3/whatever/9_08-01-2024_18-42-16/feather/gen_info.feather")
pop_df = pd.read_feather("E:/migrate_o/github_repos/EvolvePetriNets/results/data/1000g_10runs_still_not_working_improvement_at_end_run_3/whatever/9_08-01-2024_18-42-16/feather/population.feather")
species_df = pd.read_feather("E:/migrate_o/github_repos/EvolvePetriNets/results/data/1000g_10runs_still_not_working_improvement_at_end_run_3/whatever/9_08-01-2024_18-42-16/feather/species.feather")

savedir = "E:/migrate_o/github_repos/EvolvePetriNets/results/data/1000g_10runs_still_not_working_improvement_at_end_run_3/whatever/9_08-01-2024_18-42-16/feather"
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
from neatutils import endreports as er
from importlib import reload
reload(er)

er.mutation_effects_plot(pop_df, savedir)