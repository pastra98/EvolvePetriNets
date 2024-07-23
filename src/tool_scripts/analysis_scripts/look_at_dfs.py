# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from statistics import fmean
from math import ceil

gen_info_df = pd.read_feather("E:/migrate_o/github_repos/EvolvePetriNets/results/data/test_new_df_3_07-23-2024_16-42-44/whatever/1_07-23-2024_16-42-50/reports/gen_info.feather")
pop_df = pd.read_feather("E:/migrate_o/github_repos/EvolvePetriNets/results/data/test_new_df_3_07-23-2024_16-42-44/whatever/1_07-23-2024_16-42-50/reports/population.feather")
species_df = pd.read_feather("E:/migrate_o/github_repos/EvolvePetriNets/results/data/test_new_df_3_07-23-2024_16-42-44/whatever/1_07-23-2024_16-42-50/reports/species.feather")

# %%
savedir = "E:/migrate_o/github_repos/EvolvePetriNets/results/data/test_new_df_3_07-23-2024_16-42-44/whatever/1_07-23-2024_16-42-50/reports"

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
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.stackplot(
        times_df.index,
        times_df["evaluate_curr_generation"],
        times_df["pop_update"],
        labels=[
            "Evaluate Current Generation", "Population Update"
        ])
    ax.legend()
    plt.title('Time Spent Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Time')
    plt.savefig(f"{savedir}/time_plot.pdf")

# help(ax.stackplot)
time_stackplot(gen_info_df, savedir)



# def history_plots(plotting_history_df, use_species: bool, savedir: str) -> None:
#     plotvars = {
#         "fitness" : ["best_genome_fitness", "avg_pop_fitness"],
#         "times" : ["pop_update", "evaluate_curr_generation"],
#         "innovs" : ["num new innovations"],
#     }
#     if use_species:
#         plotvars["species num"] = ["num total species"]
#         plotvars["fitness"].append("best species avg fitness")
#     plt.rcParams["figure.figsize"] = (15,5)
#     for name, vars in plotvars.items():
#         plot = plotting_history_df[vars].plot(title=name)
#         fig = plot.get_figure()
#         try:
#             fig.savefig(f"{savedir}/{name}.pdf", dpi=300)
#         except:
#             print(f"could not save in the given path\n{savedir}")
#         fig.clf()
#         del fig
#     plt.close("all")
#     gc.collect()