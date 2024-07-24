# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from statistics import fmean
from math import ceil

gen_info_df = pd.read_feather("C:/Users/pauls/Projekte/Programming/GithubRepos/EvolvePetriNets/results/data/test_working_07-24-2024_12-29-47/whatever/1_07-24-2024_12-29-53/feather/gen_info.feather")
pop_df = pd.read_feather("C:/Users/pauls/Projekte/Programming/GithubRepos/EvolvePetriNets/results/data/test_working_07-24-2024_12-29-47/whatever/1_07-24-2024_12-29-53/feather/population.feather")
species_df = pd.read_feather("C:/Users/pauls/Projekte/Programming/GithubRepos/EvolvePetriNets/results/data/test_working_07-24-2024_12-29-47/whatever/1_07-24-2024_12-29-53/feather/species.feather")

savedir = "C:/Users/pauls/Projekte/Programming/GithubRepos/EvolvePetriNets/results/data/test_working_07-24-2024_12-29-47/whatever/1_07-24-2024_12-29-53"
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
def fitness_metrics_plot(full_history, savedir: str) -> None:
    plotvars = {
        "fitness": {"best": [], "pop_avg": []},
        "io": {"best": [], "pop_avg": []},
        "mbm": {"best": [], "pop_avg": []},
        "ftt": {"best": [], "pop_avg": []},
        "tbt": {"best": [], "pop_avg": []},
        "precision": {"best": [], "pop_avg": []},
        "execution_score": {"best": [], "pop_avg": []}
    }
    # read data into plotvars
    for info_d in full_history.values():
        best, pop = info_d["best genome"], info_d["population"]
        for vname in plotvars:
            try:  # object
                plotvars[vname]["best"].append(getattr(best, vname))
                plotvars[vname]["pop_avg"].append(fmean([getattr(g, vname) for g in pop]))
            except:  # dict
                plotvars[vname]["best"].append(best[vname])
                plotvars[vname]["pop_avg"].append(fmean([g[vname] for g in pop]))

    # Separate plots for 'fitness' and 'execution_score'
    for special_var in ["fitness", "execution_score"]:
        d = plotvars.pop(special_var)  # Remove and retrieve special_var data
        fig, ax = plt.subplots()
        for metricname, values in d.items():
            ax.plot(values, label=metricname)
        ax.legend()
        plt.title(special_var)
        try:
            fig.savefig(f"{savedir}/{special_var}.pdf", dpi=300)
        except:
            print(f"could not save in the given path\n{savedir}")
        plt.close(fig)


    # Combined plots for the rest
    fig, ax_best = plt.subplots()
    fig, ax_pop_avg = plt.subplots()
    for vname, d in plotvars.items():
        ax_best.plot(d["best"], label=vname)
        ax_pop_avg.plot(d["pop_avg"], label=vname)
    ax_best.legend()
    ax_best.set_title("Best of Each Variable")
    ax_pop_avg.legend()
    ax_pop_avg.set_title("Population Average of Each Variable")
    try:
        ax_best.figure.savefig(f"{savedir}/combined_best.pdf", dpi=300)
        ax_pop_avg.figure.savefig(f"{savedir}/combined_pop_avg.pdf", dpi=300)
    except:
        print(f"could not save in the given path\n{savedir}")
    plt.close("all")



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

# %%
pop_df.iloc[0]["fitness_metrics"]

# %%
# metrics = pd.json_normalize(pop_df['fitness_metrics']).add_prefix("metric_")
# pop_df = pd.concat([pop_df.drop(columns=['fitness_metrics']), metrics], axis=1)

def metrics_plot(pop_df: pd.DataFrame, savedir: str):
    """Combined plots of metrics for best genome and populaiton avg
    """
    metrics = [col for col in pop_df.columns if col.startswith("metric_")]
    plt.figure(figsize=FSIZE)
    plt.plot(gen_info_df[["best_genome_fitness", "best_species_avg_fitness", "avg_pop_fitness"]])
    plt.legend(["Best Genome Fitness", "Best Species Average Fitness", "Average Population Fitness"])
    plt.title("Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    # plt.savefig(f"{savedir}/best_metrics.pdf")
    # plt.savefig(f"{savedir}/avg_metrics.pdf")

metrics_plot(pop_df, savedir)
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
        plt.legend()
        plt.savefig(f"{savedir}/{title.split()[0]}_metrics.pdf")
    # find metrics columns, aggregate them over generations, plot best and avg
    metrics = [col for col in pop_df.columns if col.startswith("metric_")]
    aggregated_metrics = pop_df.groupby('gen')[metrics].agg(['max', 'mean'])
    df_best = aggregated_metrics.xs('max', level=1, axis=1)
    df_avg = aggregated_metrics.xs('mean', level=1, axis=1)
    plot_metrics(df_best, 'Best Genome Metrics Over Generations')
    plot_metrics(df_avg, 'Average Population Metrics Over Generations')


metrics_plot(pop_df, savedir)