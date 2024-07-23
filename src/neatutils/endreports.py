import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from statistics import fmean
from math import ceil

import gc
import pickle
import os


def save_report(ga_info: dict, savedir: str) -> None:
    """saves some plots in the specified dir
    """
    matplotlib.use('Agg')

    # save run report
    run_report(ga_info, savedir=savedir)

    # get the full history df, and the species df (if it exists)
    full_history = ga_info["history"]
    use_species = "species" in full_history[1]
    if use_species:
        species_df = get_species_df(full_history)
    pop_df = get_population_df(full_history)
    gen_info_df = get_gen_info_df(full_history)
    
    # save the dataframes
    species_df.to_feather(f"{savedir}/species.feather")
    pop_df.to_feather(f"{savedir}/population.feather")
    gen_info_df.to_feather(f"{savedir}/gen_info.feather")

    # save the improvements
    save_improvements(ga_info["improvements"], savedir=savedir)

    # make the plots
    if use_species:
        species_plot(species_df, savedir)
    time_stackplot(gen_info_df, savedir)
    history_plots(plotting_history_df, use_species, savedir=savedir)
    save_genome_gviz(best_genome, "pdf", savedir=savedir, name_prefix="best_genome")
    pickle_best_genome(best_genome, savedir=savedir)
    plot_detailed_fitness(full_history, savedir=savedir)
    plot_mutation_effects(pop_df, savedir=savedir)
    # close all plots and free memory
    plt.close("all")
    gc.collect()


def time_stackplot(gen_info_df, savedir: str) -> None:
    """Stackplot showing how long each step took
    """
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


def history_plots(plotting_history_df, use_species: bool, savedir: str) -> None:
    plotvars = {
        "fitness" : ["best genome fitness", "avg pop fitness"],
        "times" : ["pop_update", "evaluate_curr_generation"],
        "innovs" : ["num new innovations"],
    }
    if use_species:
        plotvars["species num"] = ["num total species"]
        plotvars["fitness"].append("best species avg fitness")
    plt.rcParams["figure.figsize"] = (15,5)
    for name, vars in plotvars.items():
        plot = plotting_history_df[vars].plot(title=name)
        fig = plot.get_figure()
        try:
            fig.savefig(f"{savedir}/{name}.pdf", dpi=300)
        except:
            print(f"could not save in the given path\n{savedir}")
        fig.clf()
        del fig
    plt.close("all")
    gc.collect()


def species_plot(species_df, savedir: str):
    # Group by gen and species_id, sum num_members, then do stackplot
    grouped = species_df.groupby(['gen', 'name'])['num_members'].sum().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.stackplot(grouped.index, grouped.T, labels=grouped.columns)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.title('Species Sizes Over Time')
    plt.xlabel('Generation')
    plt.ylabel('Number of Members')
    plt.savefig(f"{savedir}/species_plot.pdf")


def save_genome_gviz(genome, ftype: str, savedir: str, name_prefix="") -> None:
    gviz = genome.get_gviz()
    try:
        gviz.format = ftype
        with open(f"{savedir}/{name_prefix}_id-{genome.id}.{ftype}", "wb") as f:
            f.write(gviz.pipe(format=ftype))
    except:
        print(f"couldn't save gviz in\n{savedir}")

def pickle_best_genome(best_genome, savedir: str) -> None:
    best_genome.clear_cache()
    try:
        with open(f"{savedir}/best_genome.pkl", "wb") as f:
            pickle.dump(best_genome, f)
    except:
        print(f"couldn't save best_genome in\n{savedir}")


def plot_detailed_fitness(full_history, savedir: str) -> None:
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

def run_report(ga_info, savedir: str) -> None:
    savedir = savedir.rstrip("/reports")
    with open(f"{savedir}/report.txt", "w") as f:
        f.write(f"Best fitness:\n{ga_info['best_genome'].fitness}\n")
        f.write(f"Total innovs discovered:\n{ga_info['total innovs']}\n")
        f.write(f"Duration of run:\n{ga_info['time']}")


def get_species_df(full_history: dict):
    l = []
    for gen, info_d in full_history.items():
        for s in info_d["species"]:
            l.append(s | {"gen": gen})
    return pd.DataFrame(l)


def get_population_df(full_history: dict):
    l = []
    for gen, info_d in full_history.items():
        for g in info_d["population"]:
            l.append(g | {"gen": gen})
    return pd.DataFrame(l)


def get_gen_info_df(full_history: dict):
    l = []
    excludes = ["species", "population"] # exclude the lists
    for gen, info_d in full_history.items():
        for key in excludes:
            info_d.pop(key)
        l.append(info_d | {"gen": gen})
    df = pd.DataFrame(l)
    df.set_index("gen")
    return df


def plot_mutation_effects(pop_df, savedir: str):
    df_with_parents = pop_df.dropna(subset=['parent_id']).copy()
    df_with_parents['parent_id'] = df_with_parents['parent_id']
    fitness_dict = pop_df.set_index('id')['fitness'].to_dict()

    df_with_parents.loc[:, 'fitness_difference'] = df_with_parents.apply(lambda row: row['fitness'] - fitness_dict[row['parent_id']], axis=1)
    mutation_effects = {}
    mutation_frequency = {}

    for _, row in df_with_parents.iterrows():
        mutations = row['my_mutations']
        fitness_diff = row['fitness_difference']
        for mutation in mutations:
            if mutation not in mutation_effects:
                mutation_effects[mutation] = []
            mutation_effects[mutation].append(fitness_diff)
            mutation_frequency[mutation] = mutation_frequency.get(mutation, 0) + 1

    mutation_effects = {m: mutation_effects[m] for m in sorted(mutation_effects)}
    total_mutations = sum(mutation_frequency.values())
    relative_frequencies = {mutation: count / total_mutations for mutation, count in mutation_frequency.items()}

    mutations = mutation_effects.keys()
    data_for_boxplot = mutation_effects.values()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Mutation')
    ax1.set_ylabel('Fitness Impact Distribution', color=color)
    ax1.axhline(y=0, color='lightgray', linestyle='--')

    bp = ax1.boxplot(data_for_boxplot, patch_artist=True, meanline=True, showmeans=True)

    for box in bp['boxes']:
        box.set(color=color, linewidth=2)
        box.set(facecolor='lightblue')

    ax1.set_xticks(np.arange(1, len(mutations) + 1))
    ax1.set_xticklabels(mutations, rotation=45, ha='right')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Relative Frequency', color=color)
    ax2.plot(np.arange(1, len(mutations) + 1), [relative_frequencies[mutation] for mutation in mutations], color=color, marker='o', linestyle='-')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Fitness Impact Distribution and Relative Frequency of Each Mutation')
    fig.tight_layout()

    # save a dataframe as well
    summary_df = pd.DataFrame(columns=['Min', '25%', 'Median', '75%', 'Max', 'Mean', 'Frequency'])
    for mutation, effects in mutation_effects.items():
        df_temp = pd.DataFrame(effects, columns=['Effects'])
        summary = df_temp['Effects'].describe(percentiles=[.25, .5, .75])
        summary_df.loc[mutation] = [
            summary['min'], summary['25%'], summary['50%'], summary['75%'], summary['max'],
            summary['mean'], summary['count']
        ]

    try:
        fig.savefig(f"{savedir}/mutation_analysis.pdf", dpi=300)
        summary_df.to_markdown(f"{savedir}/mutation_effects.txt")
    except:
        print(f"could not save in the given path\n{savedir}")
    fig.clf()
    del fig
    plt.close("all")
    gc.collect()

def save_improvements(improvements: str, savedir: str):
    os.makedirs(f"{savedir}/improvements")
    for gen, genome in improvements.items():
        save_genome_gviz(genome, "pdf", f"{savedir}/improvements", name_prefix=f"improvement_gen-{gen}")