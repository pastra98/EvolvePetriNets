from pm4py.visualization.petri_net import visualizer

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statistics import fmean
from math import ceil

import traceback
import gc
import importlib
import matplotlib
import pickle
import os
import re


def save_report(
        ga_info: dict,
        savedir: str,
        save_df: bool,
        is_min_serialize: bool
    ) -> None:
    """saves some plots in the specified dir
    """

    matplotlib.use('Agg')
    use_species = ga_info["param_values"]["selection_strategy"] == "speciation"

    full_history = ga_info["history"]
    best_genome = ga_info["best_genome"]

    plotting_history_df = get_plotting_history_df(full_history)
    if save_df: # only works with minimal serialization lolz
        # save history df and species df (if using speciation)
        if use_species:
            # extract species names if using speciation
            fixup_history_df(plotting_history_df).to_feather(f"{savedir}/history.feather")
            get_species_df(full_history, is_min_serialize).to_feather(f"{savedir}/species.feather")
        else:
            plotting_history_df.to_feather(f"{savedir}/history.feather")
        # save population df
        pop_df = get_population_df(full_history, is_min_serialize)
        pop_df.to_feather(f"{savedir}/population.feather")
    if use_species:
        species_plot(full_history, savedir=savedir)
    history_plots(plotting_history_df, use_species, savedir=savedir)
    save_genome_gviz(best_genome, "pdf", savedir=savedir, name_prefix="best_genome")
    save_improvements(ga_info["improvements"], savedir=savedir)
    pickle_best_genome(best_genome, savedir=savedir)
    plot_detailed_fitness(full_history, savedir=savedir)
    try:
        plot_mutation_effects(pop_df, savedir=savedir)
    except:
        pass # means that multi-mutation was used, making mut effects impossible
    run_report(ga_info, savedir=savedir)
    gc.collect()


def get_plotting_history_df(history: dict):
    """expects the history from a run
    """
    dlist = []
    excludes = ["population", "species", "best genome", "times"]
    for gen, info_dict in history.items():
        d = {k: info_dict[k] for k in info_dict if k not in excludes}
        d["gen"] = int(gen)
        times = info_dict["times"]
        d |= {k: times[k] for k in times}
        dlist.append(d)
    df = pd.DataFrame(dlist)
    df.reset_index(inplace=True)
    return df


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


def species_plot(full_history, savedir: str):
    s_dict = {}
    for gen, info in full_history.items():
        for g in info["population"]:
            if isinstance(g, dict):
                s_id = g["species_id"]
            else:
                s_id = g.species_id
            if not s_id in s_dict:
                s_dict[s_id] = {gen: 1}
            elif not gen in s_dict[s_id]:
                s_dict[s_id][gen] = 1
            else:
                s_dict[s_id][gen] += 1
    total_gens = len(full_history)
    pop_sizes = []
    for s, gens in s_dict.items():
        s_sizes = []
        if (first_appear := list(gens.keys())[0]) > 1:
            s_sizes = [0] * (first_appear - 1)
        s_sizes += gens.values()
        if (last_appear := list(gens.keys())[-1]) < total_gens:
            s_sizes += [0] * (total_gens - last_appear)
        pop_sizes.append(s_sizes)
    ##
    fig, ax = plt.subplots()
    ax.stackplot(list(full_history.keys()), *pop_sizes, labels=list(s_dict.keys()), edgecolor="black")
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    legend = ax.legend(
        loc="upper center",
        ncol=ceil(len(s_dict)/8),
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True, shadow=True
    )
    plt.rcParams["figure.figsize"] = (15,5)
    try:
        fig.savefig(
            f"{savedir}/species_plot.pdf",
            bbox_extra_artists=(legend,),
            bbox_inches='tight',
            dpi=300)
    except:
        print(f"could not save in the given path\n{savedir}")
    fig.clf()
    del fig
    plt.close("all")
    gc.collect()


def save_genome_gviz(genome, ftype: str, savedir: str, name_prefix="") -> None:
    net, im, fm = genome.build_petri()
    # remove the label for all non-task transitions (without task list)
    empty_t = [t.id for t in genome.transitions.values() if not t.is_task]
    for t in net.transitions:
        if t.label in empty_t:
            t.label = None
    gviz = visualizer.apply(net, im, fm)
    # try saving in desired format
    try:
        gviz.format = ftype
        with open(f"{savedir}/{name_prefix}_id-{genome.id}.{ftype}", "wb") as f:
            f.write(gviz.pipe(format=ftype))
    except:
        print(f"couldn't save gviz in\n{savedir}")

def pickle_best_genome(best_genome, savedir: str) -> None:
    try:
        with open(f"{savedir}/best_genome.pkl", "wb") as f:
            pickle.dump(best_genome, f)
    except:
        print(f"couldn't save best_genome in\n{savedir}")

def plot_detailed_fitness(full_history, savedir: str) -> None:
    plotvars = {
        "fitness": {"best": [], "pop_avg": []},
        "perc_fit_traces": {"best": [], "pop_avg": []},
        "average_trace_fitness": {"best": [], "pop_avg": []},
        "log_fitness": {"best": [], "pop_avg": []},
        "precision": {"best": [], "pop_avg": []},
        "generalization": {"best": [], "pop_avg": []},
        "simplicity": {"best": [], "pop_avg": []},
        "is_sound": {"best": [], "pop_avg": []},
        "fraction_used_trans": {"best": [], "pop_avg": []},
        "fraction_tasks": {"best": [], "pop_avg": []},
        "execution_score": {"best": [], "pop_avg": []}
    }
    # read data into plotvars
    for info_d in full_history.values():
        best, pop = info_d["best genome"], info_d["population"]
        for vname in plotvars:
            try: # object
                plotvars[vname]["best"].append(getattr(best, vname))
                plotvars[vname]["pop_avg"].append(fmean([getattr(g, vname) for g in pop]))
            except: # dict
                plotvars[vname]["best"].append(best[vname])
                plotvars[vname]["pop_avg"].append(fmean([g[vname] for g in pop]))
    # iterate over plotvars to plot shit
    for vname, d in plotvars.items():
        fig, ax = plt.subplots()
        for metricname, values in d.items():
            ax.plot(values)
        ax.legend(d.keys())
        plt.title(vname)
        try:
            fig.savefig(f"{savedir}/{vname}.pdf", dpi=300)
        except:
            print(f"could not save in the given path\n{savedir}")
        fig.clf()
        del fig
        gc.collect()
    plt.close("all")


def run_report(ga_info, savedir: str) -> None:
    try:
        with open(f"{savedir}/report.txt", "w") as f:
            f.write(f"best fitness:\n{ga_info['best_genome'].fitness}\n")
            f.write(f"duration of run:\n{ga_info['duration']}")
    except:
        print(f"couldn't save report in\n{savedir}")
        print(traceback.format_exc())

def fixup_history_df(df):
    try: # in case we have object
        df["best species"] = df["best species"].apply(lambda bs: bs.name)
    except: # in case we have dict
        df["best species"] = df["best species"].apply(lambda bs: bs["name"])
    return df


def get_species_df(full_history, is_min_serialize: bool):
    # only works with minimal serialization right now
    l = []
    for gen, info_d in full_history.items():
        for s in info_d["species"]:
            if is_min_serialize:
                del s["alive_member_ids"]
                l.append(s | {"gen": gen})
            else:
                l.append({"gen": gen, "species_pickle": s})
    return pd.DataFrame(l)


def get_population_df(full_history, is_min_serialize: bool):
    # only works with minimal serialization right now
    l = []
    for gen, info_d in full_history.items():
        for g in info_d["population"]:
            if is_min_serialize:
                l.append(g | {"gen": gen})
            else:
                l.append({"gen": gen, "genome_pickle": g})
    return pd.DataFrame(l)


def plot_mutation_effects(pop_df, savedir: str):
    df_with_parents = pop_df.dropna(subset=['parent_id']).copy()
    df_with_parents['parent_id'] = df_with_parents['parent_id'].astype(int)
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