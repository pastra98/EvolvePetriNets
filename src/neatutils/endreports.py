import matplotlib.pyplot as plt
import pandas as pd
from statistics import fmean
from math import ceil
import traceback

import gc
import importlib
import matplotlib


def save_report(
        ga_info: dict,
        savedir: str,
        save_df: bool
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
            get_species_df(full_history).to_feather(f"{savedir}/species.feather")
        else:
            plotting_history_df.to_feather(f"{savedir}/history.feather")
        # save population df
        get_population_df(full_history).to_feather(f"{savedir}/population.feather")

    if use_species:
        species_plot(full_history, savedir=savedir)
    history_plots(plotting_history_df, use_species, savedir=savedir)
    best_genome_gviz(best_genome, savedir=savedir)
    plot_detailed_fitness(full_history, savedir=savedir)

    run_report(full_history, savedir=savedir)
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
            fig.savefig(f"{savedir}/{name}.png", dpi=300)
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
            f"{savedir}/species_plot.png",
            bbox_extra_artists=(legend,),
            bbox_inches='tight',
            dpi=300)
    except:
        print(f"could not save in the given path\n{savedir}")
    fig.clf()
    del fig
    plt.close("all")
    gc.collect()


def best_genome_gviz(best_genome, savedir: str) -> None:
    gviz = best_genome.get_graphviz()
    try:
        gviz.format = "png"
        with open(f"{savedir}/best_genome.png", "wb") as img:
            img.write(gviz.pipe())
    except:
        print(f"couldn't save gviz in\n{savedir}")


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
        "fraction_tasks": {"best": [], "pop_avg": []}
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
            fig.savefig(f"{savedir}/{vname}.png", dpi=300)
        except:
            print(f"could not save in the given path\n{savedir}")
        fig.clf()
        del fig
        gc.collect()
    plt.close("all")


def run_report(full_history, savedir: str) -> None:
    try:
        with open(f"{savedir}/report.txt", "w") as f:
            # can put usefull stuff here
            best_fit = list(full_history.values())[-1]["best genome fitness"]
            f.write(str(best_fit))
    except:
        print(f"couldn't save report in\n{savedir}")
        print(traceback.format_exc())

def fixup_history_df(df):
    try: # in case we have object
        df["best species"] = df["best species"].apply(lambda bs: bs.name)
    except: # in case we have dict
        df["best species"] = df["best species"].apply(lambda bs: bs["name"])
    return df


def get_species_df(full_history):
    # only works with minimal serialization right now
    l = []
    for gen, info_d in full_history.items():
        for s in info_d["species"]:
            del s["alive_member_ids"]
            l.append(s | {"gen": gen})
    return pd.DataFrame(l)


def get_population_df(full_history):
    # only works with minimal serialization right now
    l = []
    for gen, info_d in full_history.items():
        for g in info_d["population"]:
            l.append(g | {"gen": gen})
    return pd.DataFrame(l)
