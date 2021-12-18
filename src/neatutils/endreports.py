import matplotlib.pyplot as plt
import pandas as pd
from statistics import fmean
import traceback

# stuff in here should expect inputs from ga.get_ga_final_info()
# :   {"history": self.history, "param values": params.get_curr_curr_dict()}

# TODO: really not sure how to deal with multiple plots

def save_report(
        ga_info: dict,
        savedir: str,
        show_plots: bool,
        save_df: bool
    ) -> None:
    """saves some plots in the specified dir
    """

    full_history = ga_info["history"]
    reduced_history_df = get_reduced_history_df(full_history)
    if save_df:
        reduced_history_df.to_csv()

    species_plot(full_history, savedir=savedir, show=show_plots)
    history_plots(reduced_history_df, savedir=savedir, show=show_plots)
    best_genome_gviz(full_history, savedir=savedir, show=show_plots)
    plot_detailed_fitness(full_history, savedir=savedir, show=show_plots)

    run_report(full_history, savedir=savedir)


def get_reduced_history_df(history: dict):
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
    return df.set_index("gen")


def history_plots(reduced_history_df, savedir: str, show: bool) -> None:
    plotvars = {
        "fitness" : ["best species avg fitness", "best genome fitness", "avg pop fitness"],
        "times" : ["pop_update", "evaluate_curr_generation"],
        "species num" : ["num total species"],
        "innovs" : ["num new innovations"],
    }
    for name, vars in plotvars.items():
        plot = reduced_history_df[vars].plot(title=name)
        fig = plot.get_figure()
        try:
            fig.savefig(f"{savedir}/{name}.png")
        except:
            print(f"could not save in the given path\n{savedir}")
        if show:
            plt.show()


def species_plot(full_history, savedir: str, show: bool):
    s_dict = {}
    for gen, info in full_history.items():
        for g in info["population"]:
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
    ax.legend(
        loc="upper center",
        ncol=int(len(s_dict)/8),
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True, shadow=True
    )
    plt.rcParams["figure.figsize"] = (20,20)
    try:
        fig.savefig(f"{savedir}/species_plot.png")
    except:
        print(f"could not save in the given path\n{savedir}")
    if show:
        plt.show()


def best_genome_gviz(full_history, savedir: str, show: bool) -> None:
    gviz = list(full_history.values())[-1]["best genome"].get_graphviz()
    try:
        gviz.format = "png"
        with open(f"{savedir}/best_genome.png", "wb") as img:
            img.write(gviz.pipe())
            if show:
                gviz.view(f"{savedir}/best_genome.png")
    except:
        print(f"couldn't save gviz in\n{savedir}")


def plot_detailed_fitness(full_history, savedir: str, show: bool) -> None:
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
            plotvars[vname]["best"].append(getattr(best, vname))
            plotvars[vname]["pop_avg"].append(fmean([getattr(g, vname) for g in pop]))
    # iterate over plotvars to plot shit
    for vname, d in plotvars.items():
        fig, ax = plt.subplots()
        for metricname, values in d.items():
            ax.plot(values)
        ax.legend(d.keys())
        plt.title(vname)
        try:
            fig.savefig(f"{savedir}/{vname}.png")
        except:
            print(f"could not save in the given path\n{savedir}")
        if show:
            plt.show()


def run_report(full_history, savedir: str) -> None:
    try:
        with open(f"{savedir}/report.txt", "w") as f:
            # can put usefull stuff here
            best_fit = list(full_history.values())[-1]["best genome fitness"]
            f.write(str(best_fit))
    except:
        print(f"couldn't save report in\n{savedir}")
        print(traceback.format_exc())