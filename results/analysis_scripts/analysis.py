# %%
import sys # syspath fuckery to run from same location as main
import os # set cwd to same as project cwd from vscode
from pathlib import Path

cwd = Path.cwd()

# RUN ONLY ONCE
if not os.getcwd().endswith("EvolvePetriNets"): # rename dir on laptop to repo name as well
    sys.path.append(str(cwd.parent.parent / "src")) # src from where all the relative imports work
    os.chdir(cwd.parent.parent) # workspace level from where I execute scripts

# notebook specific - autoreload modules
from IPython import get_ipython
ipython = get_ipython()
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")

# import various other shit
import pickle as pkl
import pandas as pd
from src.tests import visualize_genome as vg
import matplotlib.pyplot as plt
import matplotlib as mpl
from statistics import fmean
from src import neat

# %%
# list available files
files = [f"results/data/{f}" for f in os.listdir("results/data")]
print(f"files in data:")
print('\n'.join(files))

# %%
# define used funcs

def plot_run_df(df):
    plotvars = {
        "fitplotvars" : ["best species avg fitness", "best genome fitness", "avg pop fitness"],
        "timeplotvars" : ["pop_update", "evaluate_curr_generation"],
        "speciesplotvars" : ["num total species"],
        "innovplotvars" : ["num new innovations"],
    }
    for name, vars in plotvars.items():
        plot = df[vars].plot(title=name)
        plt.show()

    

def get_run_df(picklename):
    d = get_unpickled(picklename)
    df = pickle_to_df(d)
    return df


def get_unpickled(picklename):
    with open(picklename, "rb") as f:
        d = pkl.load(f)
    return d


def get_gen_species_from_pickle(picklef, gen):
    gen = picklef["history"][gen]
    return gen["species"]


def print_info_about_gen(picklef, gen: int):
    print(f"showing info about generation {gen}:")
    history = picklef["history"]
    for s in history[gen]["species"]:
        print(f"{80*'-'}\n{80*'-'}")
        print(f"{s.name} - Age: {s.age} - num members: {len(s.alive_members)}")
        print(f"fitness stuff\navg_fitness {s.avg_fitness} - adjusted: {s.avg_fitness_adjusted}")
        print(f"best eva fit {s.best_ever_fitness}")
        print(f"\ngens no improvement {s.num_gens_no_improvement}, mut rate {s.curr_mutation_rate}")
        print(f"{60*'-'} new genome:\n")
        for g in s.alive_members:
            print(f"id: {g.id}, fitness: {g.fitness}")
            print("fitness detailed:")
            print(f"trace_fitness: {g.trace_fitness}")
            print(f"is_sound: {g.is_sound}")
            print(f"precision: {g.precision}")
            print(f"generalization: {g.generalization}")
            print(f"simplicity: {g.simplicity}")
            print(f"n. transitions: {len(g.transitions)}")
            print(f"n. places: {len(g.places)}")
            print(f"n. arcs: {len(g.arcs)}")
            if g == s.leader:
                print("GENOME IS LEADER")
            if g == s.representative:
                print("GENOME IS REPRESENTATIVE")
            vg.show_graphviz(g)
            print()
        print(f"{80*'-'}\n{80*'-'}")


def pickle_to_df(picklef):
    dlist = []
    excludes = ["population", "species", "best genome", "times"]
    for gen, info_dict in picklef["history"].items():
        d = {k: info_dict[k] for k in info_dict if k not in excludes}
        d["gen"] = int(gen)
        times = info_dict["times"]
        d |= {k: times[k] for k in times}
        dlist.append(d)
    df = pd.DataFrame(dlist)
    return df.set_index("gen")


def plot_species2(results_d: dict):
    s_dict = {}
    hist = results_d["history"]
    for gen, info in hist.items():
        for g in info["population"]:
            s_id = g.species_id
            if not s_id in s_dict:
                s_dict[s_id] = {gen: 1}
            elif not gen in s_dict[s_id]:
                s_dict[s_id][gen] = 1
            else:
                s_dict[s_id][gen] += 1
    total_gens = len(hist)
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
    ax.stackplot(list(hist.keys()), *pop_sizes, labels=list(s_dict.keys()), edgecolor="black")
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
    plt.show()


def plot_species(results_d: dict):
    s_dict = {}
    hist = results_d["history"]
    for gen, info in hist.items():
        for s in info["species"]: # list of all species objects (assuming != minimal serialize)
            if s.name in s_dict:
                s_dict[s.name][gen] = s.num_members
            else:
                s_dict[s.name] = {gen: s.num_members}
    total_gens = len(hist)
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
    ax.stackplot(list(hist.keys()), *pop_sizes, labels=list(s_dict.keys()))
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
    plt.show()

def plot_detailed_fitness(result_d):
    hist = result_d["history"]
    plotvars = {
        "fitness": {"best": [], "pop_avg": []},
        "perc_fit_traces": {"best": [], "pop_avg": []},
        "average_trace_fitness": {"best": [], "pop_avg": []},
        "log_fitness": {"best": [], "pop_avg": []},
        "precision": {"best": [], "pop_avg": []},
        "generalization": {"best": [], "pop_avg": []},
        "simplicity": {"best": [], "pop_avg": []},
        "is_sound": {"best": [], "pop_avg": []},
        "fraction_used_trans": {"best": [], "pop_avg": []}
    }
    # read data into plotvars
    for info_d in hist.values():
        best, pop = info_d["best genome"], info_d["population"]
        for vname in plotvars:
            plotvars[vname]["best"].append(getattr(best, vname))
            plotvars[vname]["pop_avg"].append(fmean([getattr(g, vname) for g in pop]))
    # iterate over plotvars to plot shit
    for vname, d in plotvars.items():
        for metricname, values in d.items():
            plt.plot(values)
        plt.legend(d.keys())
        plt.title(vname)
        plt.show()
    # # plt.legend(plotvars.keys())
    # # plt.rcParams["figure.figsize"] = 10, 10
    # # plt.show()

# plot_species(d)
plot_species2(d)

# %%
fp = "results/data/much_higher_boundary_12-18-2021_15-47-58/speciation_test_0___12-18-2021_15-47-58/speciation_test_0___12-18-2021_15-47-58_results.pkl"

d = get_unpickled(fp)
df = pickle_to_df(d)
plot_species(d)

plot_run_df(df)
# plot_species(d)

# %%
# target_g = d["history"][300]["best genome"]

from pm4py.objects.log.importer.xes import importer as xes_importer
from src.neat import params, genome
from importlib import reload
lp = "pm_data/running_example.xes" # "pm_data/m1_log.xes"
log = xes_importer.apply(lp)
params.load("params/testing/default_speciation_params.json")

reload(target_g)
# target_g.trace_fitness
target_g.evaluate_fitness(log)
# target_g.im
# check if trace fitness can really be this high?? somethings fukced here
# consider getting woflan back


#  %%
# find out why best species killed

# def why_best_killed(result_d):

#     def get_set_of_spec(gen):
#         return set(map(lambda s: s.name, gen["species"]))

#     popsize = result_d["param values"]["popsize"]
#     hist = result_d["history"]
    
#     for gen, info_d in hist.items():
#         if gen == 1:
#             prev_best_g = info_d["best genome"]
#             prev_best_s = info_d["best species"]
#         else:
#             best_g = info_d["best genome"]
#             best_s = info_d["best species"]
#             if best_g.fitness < prev_best_g.fitness:
#                 print(f"\nbest genome fitness changed in gen {gen}")
#                 print(f"change from {prev_best_g.fitness} to {best_g.fitness}")
#                 print(f"prev best g spec: {prev_best_g.species_id}\nnew best g spec {best_g.species_id}")
#                 prev_s, curr_s = get_set_of_spec(hist[gen-1]), get_set_of_spec(hist[gen])
#                 diff = prev_s.difference(curr_s)
#                 print(f"species killed:\n{diff}")
#                 print(f"best genome spec id: {best_g.species_id}")
#                 print(f"alive spec:\n {curr_s}")
#                 print(f"prev spec:\n {prev_s}")
#                 print(f"prev prev spec:\n {get_set_of_spec(hist[gen-2])}")

#             # if best_s_fit < prev_best_s_fit:
#             #     print(f"\nbest species fitness changed in gen {gen}")
#             #     print(f"change from {prev_best_s_fit} to {best_s_fit}")
#             # ...
#             prev_best_g = best_g
#             prev_best_s = best_s


# why_best_killed(d)
# # plot_run_df(df)

# def get_n_sound(picklename):
#     with open(picklename, "rb") as f:
#         d = pkl.load(f)
#     sounds = 0
#     all = 0
#     for gen, info_dict in d["speciation_params_ga_params"].items():
#         if gen != "time took":
#             if int(gen) > 0:
#                 pop = info_dict["population"]
#                 for g in pop:
#                     if g["is_sound"]:
#                         sounds += 1
#                     all += 1
#     return all, sounds

# a, s = get_n_sound("results/data/12-09-2021_20-53-10_results.pkl")

# print(a)
# print(s)

# %%
################################################################################
########### PERC_FIT_TRACES CANT BE RIGHT??!!! #################################
################################################################################

from src.neat.genome import GeneticNet
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py import view_petri_net
from pm4py.objects.log.importer.xes import importer as xes_importer
import pprint as pp

lp = "pm_data/running_example.xes" # "pm_data/m1_log.xes"
log = xes_importer.apply(lp)


best_g: GeneticNet = d["history"][100]["best genome"]
# best_g.evaluate_fitness(log)

best_g.show_nb_graphviz()

net, im, fm = best_g.build_petri()
# view_petri_net(net, im, fm)

from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

# replayed_traces = token_replay.apply(log, net, im, fm)

# tracel = []
# for t in log:
#     tracel.append(" -> ".join([e["concept:name"] for e in t]))
# for t, fit in zip(tracel, replayed_traces):
#     print(t)
#     pp.pprint(fit)

# parameters_tbr = {
#     token_replay.Variants.TOKEN_REPLAY.value.Parameters.DISABLE_VARIANTS: True,
#     token_replay.Variants.TOKEN_REPLAY.value.Parameters.ENABLE_PLTR_FITNESS: True,
#     token_replay.Variants.TOKEN_REPLAY.value.Parameters.CONSIDER_REMAINING_IN_FITNESS: False
# }

# tbr_tuple = token_replay.apply(log, net, im, fm, parameters=parameters_tbr)
# replayed_traces, place_fitness, trans_fitness, unwanted_activities = tbr_tuple


fit = replay_fitness_evaluator.apply(
    log, net, im, fm,
    # parameters=parameters_tbr,
    variant=replay_fitness_evaluator.Variants.TOKEN_BASED
)

pp.pprint(fit)

# # ---------- ALIGNMENTS SHIT ----------
# from pm4py.algo.conformance.alignments.petri_net import algorithm as aligner

# alignments = aligner.apply_log(log, net, im, fm)
# total_cost = 0
# total_fit = 0
# for alignment in alignments:
#     pp.pprint(alignment)
#     total_fit += alignment["fitness"]
#     total_cost += alignment["cost"]
# print(f"total alignments: {len(alignments)}")
# print(f"fitness fraction: {total_fit / len(alignments)}")
# print(f"cost fraction {total_cost / len(alignments)}")

# %%
################################################################################
########### SPECIES PLOT MUST BE WRONG ?? ######################################
################################################################################

# for gen, info in d["history"].items():
#     # print(all([g.species_id for g in info["population"]]))
#     # print(len(info["population"]))
#     # if len(info["population"]) != 100:
#     #     print(80*"-")
#     s_members = 0
#     s_num = 0
#     for s in info["species"]:
#         s_num += s.num_members
#         if not s.obliterate:
#             s_members += len(s.alive_members)
#     print(f"{s_members} - {s_num} diff: {s_members - s_num}")


plot_species(d)
plot_species2(d)