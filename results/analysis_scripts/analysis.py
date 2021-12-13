# %%
import sys # syspath fuckery to run from same location as main
import os # set cwd to same as project cwd from vscode
from pathlib import Path

cwd = Path.cwd()

# RUN ONLY ONCE
if not os.getcwd().endswith("genetic_miner"):
    sys.path.append(str(cwd.parent.parent / "src")) # src from where all the relative imports work
    os.chdir(cwd.parent.parent) # workspace level from where I execute scripts

# import various other shit
import pickle as pkl
import pandas as pd
from src.tests import visualize_genome as vg
import matplotlib.pyplot as plt

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
        fig = plot.get_figure()
        fig.savefig(f"{name}.png")

    

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
    ax.legend(loc='upper left')
    plt.rcParams["figure.figsize"] = (20,20)
    plt.show()


# %%
fp = "results/data/after_fixing_extensions_12-13-2021_13-38-07/speciation_test_0___12-13-2021_13-38-07/speciation_test_0___12-13-2021_13-38-07_results.pkl"
# fp = "results/data/after_fixing_extensions_12-13-2021_13-38-07/speciation_test_1___12-13-2021_13-45-15/speciation_test_1___12-13-2021_13-45-15_results.pkl"
# fp = "results/data/after_fixing_extensions_12-13-2021_13-38-07/speciation_test_2___12-13-2021_13-52-40/speciation_test_2___12-13-2021_13-52-40_results.pkl"
# df = get_run_df()

d = get_unpickled(fp)
df = pickle_to_df(d)

plot_run_df(df)
# plot_species(d)

# %%
# print_info_about_gen(pf, 10)
# plot_run_df(df)
plot_species(d)

# %%
for genome in last_gen["population"]:
    # vg.show_graphviz(genome)
    vg.show_graphviz(genome)

# %%

def get_n_sound(picklename):
    with open(picklename, "rb") as f:
        d = pkl.load(f)
    sounds = 0
    all = 0
    for gen, info_dict in d["speciation_params_ga_params"].items():
        if gen != "time took":
            if int(gen) > 0:
                pop = info_dict["population"]
                for g in pop:
                    if g["is_sound"]:
                        sounds += 1
                    all += 1
    return all, sounds

a, s = get_n_sound("results/data/12-09-2021_20-53-10_results.pkl")

print(a)
print(s)


# # %%
# # unused crap

# for k in d:
#     print(k)
#     # print(d[k])
#     d2 = d[k]
#     for k2 in d2:
#         print(k2)
#         # print(d2[k2])
