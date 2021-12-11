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
        df[vars].plot(title=name)
    

def get_run_df(picklename):
    d = get_unpickled(picklename)
    df = pickle_to_df(d)
    return df


def get_unpickled(picklename):
    with open(picklename, "rb") as f:
        d = pkl.load(f)
    return d


def get_gen_species_from_pickle(picklef, gen):
    gen = picklef["speciation_params_ga_params"][gen]
    return gen["species"]


def print_info_about_gen(picklef, gen: int):
    print(f"showing info about generation {gen}:")
    ga_inst = picklef['speciation_params_ga_params']
    for s in ga_inst[gen]["species"]:
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
    for gen, info_dict in picklef["speciation_params_ga_params"].items():
        if gen > 0: # fix this one day
            d = {k: info_dict[k] for k in info_dict if k not in excludes}
            d["gen"] = int(gen)
            times = info_dict["times"]
            d = d | {k: times[k] for k in times}
            dlist.append(d)
    df = pd.DataFrame(dlist)
    return df.set_index("gen")


def plot_species(picklef):
    return # TODO implement

# %%
# df = get_run_df()
pf = get_unpickled("results/data/12-10-2021_21-57-14_results.pkl")
df = pickle_to_df(pf)
# %%
print_info_about_gen(pf, 10)

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
