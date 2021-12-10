# %%
import sys # syspath fuckery to run from same location as main
import os # set cwd to same as project cwd from vscode
from pathlib import Path

cwd = Path.cwd()

# RUN ONLY ONCE
if not os.getcwd().endswith("genetic_miner"):
    sys.path.append(str(cwd.parent.parent / "src")) # src from where all the relative imports work
    os.chdir(cwd.parent.parent) # workspace level from where I execute scripts

# load pickle from fname, assign name d
import pickle as pkl
import pandas as pd

# %%
# list available files
files = [f"results/data/{f}" for f in os.listdir("results/data")]
print(f"files in data:")
print('\n'.join(files))

# %%

def plot_basic_info():
    pass

def get_run_df(picklename):
    with open(picklename, "rb") as f:
        d = pkl.load(f)
    dlist = []
    excludes = ["population", "species", "best genome", "times"]
    for gen, info_dict in d["speciation_params_ga_params"].items():
        if gen != "time took":
            if int(gen) > 0:
                d = {k: info_dict[k] for k in info_dict if k not in excludes}
                d["gen"] = int(gen)
                times = info_dict["times"]
                d = d | {k: times[k] for k in times}
                dlist.append(d)
    df = pd.DataFrame(dlist)
    return df.set_index("gen")

# %%

df_small = get_run_df("results/data/12-09-2021_20-53-10_results.pkl")
df_large = get_run_df("results/data/12-10-2021_00-19-08_results.pkl")

# %%
plotvars = {
    "fitplotvars" : ["best species avg fitness", "best genome fitness", "avg pop fitness"],
    "timeplotvars" : ["pop_update", "evaluate_curr_generation"],
    "speciesplotvars" : ["num total species"],
    "innovplotvars" : ["num new innovations"],
}

for name, vars in plotvars.items():
    print("small")
    df_small[vars].plot(title=name)
    print("large")
    df_large[vars].plot(title=name)

# %%
# fname = files[-1]
# fname = "results/data/12-08-2021_17-04-34_results.pkl"

fname = "results/data/12-09-2021_22-55-41_results.pkl"

with open(fname, "rb") as f:
    print(f"open file:\n{fname}")
    d = pkl.load(f)

for k in d:
    print(k)
    # print(d[k])
    d2 = d[k]
    for k2 in d2:
        print(k2)
        # print(d2[k2])

# %%
ga_inst = d['speciation_params_ga_params']
print("small")
last_gen = ga_inst[0]

# %%
for genome in last_gen["population"]:
    # vg.show_graphviz(genome)
    vg.show_graphviz(genome)

# %%
from src.tests import visualize_genome as vg

def info_about_gen(gen: int):
    print(f"showing info about generation {gen}:")
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

# %%
info_about_gen(20)

# %%
import matplotlib.pyplot as plt

gens = d["speciation_params_ga_params"]
if "time took" in gens:
    del gens["time took"]

avgs = []

for gen in gens:
    # print(f"{gen} - {gens[gen]['avg pop fitness']}")
    avgs.append(gens[gen]['avg pop fitness'])

plt.plot(avgs)
