# %%
import sys # syspath fuckery to run from same location as main
import os # set cwd to same as project cwd from vscode
from pathlib import Path

cwd = Path.cwd()

# RUN ONLY ONCE
if not os.getcwd().endswith("genetic_miner"):
    sys.path.append(str(cwd.parent.parent / "src")) # src from where all the relative imports work
    os.chdir(cwd.parent.parent) # workspace level from where I execute scripts

# %%
# list available files
files = [f"results/data/{f}" for f in os.listdir("results/data")]
print(f"files in data:")
print('\n'.join(files))

# %%
# load pickle from fname, assign name d
import pickle as pkl

# fname = files[-1]
fname = "results/data/12-08-2021_17-04-34_results.pkl"

with open(fname, "rb") as f:
    print(f"open file:\n{fname}")
    d = pkl.load(f)


# %%
ga_inst = d['speciation_params_ga_params']
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