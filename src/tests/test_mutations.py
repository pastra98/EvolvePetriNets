# %%
import sys, os, pprint
from pathlib import Path
from pm4py import view_petri_net



cwd = Path.cwd()

# RUN ONLY ONCE
if not os.getcwd().endswith("genetic_miner"):
    sys.path.append(str(cwd.parent.parent / "src")) # src from where all the relative imports work
    os.chdir(cwd.parent.parent) # workspace level from where I execute scripts

from IPython import get_ipython
ipython = get_ipython()
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")

from src.neat.genome import GeneticNet

# %%
from neat import startconfigs, innovs, params
from pm4py.objects.log.importer.xes import importer as xes_importer
from IPython.core.display import display, HTML, Image

def show_graphviz(g):
    gviz = g.get_graphviz()
    display(Image(data=gviz.pipe(format="png"), unconfined=True, retina=True))

lp = "pm_data/running_example.xes" # "pm_data/m1_log.xes"
log = xes_importer.apply(lp)
innovs.reset()
params.load("params/testing/default_speciation_params.json")

test_genomes = startconfigs.traces_with_concurrency(log)

# %%
# check cloning
target_g = test_genomes[1]
g2 = target_g.clone()
for _ in range(3):
    print()

    # pp.pprint(g.transitions)
    show_graphviz(target_g)

    print("clone:")
    # pp.pprint(g2.transitions)
    show_graphviz(g2)
    g2.trans_place_arc()

# %%
# visualize all test genomes
from pm4py import view_petri_net

for g in test_genomes:
    print(g.id)
    show_graphviz(g)
    # net, im, fm = g.build_petri()
    # view_petri_net(net, im, fm)

# %%
# perform mutations on target g
target_g: GeneticNet = test_genomes[1]

target_g.evaluate_fitness(log)
print(f"BEFORE mutation fitness:\n{target_g.fitness}")

target_g.show_nb_graphviz()

# target_g.split_arc()
# target_g.trans_trans_conn()

# target_g.place_trans_arc()
# target_g.trans_place_arc()

# target_g.extend_new_place()
# target_g.extend_new_trans()

# target_g.show_nb_graphviz()

# target_g.prune_extensions()

# target_g.show_nb_graphviz()

# target_g.evaluate_fitness(log)

print(f"AFTER mutation fitness:\n{target_g.fitness}")

# %%
target_g: GeneticNet = test_genomes[1]
target_g.show_nb_graphviz()

for _ in range(100):
    target_g.mutate(0)
    target_g.show_nb_graphviz()

# %%
target_g = test_genomes[4]
target_g.evaluate_fitness(log)

print(f"log_fitness: {target_g.log_fitness}")
print(f"is_sound: {target_g.is_sound}")
print(f"precision: {target_g.precision}")
print(f"generalization: {target_g.generalization}")
print(f"simplicity: {target_g.simplicity}")
print(f"fraction_used_trans: {target_g.fraction_used_trans}")
print(f"fraction_tasks: {target_g.fraction_tasks}")