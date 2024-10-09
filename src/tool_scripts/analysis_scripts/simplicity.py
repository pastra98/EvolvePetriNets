"""
Copied pm4py simplicity code out and make it work standalone. Turns out pm4py simplicity
has nothing to do with ProDiGen implementation.
"""
# %%
from tool_scripts.analysis_scripts.useful_functions import \
    load_genome, show_genome, get_aligned_traces, get_log_variants, log, reset_ga, \
    eval_and_print_fitness

from neat import params, genome, initial_population
from neatutils import fitnesscalc as fc
from pprint import pprint
from importlib import reload

# %%
path = "./tool_scripts/model_analysis/"

alpha_g = load_genome(path + "alpha_bootstrap.pkl")
inductive_g = load_genome(path + "inductive_bootstrap.pkl")
ilp_g = load_genome(path + "ilp_bootstrap.pkl")
spaghetti_g1 = load_genome(path + "spaghetti_g1.pkl")
spaghetti_g2 = load_genome(path + "spaghetti_g2.pkl")

all_g = [alpha_g, inductive_g, ilp_g, spaghetti_g1, spaghetti_g2]

show_genome(spaghetti_g2)


# %%
def get_genome_place_degrees(g: genome.GeneticNet):
    pass

# %%
from statistics import mean
"""
todos
* copy pm4py simplicity into this
"""

def pm4py_simplicity(g: genome.GeneticNet):
    # keep the default to 2
    k = 2

    # TODO: verify the real provenence of the approach before!

    all_nodes = g.places | g.transitions
    all_arc_degrees  = {k: 0 for k in all_nodes.keys()}

    for a in g.arcs.values():
        all_arc_degrees[a.source_id] += 1
        all_arc_degrees[a.target_id] += 1

    degrees = [d for d in all_arc_degrees.values() if d > 0]
    mean_degree = mean(degrees) if all_arc_degrees else 0.0

    t = mean_degree / max(degrees) 
    print(t)

    simplicity = 1.0 / (1.0 + max(mean_degree - k, 0))

    return simplicity


def analyze_simp(g: genome.GeneticNet):
    show_genome(g)
    print("precalc simp: ", g.simplicity)
    print("calc simp: ", pm4py_simplicity(g))
    net = g.build_fc_petri()
    rp = net.replay_log(log)
    print("new calc simp: ", net.get_simplicity(rp))

reload(fc)
for g in all_g:
    analyze_simp(g)
    print()

# %%
inductive_g.evaluate_fitness(log)

# %%
old = sorted([1, 2, 2, 3, 3, 3, 2, 3, 3, 2, 2, 2, 2, 2, 5, 3, 2, 2])
new = sorted([1, 2, 3, 3, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 0, 5])

print(old)
print(new)
