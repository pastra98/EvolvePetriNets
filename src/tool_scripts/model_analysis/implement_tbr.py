# %%
from tool_scripts.useful_functions import \
    load_genome, show_genome, get_aligned_traces, get_log_variants, log, reset_ga, \
    eval_and_print_fitness

from neat import params, genome, initial_population
import neatutils.fitnesscalc as fc
from pprint import pprint
from importlib import reload

# %%
path = "./tool_scripts/model_analysis/"
alpha_g = load_genome(path + "alpha_bootstrap.pkl")
inductive_g = load_genome(path + "inductive_bootstrap.pkl")
ilp_g = load_genome(path + "inductive_bootstrap.pkl")
spaghetti_g2 = load_genome(path + "inductive_bootstrap.pkl")
show_genome(alpha_g)


# %%
eval_and_print_fitness(alpha_g, log)

# %%
def compare_replays(g: genome.GeneticNet, log):
    show_genome(g)

    pnet = g.build_fc_petri()
    my_replay = pnet.replay_log(log)
    print("my agg fit", my_replay["fitness"])
    pm4py_replay = get_aligned_traces(g, log)

    for mr, pr in zip(my_replay["log_replay"], pm4py_replay):
        pprint(mr); print()
        pprint(pr)
        print(80*"-", "\n")

test_g = load_genome(path + "crappy_g1" + ".pkl")
compare_replays(test_g, log)

# %%
_ = get_log_variants(log, debug=True)
pnet = show_genome(inductive_g)
inductive_g.remove_arcs(["7559713d-2310-42f9-ad43-561ce122759f"])
# %%
inductive_g.clear_cache()