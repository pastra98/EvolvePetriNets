# %%
from tool_scripts.useful_functions import \
    load_genome, show_genome, get_aligned_traces, get_log_variants, log, reset_ga

from neat import params, genome, initial_population
import neatutils.fitnesscalc as fc
from pprint import pprint
from importlib import reload

# %%
path = "./tool_scripts/model_analysis/"
alpha_g = load_genome(path + "alpha_bootstrap.pkl")
# inductive_g = load_genome(path + "inductive_bootstrap.pkl")
# ilp_g = load_genome(path + "inductive_bootstrap.pkl")
# spaghetti_g3 = load_genome(path + "spaghetti_g3.pkl")
# imprecise = load_genome(path + "imprecise_model.pkl")
only_exec = load_genome(path + "only_exec_score.pkl")
# show_genome(only_exec)
show_genome(alpha_g)


# %%
reload(genome); reload(fc)
# m = only_exec.evaluate_fitness(log)
fc_net = only_exec.build_fc_petri(log)
# replay = fc_net.replay_log()
# fc.max_replay_fitness(log)
# fc_net.evaluate()["metrics"]
# fc_net.evaluate()["replay"][4]
only_exec.evaluate_fitness(log)["metrics"]

# %%
def compare_replay_implementations(g: genome.GeneticNet, log):
    # compare the implementations
    show_genome(g)

    pnet = g.build_fc_petri()
    my_replay = pnet.replay_log(log)
    print("my agg fit", my_replay["replay_score"])
    pm4py_replay = get_aligned_traces(g, log)

    for mr, pr in zip(my_replay["log_replay"], pm4py_replay):
        pprint(mr); print()
        pprint(pr)
        print(80*"-", "\n")

def compare_replays(g1: genome.GeneticNet, g2: genome.GeneticNet, log):
    # compare replay of two different models on the same log
    show_genome(g1)
    show_genome(g2)

    pnet1 = g1.build_fc_petri()
    r1 = pnet1.replay_log(log)
    print("g1 agg fit", r1["replay_score"])

    pnet2 = g2.build_fc_petri()
    r2 = pnet2.replay_log(log)
    print("g2 agg fit", r2["replay_score"])

    for e1, e2 in zip(r1["log_replay"], r2["log_replay"]):
        pprint(e1); print()
        pprint(e2)
        print(80*"-", "\n")


reload(fc)
compare_replays(only_exec, only_exec, log)

# %%
_ = get_log_variants(log, debug=True)
pnet = show_genome(inductive_g)
inductive_g.remove_arcs(["7559713d-2310-42f9-ad43-561ce122759f"])
# %%
inductive_g.clear_cache()

# %%
from pm4py.algo.discovery.footprints.algorithm import apply as footprints
show_genome(only_exec)

fp = footprints(log)
s_dict = {t: [] for t in fp["activities"]}
for s in fp["dfg"]:
    s_dict[s[0]].append(s[1])

rp = only_exec.evaluate_fitness(log)

enabled_too_much = 0
for t in rp["log_replay"]:
    print("curr_trace")
    print([q[0] for q in t["replay"]])
    for q in t["replay"]:
        print("task:", q[0])
        should_enable = s_dict[q[0]]
        enables = q[2]
        if len(enables) > len(should_enable):
            enabled_too_much += len(enables) - len(should_enable)
        print("exec_quality:", q[1])
        print("should enable:\n", should_enable)
        print("enables:\n", enables)
        print("lendif:\n", len(should_enable) - len(enables))
        print("too much total:", enabled_too_much)
        print("-"*20)
    print("X"*20)

print("Too much enabled:", enabled_too_much)
# %%
from neatutils.log import get_log_from_xes

reload(fc)
reload(genome)

log = get_log_from_xes("../pm_data/running_example.xes")

rp = only_exec.build_fc_petri(log).evaluate()
pprint(rp)
# %%