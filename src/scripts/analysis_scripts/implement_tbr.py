"""
Used this when implementing token-replay from scratch, to compare with pm4py implementation.
Could also be used later for comparing numpy-based implementation to current OOP implementation.
"""
# %%
import scripts.analysis_scripts.useful_functions as uf
from scripts.analysis_scripts.useful_functions import \
    load_genome, show_genome, get_aligned_traces, get_log_variants, log, reset_ga, \
    eval_and_print_metrics

from neat import params, genome, initial_population
import neatutils.fitnesscalc as fc
import neatutils.fitnesscalc_np as fc_np
from pprint import pprint
from importlib import reload
from pm4py.objects.petri_net.exporter.variants.pnml import export_net

# %%
reload(genome)
reload(uf)
path = "scripts/pickled_genomes/"
alpha_g = load_genome(path + "alpha_bootstrap.pkl")
inductive_g = load_genome(path + "inductive_bootstrap.pkl")
ilp_g = load_genome(path + "inductive_bootstrap.pkl")
spaghetti_g3 = load_genome(path + "spaghetti_g3.pkl")
imprecise = load_genome(path + "imprecise_model.pkl")
only_exec = load_genome(path + "only_exec_score.pkl")

glist = [alpha_g, inductive_g, ilp_g, spaghetti_g3, imprecise, only_exec]

# show_genome(only_exec)

# show_genome(alpha_g)
# eval_and_print_metrics(alpha_g, log)

show_genome(inductive_g)
eval_and_print_metrics(inductive_g, log)

# %%
reload(fc); reload(genome)

def export_pnml(g: genome.GeneticNet, fp):
    g.clear_cache()
    pn, im, fm = g.build_petri()
    export_net(pn, im, fp, fm)

def print_replay(g: genome.GeneticNet, log):
    g.clear_cache()
    fc_net = g.build_fc_petri(log)
    evaluation = fc_net.evaluate()
    pprint(evaluation)
    # pprint(evaluation["metrics"])

print_replay(inductive_g, log)
# print_replay(alpha_g, log)
# print(log["variants"])
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
################################################################################
#################### TESTING NEW MATMUL IMPLEMENTATION #########################
################################################################################

from neatutils.log import get_log_from_xes
reload(fc)
reload(fc_np)
reload(genome)

# testing genomes of the small log
def compare_oop_np_genomes(genome_list, eval_log):
    for g in genome_list:
        # show_genome(g)
        old_petri = g.build_fc_petri(log)
        old_replay = old_petri.replay_log()
        old_eval = old_petri.evaluate()

        # new_petri = g.build_fc_np_petri(log)
        # new_replay = new_petri.replay_log()
        # new_eval = new_petri.evaluate()
        print()
        print("OOP:  ", old_eval["metrics"]["aggregated_replay_fitness"])
        # print("numpy:", new_eval["metrics"]["aggregated_replay_fitness"])

compare_oop_np_genomes(glist, get_log_from_xes("../pm_data/running_example.xes"))

# midlog_genome = load_genome("I:/EvolvePetriNets/analysis/data/popsize_medium_log/execution_data/400/6_12-15-2024_16-54-10/best_genome.pkl.gz")
# midlog = get_log_from_xes()

# compare_oop_np_genomes(glist, get_log_from_xes("../pm_data/running_example.xes"))
# %%
reload(genome)
reload(fc)
reload(fc_np)

big_genome = load_genome(path + "biglog_genome.pkl")
biglog = get_log_from_xes("I:/EvolvePetriNets/pm_data/pdc_logs/2024/Training Logs/pdc2024_11100000.xes")
# show_genome(big_genome)

# %%
reload(fc)
eval_and_print_metrics(big_genome, biglog)