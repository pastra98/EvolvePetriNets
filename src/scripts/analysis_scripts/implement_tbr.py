"""
Used this when implementing token-replay from scratch, to compare with pm4py implementation.
Could also be used later for comparing numpy-based implementation to current OOP implementation.
"""
# %%
################################################################################
#################### LOAD SOME GENOMES / DEFINE HELPER FUNCS ###################
################################################################################

import scripts.analysis_scripts.useful_functions as uf
from scripts.analysis_scripts.useful_functions import \
    load_genome, show_genome, get_aligned_traces, get_log_variants, log, reset_ga, \
    eval_and_print_metrics, analyze_log_variants

from neat import params, genome, initial_population
import neatutils.fitnesscalc as fc
# import neatutils.fitnesscalc_np as fc_np
from pprint import pprint
from importlib import reload
from pm4py.objects.petri_net.exporter.variants.pnml import export_net

path = "scripts/pickled_genomes/"
alpha_g = load_genome(path + "alpha_bootstrap.pkl")
inductive_g = load_genome(path + "inductive_bootstrap.pkl")
ilp_g = load_genome(path + "inductive_bootstrap.pkl")
spaghetti_g3 = load_genome(path + "spaghetti_g3.pkl")
imprecise = load_genome(path + "imprecise_model.pkl")
only_exec = load_genome(path + "only_exec_score.pkl")

glist = [alpha_g, inductive_g, ilp_g, spaghetti_g3, imprecise, only_exec]


show_genome(alpha_g)
eval_and_print_metrics(alpha_g, log)

# show_genome(inductive_g)
# eval_and_print_metrics(inductive_g, log)

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

# print_replay(inductive_g, log)

# %%
################################################################################
#################### COMPARE REPLAYS ###########################################
################################################################################

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

    pnet1 = g1.build_fc_petri(log)
    r1 = pnet1.replay_log()
    print("g1 agg fit", r1)

    pnet2 = g2.build_fc_petri(log)
    r2 = pnet2.replay_log()
    print("g2 agg fit", r2)

    for e1, e2 in zip(r1, r2):
        pprint(e1); print()
        pprint(e2)
        print(80*"-", "\n")


reload(fc)
# compare_replays(only_exec, only_exec, log)


# %%
################################################################################
#################### TEST OVER-ENABLED FITNESS SCORE ###########################
################################################################################

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

big_genome = load_genome(path + "biglog_genome.pkl")
biglog = get_log_from_xes("I:/EvolvePetriNets/pm_data/pdc_logs/2024/Training Logs/pdc2024_11100000.xes")
# show_genome(big_genome)
# biglog["fragments"]
# blv = [v[0] for v in biglog["shortened_variants"].values() if len(v[0]) > 1]
# blv
target_log = biglog
# %%
################################################################################
#################### TESTING FRAGMENTING THE LOG ###############################
################################################################################
# Extremely messy ugly code, but this should work to identify overlaps in variants
# there are surely 100 ways to improve this but I'm glad it works :)
from copy import copy

# TODO clean this up
variants = list(target_log["variants"].keys())
longest_v = max([len(v) for v in variants])

# first get the branches of the task tree
task_tree = {}
for task_nr in range(longest_v):
    curr_tasks = {}
    for var_i, var in enumerate(variants):
        if task_nr+1 > len(var):
            continue # shorter variants
        task = var[task_nr]
        curr_tasks.setdefault(task, set()).add(var_i)
    task_tree[task_nr] = curr_tasks

# then construct fragments/prefixes
final_fragments = {}
fragments = {tuple([k]): v for k, v in task_tree[0].items()}
for task_nr, branches in list(task_tree.items())[1:]:
    new_fragments = {}
    for t, vids in branches.items():
    
        for fk, fv in fragments.items():
            if overlap := vids.intersection(fv):
                if len(overlap) == len(fv): # extend current fragment
                    new_fragments[fk + tuple([t])] = overlap
                else: # fragment branches of
                    if fk in final_fragments:
                        final_fragments[fk] = final_fragments[fk].union(fv)
                    else:
                        final_fragments[fk] = fv # keep the old fragment in final fragments
                    new_fragments[tuple([t])] = overlap # start a new one
    fragments = new_fragments

# a dict to give each fragment/prefix an id
tagged_fragments = {}
for i, item in enumerate(final_fragments.items()):
    k, v = item
    tagged_fragments[i] = [k, v]

# assign prefixes to traces, save the rest of the trace
shortened_variants = {}
prefix_predecessors = {}
unique_prefixes = set()
for i, var in enumerate(variants):
    match_frags = []
    rest_of_var = copy(var)
    for j, f in tagged_fragments.items():
        tasks, matching_vars = f
        if i in matching_vars and rest_of_var[:len(tasks)] == tasks:
            if len(match_frags) <= 1:
                match_frags.append(j)
                rest_of_var = rest_of_var[len(tasks):]
                prefix_predecessors[j] = match_frags[0]
            # check if this is the one predecessor
            else:
                prefix_pred = prefix_predecessors.get(j)
                if prefix_pred == None or prefix_pred == match_frags[-1]:
                    prefix_predecessors[j] = match_frags[-1]
                    match_frags.append(j)
                    rest_of_var = rest_of_var[len(tasks):]
    shortened_variants[i] = [match_frags, rest_of_var]
    unique_prefixes.add(tuple(match_frags))

shortened_variants
prefix_predecessors
unique_prefixes
prefix_map = {k: v[0] for k, v in tagged_fragments.items()}

pprint(prefix_predecessors)
pprint(unique_prefixes)
pprint(prefix_map)

# %%

# %%
################################################################################
#################### TESTING USING FRAGMENTS DURING REPLAY #####################
################################################################################
import neatutils.log as lg
import cProfile
import time

reload(lg)
reload(genome)
reload(fc)

# log = lg.get_log_from_xes("../pm_data/running_example.xes")
# pnet = alpha_g.build_fc_petri(log)

big_genome = load_genome(path + "biglog_genome.pkl")
biglog = lg.get_log_from_xes("I:/EvolvePetriNets/pm_data/pdc_logs/2024/Training Logs/pdc2024_11100000.xes")
pnet = big_genome.build_fc_petri(biglog)

# mid_genome = load_genome(path + "biglog_genome.pkl")
# midlog = lg.get_log_from_xes("I:/EvolvePetriNets/pm_data/pdc_logs/2022/Training Logs/pdc2022_1110000.xes")
# pnet = big_genome.build_fc_petri(biglog)


# Timing the replay_log method
start_time = time.time()
old = pnet.replay_log()
end_time = time.time()
replay_log_time = end_time - start_time

# Timing the prefix_replay_log method
start_time = time.time()
new = pnet.prefix_replay_log()
end_time = time.time()
prefix_replay_log_time = end_time - start_time

# Print the timing results
print(f"replay_log method took {replay_log_time:.4f} seconds")
print(f"prefix_replay_log method took {prefix_replay_log_time:.4f} seconds")

# Compare the results
print(old == new)

# %%
reload(genome)
reload(fc)
big_genome.evaluate_fitness(biglog)["metrics"]