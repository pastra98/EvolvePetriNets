# %%
import os
os.chdir('..')

import pm4py
log = pm4py.read_xes("../pm_data/running_example.xes")
log

# %%
################################################################################
#################### SPLICING THE LOG AND MINING ON IT #########################
################################################################################

from neatutils.splicing import balanced_splice
from pm4py.stats import get_variants
from pm4py.discovery import (
    discover_petri_net_alpha as alpha,
    discover_petri_net_inductive as inductive,
    discover_petri_net_heuristics as heuristics,
    discover_petri_net_ilp as ilp
)

# this is just for testing the layout of the new params layout for bootstrap
configdesign = {
    "bootstrap": {
        "alpha" : {
            "n_splices": 10,
            "splicing_method": "balanced" # balanced should also include the full log
        }
    }
}

default_params = {"show_progress_bar": False}

def print_variants(log):
    for variant in get_variants(log):
        print(' -> '.join(variant))

# -------------------- applies the 4 mining algos on a given log
def mine_bootstrapped_nets(log, debug=False):
    mined_nets = []
    for miner in [alpha, inductive, heuristics, ilp]:
        net, im, fm = miner(log)
        mined_nets.append({"net": net, "im": im, "fm": fm})
        if debug:
            print(miner)
            pm4py.view_petri_net(net, debug=True)
    return mined_nets

# -------------------- splices the log, and applies the mining algos on each splice
def get_all_nets(log):
    allnets = []
    for splice in balanced_splice(log, 4):
        mined_nets = mine_bootstrapped_nets(splice)
        allnets.extend(mined_nets)
    return allnets

allnets = get_all_nets(log)

# %%
################################################################################
#################### NEW COMPATIBILITY FUNCTION ################################
################################################################################


# ---------- GET THE LONGEST VARIANT
def find_longest_variant(log):
    return max([len(t) for t in pm4py.stats.get_variants(log).keys()])

lv = find_longest_variant(log)


# ---------- DO AN EXTENSIVE PLAYOUT TEST
def get_extensive_variants(pn, maxlen=None, debug=False):
    from pm4py.algo.simulation.playout.petri_net.variants.extensive import apply as extensive_playout

    net, im, fm = pn["net"], pn["im"], pn["fm"]
    if maxlen:
        res = extensive_playout(net, im, fm, parameters={"maxTraceLength": maxlen})
    else:
        res = extensive_playout(net, im, fm)
    variants = list(pm4py.stats.get_variants(res).keys()) # return only the variants

    # can delete the following block later
    if debug:
        pm4py.view_petri_net(net, debug=True)
        print("variants in net:", len(variants))

    return variants

def number_overlapping_variants(variants0, variants1):
    v1, v2 = set(variants0), set(variants1)
    overlap = len(v1.intersection(v2))
    fraction = overlap / len(v1.union(v2))
    return fraction

import pandas as pd


def compare_all_nets(netlist):
    netdict = dict()
    for i, net in enumerate(netlist):
        netdict[i] = get_extensive_variants(net)

    # Prepare a DataFrame to store overlap scores
    overlap_scores = pd.DataFrame(index=range(len(netlist)), columns=range(len(netlist)))
    
    # Calculate overlap scores for each combination of networks
    for i in range(len(netlist)):
        for j in range(i + 1, len(netlist)):  # Start from i+1 to avoid redundant and self comparisons
            score = number_overlapping_variants(netdict[i], netdict[j])
            overlap_scores.at[i, j] = score
            overlap_scores.at[j, i] = score  # Mirror the score to avoid redundant function calls

    # Fill NaN walues with 1 for the same net
    overlap_scores = overlap_scores.fillna(1)
    return overlap_scores

# %%
################################################################################
#################### PLOT OVERLAP AS HEATMAP ###################################
################################################################################


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def visualize_heatmap(df):
    """
    Visualizes the given DataFrame as a heatmap without annotations.
    Assumes DataFrame values are already log-transformed.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, cmap="viridis", cbar=True, annot=False)
    plt.title('Overlap Scores Heatmap')
    plt.xlabel('Network Index')
    plt.ylabel('Network Index')
    plt.show()

overlap_df = compare_all_nets(allnets)
log_transformed_df = np.log(overlap_df + 0.001) # Add a small value to avoid log(0)

print("normal heatmap")
visualize_heatmap(overlap_df)
print("log-transformed heatmap")
visualize_heatmap(log_transformed_df)


# %%
################################################################################
######################### FOR BUILDING THE MINED NETS ##########################
################################################################################
from pm4py.objects.petri_net.obj import PetriNet as pn
from pm4py.algo.discovery.footprints.algorithm import apply as footprints
from neat import params, innovs, genome, netobj

params.load('../params/testing/speciation_test.json')
innovs.reset()

def build_mined_nets(net_list):
    # There will be a higher level function that will call this function
    # It shall be responsible for creating the task list and setting it in innovs
    fp_log = footprints(log)
    task_list = list(fp_log["activities"])
    innovs.set_tasks(task_list)
    # generate genomes
    new_genomes = []
    #
    for i, net in enumerate(net_list):
        g = construct_genome_from_mined_net(net["net"], i+1)
        new_genomes.append(g)
    return new_genomes

def construct_genome_from_mined_net(net, node_prefix: int):
    # update the labels of the net for consistent naming for all mining algos
    pi, ti = 0, 0
    gplaces, gtransitions, garcs = dict(), dict(), dict()

    for p in net.places:
        if p.name not in ["start", "end"]:
            p.label = f"p{node_prefix}0{pi}" # new property added
            pi += 1
            gplaces[p.label] = netobj.GPlace(p.label)
        else: # add label property also for start and end places
            p.label = p.name

    for t in net.transitions:
        if t.label not in innovs.tasks:
            t.label = f"t{node_prefix}0{ti}" # insert 0 to avoid collision when ti>9
            ti += 1
            gtransitions[t.label] = netobj.GTrans(t.label, is_task=False)

    for i, a in enumerate(net.arcs):
        arc_id = 100*node_prefix + i
        garcs[arc_id] = netobj.GArc(arc_id, a.source.label, a.target.label)
    
    return genome.GeneticNet(gtransitions, gplaces, garcs)

genetic_nets = build_mined_nets(allnets)

#%%
from IPython.display import display

for g in genetic_nets:
    display(g.get_graphviz())
    # print(type(g.get_graphviz()))
    # break

# %%
###############################################################
###### OLD BROKEN FUNCTION; ONLY LEFT HERE FOR REFERENCE ######
###############################################################

fp_log = footprints_discovery.apply(log)
fp_log['end_activities']

from random import gauss

from neat import netobj, innovs, genome, params


# todo - this can be improved
def generate_n_random_genomes(n_genomes, log):
    # get footprints needed to get the task list
    fp_log = footprints(log, visualize=False, printit=False)
    task_list = list(fp_log["activities"])
    innovs.set_tasks(task_list)
    # generate n random genomes
    new_genomes = []
    for _ in range(n_genomes):
        gen_net = genome.GeneticNet(dict(), dict(), dict())
        for _ in range(int(abs(gauss(*params.initial_tp_gauss_dist)))):
            gen_net.trans_place_arc()
        for _ in range(int(abs(gauss(*params.initial_pt_gauss_dist)))):
            gen_net.place_trans_arc()
        for _ in range(int(abs(gauss(*params.initial_tt_gauss_dist)))):
            gen_net.trans_trans_conn()
        for _ in range(int(abs(gauss(*params.initial_pe_gauss_dist)))):
            gen_net.extend_new_place()
        for _ in range(int(abs(gauss(*params.initial_te_gauss_dist)))):
            gen_net.extend_new_trans()
        for _ in range(int(abs(gauss(*params.initial_as_gauss_dist)))):
            gen_net.split_arc()
        new_genomes.append(gen_net)
        # connect all start and end activities to start and end - debatable
        for sa in list(fp_log["start_activities"]):
            gen_net.place_trans_arc("start", sa)
        for ea in list(fp_log["end_activities"]):
            gen_net.trans_place_arc(ea, "end")

    return new_genomes
