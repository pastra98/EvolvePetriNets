# %%
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
from pm4py.algo.discovery.footprints.algorithm import apply as footprints
from neat import params, innovs, genome, netobj

params.load('../params/testing/speciation_test.json')
innovs.reset()

def build_mined_nets(net_list):
    # There will be a higher level function that will call this function
    # It shall be responsible for creating the task list and setting it in innovs
    innovs.set_tasks(log)
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
        if t.label not in innovs.fp_log['activities']:
            t.label = f"t{node_prefix}0{ti}" # insert 0 to avoid collision when ti>9
            ti += 1
            gtransitions[t.label] = netobj.GTrans(t.label, is_task=False)

    for i, a in enumerate(net.arcs):
        arc_id = 100*node_prefix + i
        garcs[arc_id] = netobj.GArc(arc_id, a.source.label, a.target.label)
    
    return genome.GeneticNet(gtransitions, gplaces, garcs)

genetic_nets = build_mined_nets(allnets)

#%%
################################################################################
#################### TESTING REACHABILITY GRAPH ################################
################################################################################

from pm4py.objects.petri_net.utils.reachability_graph import construct_reachability_graph

# n = allnets[0]
for n in allnets[:2] + [allnets[3]]:
    ts = construct_reachability_graph(n["net"], n["im"], n["fm"], parameters={"max_elab_time": 1.0})
    pm4py.vis.view_transition_system(ts)

#%%
################################################################################
#################### TESTING MAXIMAL DECOMPOSITION #############################
################################################################################
from pm4py.analysis import maximal_decomposition

pn_list = maximal_decomposition(allnets[0]["net"], allnets[0]["im"], allnets[0]["fm"])
#%%
for net in allnets[:3]:
    print("initial net")
    pm4py.view_petri_net(net["net"])
    print("decomposed nets")
    for c in maximal_decomposition(net["net"], net["im"], net["fm"]):
        pm4py.view_petri_net(c[0])

#%%
################################################################################
#################### COMPONENT OVERLAP SCORE ###################################
################################################################################
from pm4py.objects.petri_net.obj import PetriNet as pn

net = allnets[0]
md = maximal_decomposition(net["net"], net["im"], net["fm"])

# for c in md:
#     pm4py.view_petri_net(c[0])

def format_tname(t):
    # all hidden transitions are named "t"
    return t.label if t.label in innovs.get_task_list() else "t"

def add_md_to_net(net): # of course this is total bullshit because we should not do this to petri net objects
    net["md"] = set()

    for md in maximal_decomposition(net["net"], net["im"], net["fm"]):
        aset = set()

        for a in md[0].arcs:
            if type(a.source) == pn.Transition: # target must be a place
                aset.add((format_tname(a.source), "p")) # there is only one place per component

            else: # source must be a place, target must be a transition
                aset.add(("p", format_tname(a.target)))

        # could remove pure (t,p) and (p,t) pairs here
        net["md"].add(frozenset(aset))

for net in allnets:
    add_md_to_net(net)

def md_similarity(net1, net2):
    return len(net1["md"].intersection(net2["md"]))/len(net1["md"].union(net2["md"]))

# compute the similarity between the first 3 nets
for i in range(3):
    for j in range(i+1, 3):
        print(md_similarity(allnets[i], allnets[j]))

n1, n2, n3 = allnets[:3]

#%%
allnets[2]['md']

#%%
from collections import Counter

# Create multisets
multiset1 = Counter()
multiset2 = Counter()

# Add a tuple
multiset1.update([("t", "p")])
# Add the same tuple again
multiset1.update([("t", "p")])

# update the second multiset
multiset2.update({("t", "p"): 1, ("p", "t"): 1})

# Display the multisets
print(multiset1)
print(multiset2)

# Perform set operations
intersection = multiset1 & multiset2
union = multiset1 | multiset2

# Display the results
print("Intersection:", intersection)
print("Union:", union)

print(multiset1 == multiset2)

multiset1.update([("p", "t")])
multiset2.update([("t", "p")])

print(multiset1 == multiset2)

# Display the results
print("Intersection:", multiset1 & multiset2)
print("Union:", multiset1 | multiset2)

#%%

t1 = (("p", "t"), ("t", "p"))
t2 = (("t", "p"), ("p", "t"))
t1 == t2

#%%
from IPython.display import display
################################################################################
#################### TESTING MUTATIONS #########################################
################################################################################
def turn_genome_into_stupid_dict(g):
    net, im, fm = g.build_petri()
    print(add_md_to_net({"net": net, "im": im, "fm": fm}))

tg = genetic_nets[0].clone()

unchanged = turn_genome_into_stupid_dict(tg.clone())


for _ in range(10):
    tg.mutate(0) # mutation rate 0 is normal mutation rate

    # display(tg.get_graphviz())
    tg_dict = turn_genome_into_stupid_dict(tg)
    print(tg_dict)
    # print(md_similarity(unchanged, tg_dict))
    print(80*"-")

# display(unchanged.get_graphviz())
# pm4py.view_petri_net(unet)

#%%
sd = turn_genome_into_stupid_dict(genetic_nets[0])
print(sd)

#%%

for g in genetic_nets:
    display(g.get_graphviz())

# %%
from pm4py.convert import convert_to_reachability_graph
rgs = []

for n in allnets[:4]:
    pm4py.view_petri_net(n["net"], debug=True)
    # rg = convert_to_reachability_graph(n["net"], n["im"], n["fm"])
    # rgs.append(rg)


#%%
from pm4py.algo.discovery.footprints.algorithm import apply as footprints

fp_log = footprints(log)

#%%
list(fp_log['activities'])

#%%

innovs.reset()
innovs.set_tasks(log)
#%%
innovs.get_task_list()