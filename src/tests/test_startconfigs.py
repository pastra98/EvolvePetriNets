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
from neat import params, innovs, genome, netobj, initial_population
from importlib import reload

def reset_ga():
    neat_modules = [params, innovs, genome, netobj, initial_population]
    for module in neat_modules:
        reload(module)
    params.load('../params/testing/speciation_test_component_similarity.json')
    innovs.reset()
    innovs.set_tasks(log)

def build_mined_nets(net_list):
    reset_ga()
    # There will be a higher level function that will call this function
    # It shall be responsible for creating the task list and setting it in innovs
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

# for net in allnets[:3]:
#     print("initial net")
#     pm4py.view_petri_net(net["net"])
#     print("decomposed nets")
#     for c in maximal_decomposition(net["net"], net["im"], net["fm"]):
#         pm4py.view_petri_net(c[0])

#%%
################################################################################
#################### COMPONENT OVERLAP SCORE ###################################
################################################################################
from pm4py.objects.petri_net.obj import PetriNet as pn
from collections import Counter


def format_tname(t):
    # all hidden transitions are named "t"
    return t.label if t.label in innovs.get_task_list() else "t"

def add_md_to_net(net): # of course this is total bullshit because we should not do this to petri net objects
    net["md"] = set()

    for md in maximal_decomposition(net["net"], net["im"], net["fm"]):
        a_multi_set = Counter() # multiset of arcs in the component

        for a in md[0].arcs:
            if type(a.source) == pn.Transition: # target must be a place
                # pack into iterable (list) to avoid unpacking
                a_multi_set.update([(format_tname(a.source), "p")]) # only one place per component
            else: # source must be a place, target must be a transition
                a_multi_set.update([("p", format_tname(a.target))])

        # convert multiset to tuple to make it hashable, order of tuples must be the same
        if res := tuple(sorted(a_multi_set.items())): # only add non-empty components
            net['md'].add(res)

for net in allnets:
    add_md_to_net(net)

# pm4py.view_petri_net(allnets[0]['net'])
allnets[0]['md']

#%%

def md_similarity(net1, net2):
    # return len(net1["md"].intersection(net2["md"]))/len(net1["md"].union(net2["md"]))
    return len(net1["md"] & net2["md"]) / len(net1["md"] | net2["md"])

# compute the similarity between the first 3 nets
for i in range(3):
    g1 = construct_genome_from_mined_net(allnets[i]["net"], i+1)
    for j in range(i+1, 3):
        print(md_similarity(allnets[i], allnets[j]))
        g2 = construct_genome_from_mined_net(allnets[j]["net"], j+1)
        print(g1.component_compatibility(g2))
        print()

n1, n2, n3 = allnets[:3]

# %%
################################################################################
#################### TESTING ATOMIC MUTATIONS ##################################
################################################################################

def reload_module_and_get_fresh_genome():
    reload(genome)
    return build_mined_nets(allnets[:4])


fresh_genomes = reload_module_and_get_fresh_genome()
tg = fresh_genomes[0].clone()

# net, im, fm = tg.build_petri()
# pm4py.view_petri_net(net)
tg.mutate(0)

# %%
################################################################################
#################### REACHABILITY GRAPH SUCKS ##################################
################################################################################
from pm4py.convert import convert_to_reachability_graph
rgs = []

for n in allnets[:4]:
    pm4py.view_petri_net(n["net"], debug=True)
    rg = convert_to_reachability_graph(n["net"], n["im"], n["fm"])
    rgs.append(rg)

# %%
################################################################################
#################### MEASURING GENOMIC DRIFT ###################################
################################################################################

from pm4py.convert import convert_petri_net_to_networkx
import networkx as nx
import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt
from tqdm import tqdm

reset_ga()

n_genomes = 10
n_generations = 30

genomes = initial_population.generate_n_random_genomes(n_genomes, log)
unchanged = [g.clone() for g in genomes]

distance_dict = {}

for i in tqdm(range(n_generations)):
    distance_dict[i] = {
        'innovation_distance': [],
        'component_distance': [],
        'graph_edit_distance': [],
        'simrank_similarity': []
    }
    for ig, g in zip(unchanged, genomes):
        g.mutate(0)
        distance_dict[i]['innovation_distance'].append(ig.innov_compatibility(g))
        distance_dict[i]['component_distance'].append(ig.component_compatibility(g))
        
        # Convert Petri nets to NetworkX graphs
        initial_nx = convert_petri_net_to_networkx(*ig.build_petri())
        mutated_nx = convert_petri_net_to_networkx(*g.build_petri())

        # Calculate Graph Edit Distance
        for ep in nx.optimize_edit_paths(initial_nx, mutated_nx, timeout=5):
            min_distance = ep[2]
        distance_dict[i]['graph_edit_distance'].append(min_distance)

        # Calculate SimRank Similarity
        simrank_scores = nx.simrank_similarity(initial_nx)
        simrank_avg = mean([simrank_scores[u][v] for u in initial_nx for v in mutated_nx if u in simrank_scores and v in simrank_scores[u]])
        distance_dict[i]['simrank_similarity'].append(simrank_avg)

def calculate_averages(distance_dict):
    averages = {
        'generation': [],
        'avg_innovation_distance': [],
        'avg_component_distance': [],
        'avg_graph_edit_distance': [],
        'avg_simrank_similarity': []
    }
    for gen, metrics in distance_dict.items():
        averages['generation'].append(gen)
        averages['avg_innovation_distance'].append(mean(metrics['innovation_distance']))
        averages['avg_component_distance'].append(mean(metrics['component_distance']))
        averages['avg_graph_edit_distance'].append(mean(metrics['graph_edit_distance']))
        averages['avg_simrank_similarity'].append(mean(metrics['simrank_similarity']))
    df = pd.DataFrame(averages)
    return df

df = calculate_averages(distance_dict)

# %%
def plot_averages(df):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(df['generation'], df['avg_innovation_distance'], marker='o')
    plt.title('Average Innovation Distance per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Average Innovation Distance')

    plt.subplot(2, 2, 2)
    plt.plot(df['generation'], df['avg_component_distance'], marker='o')
    plt.title('Average Component Distance per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Average Component Distance')

    plt.subplot(2, 2, 3)
    plt.plot(df['generation'], df['avg_graph_edit_distance'], marker='o')
    plt.title('Average Graph Edit Distance per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Average Graph Edit Distance')

    plt.subplot(2, 2, 4)
    plt.plot(df['generation'], df['avg_simrank_similarity'], marker='o')
    plt.title('Average SimRank Similarity per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Average SimRank Similarity')

    plt.tight_layout()
    plt.show()

plot_averages(df)

# %%
################################################################################
############ UPDATING GENOME COMPONENT DECOMPOSITION TO INCLUDE IDS ############
################################################################################

fg = reload_module_and_get_fresh_genome()

g = fg[1]

pm4py.view_petri_net(*g.build_petri(), debug=True)
c_set, arc_dict = g.get_component_set()
print(c_set, 3*"\n")
print(arc_dict)
