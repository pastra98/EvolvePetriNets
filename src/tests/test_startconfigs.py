# %%
import pm4py
log = pm4py.read_xes("../pm_data/running_example.xes")
log

# %%
################################################################################
#################### SPLICING THE LOG AND MINING ON IT #########################
################################################################################
from pm4py.analysis import maximal_decomposition

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

# overlap_df = compare_all_nets(allnets)
# log_transformed_df = np.log(overlap_df + 0.001) # Add a small value to avoid log(0)

# print("normal heatmap")
# visualize_heatmap(overlap_df)
# print("log-transformed heatmap")
# visualize_heatmap(log_transformed_df)


# %%
################################################################################
######################### FOR BUILDING THE MINED NETS ##########################
################################################################################
from pm4py.algo.discovery.footprints.algorithm import apply as footprints
from neat import params, genome, initial_population
from pm4py.objects.petri_net.obj import PetriNet as pn
from importlib import reload

footprints = footprints(log)
task_list = list(footprints["activities"])

def reset_ga():
    neat_modules = [params, genome, initial_population]
    for module in neat_modules:
        reload(module)
    params.load('../params/testing/test_params.json')

def build_mined_nets(net_list):
    reset_ga()
    # There will be a higher level function that will call this function
    # It shall be responsible for creating the task list and setting it in innovs
    # generate genomes
    new_genomes = []
    for i, net in enumerate(net_list):
        net, im, fm = net['net'], net['im'], net['fm']
        g = construct_genome_from_mined_net(net)
        new_genomes.append(g)
    return new_genomes

def construct_genome_from_mined_net(net):
    g = genome.GeneticNet(dict(), dict(), dict(), task_list=task_list)
    place_dict = {"source":"start", "start":"start", "sink":"end", "end":"end"}
    trans_dict = {t:t for t in task_list} # map t.label to genome id
    
    for p in net.places:
        if p.name not in ["start", "end", "source", "sink"]:
            new_id = g.add_new_place()
            place_dict[p.name] = new_id

    for t in net.transitions:
        if t.label not in task_list:
            new_id = g.add_new_trans()
            trans_dict[t.label] = new_id

    for a in net.arcs:
        if type(a.source) == pn.Place:
            p_id = place_dict[a.source.name]
            t_id = trans_dict[a.target.label]
            g.add_new_arc(p_id, t_id)
        else:
            t_id = trans_dict[a.source.label]
            p_id = place_dict[a.target.name]
            g.add_new_arc(t_id, p_id)
    
    return g

def reload_module_and_get_fresh_genome():
    reload(genome)
    nets = mine_bootstrapped_nets(log)
    return build_mined_nets(nets)

genetic_nets = build_mined_nets(allnets)

# %%
################################################################################
#################### TEST CROSSOVER ###########################################
################################################################################
from pprint import pprint

# order of algs: alpha, inductive (best), heuristics, ilp

dad: genome.GeneticNet = reload_module_and_get_fresh_genome()[1]
mom: genome.GeneticNet = reload_module_and_get_fresh_genome()[1]

def show_genome_petri(g: genome.GeneticNet):
    g.build_petri.cache_clear()
    net, im, fm = g.build_petri()
    pm4py.view_petri_net(net, im, fm)


def print_stuff():
    print("dad")
    show_genome_petri(dad)
    pprint(dad.get_unique_component_set())

    print("mom")
    show_genome_petri(mom)
    pprint(mom.get_unique_component_set())

    dad_comp = dad.get_unique_component_set()
    mom_comp = mom.get_unique_component_set()
    print("\nshared components:")
    shared = dad_comp.intersection(mom_comp)
    pprint(shared)
    print(f"\ncomponent compat: {dad.get_genetic_distance(mom.get_unique_component_set())}")
    only_dad = dad_comp.difference(mom_comp)
    print("\nonly dad:")
    pprint(only_dad)
    only_mom = mom_comp.difference(dad_comp)
    print("\nonly mom:")
    pprint(only_mom)

baby = mom.crossover(dad)

print("mom"); show_genome_petri(mom)
print("dad"); show_genome_petri(dad)
print("baby"); show_genome_petri(baby)

# %%
################################################################################
#################### TEST NODE REACHABILITY FITNESS ############################
################################################################################
# need to define my own func bc. pm4py does not add labels

def build_digraph(net: genome.GeneticNet):
    graph = nx.DiGraph()
    for p in net.places:
        graph.add_node(p)
    for t in net.transitions:
        graph.add_node(t)
    for a in net.arcs.values():
        graph.add_edge(a.source_id, a.target_id)
    return graph

nx_graph = build_digraph(mom)
nx.draw(nx_graph, with_labels=True)

nx.descendants(nx_graph, "start")

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
    for ig, mom in zip(unchanged, genomes):
        mom.mutate(0)
        distance_dict[i]['innovation_distance'].append(ig.innov_compatibility(mom))
        distance_dict[i]['component_distance'].append(ig.component_compatibility(mom))
        
        # Convert Petri nets to NetworkX graphs
        initial_nx = convert_petri_net_to_networkx(*ig.build_petri())
        mutated_nx = convert_petri_net_to_networkx(*mom.build_petri())

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
#################### WHAT IS WRONG WITH FITNESS ################################
################################################################################
from pprint import pprint
from IPython.display import display
from typing import List
reset_ga()

def print_all_fit_metrics(g: genome.GeneticNet):
    print("perc_fit_traces", g.perc_fit_traces)
    print("average_trace_fitness", g.average_trace_fitness)
    print("log_fitness", g.log_fitness)
    print("is_sound", g.is_sound)
    print("precision", g.precision)
    print("generalization", g.generalization)
    print("simplicity", g.simplicity)
    print("fraction_used_trans", g.fraction_used_trans)
    print("fraction_tasks", g.fraction_tasks)
    print("execution_score", g.execution_score)
    print("---> overall fitness", g.fitness, "\n")


bs_list: List[genome.GeneticNet] = initial_population.get_bootstrapped_population(4, log, None)
for g in bs_list:
    display(g.get_gviz())
    g.evaluate_fitness(log)
    print_all_fit_metrics(g)

# %%
import pickle
reset_ga()

# Define the path to the .pkl file
# path = "E:/migrate_o/github_repos/EvolvePetriNets/results/data/bootstrapped_only_perc_fit_trace_06-25-2024_14-17-26/whatever/1_06-25-2024_14-17-33/reports/best_genome.pkl"
# path = "E:/migrate_o/github_repos/EvolvePetriNets/results/data/bs_oops_now_actual_perc_fit_tr_06-25-2024_15-30-29/whatever/1_06-25-2024_15-30-36/reports/best_genome.pkl"
path = "E:/migrate_o/github_repos/EvolvePetriNets/results/data/bs__perc_fit_tr_gen_prec_06-25-2024_15-59-21/whatever/1_06-25-2024_15-59-31/reports/best_genome.pkl"

def get_best_genome(path):
    with open(path, 'rb') as file:
        best_genome: genome.GeneticNet = pickle.load(file)
    best_genome.clear_cache()
    best_genome.evaluate_fitness(log)
    return best_genome

best_genome = get_best_genome(path)
display(best_genome.get_gviz())
print_all_fit_metrics(best_genome)

# %%
