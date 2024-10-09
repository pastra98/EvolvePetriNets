"""
Tests log splicing techniques (i.e. feeding the miners only a subset of the variants
in the log), incl. funcs to quickly iterate pm4py implemented miners on a given log &
convert them into genomes. Also hosts the Measuring genomic drift plots (onenote: 1st.
progress report, fig. 4), as well as code for printing fitness metrics of various mined
models.
"""
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
############### TESTING BOOTSTRAP CONFIG IN STARTCONFIGS #######################
################################################################################
import neatutils.log as lg
import neat.initial_population as ip
import scripts.analysis_scripts.useful_functions as uf
from neat.ga import PopulationComponentTracker

uf.reset_ga()

log = lg.get_log_from_xes("../pm_data/running_example.xes")
comp_tracker = PopulationComponentTracker()

initial_pop = ip.create_initial_pop(log, comp_tracker)

for g in initial_pop:
    uf.show_genome(g)


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

genomes = initial_population.get_random_genomes(n_genomes, log)
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


bs_list: List[genome.GeneticNet] = initial_population.get_bootstrap_genomes(4, log, None)
for g in bs_list:
    display(g.get_gviz())
    g.evaluate_fitness(log)
    print_all_fit_metrics(g)

# %%
# %%
log_sepsis = pm4py.read_xes("E:/migrate_o/github_repos/EvolvePetriNets/pm_data/bigger_logs/sepsis/sepsis_log.xes")
# print_variants(log_sepsis)
var = get_variants(log_sepsis)
# %%
import matplotlib.pyplot as plt

# Assuming 'var' is your nested list
lengths = [len(inner_list) for inner_list in var]

plt.hist(lengths, bins='auto')  # 'auto' lets matplotlib decide the number of bins
plt.title('Distribution of List Lengths within var')
plt.xlabel('Length of Lists')
plt.ylabel('Frequency')
plt.show()

# %%
mine_bootstrapped_nets(log_sepsis, debug=True)
# %%
l = [1,2,3]

for a in {1,2,3,4}:
    print(a)

print(l)