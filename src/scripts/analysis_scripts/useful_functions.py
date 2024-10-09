"""
Small helper functions that can be re-used in other analysis scripts for stuff like
visualizing genomes, printing traces & variants, printing metrics, resetting the ga.
"""
from neat import params, genome, initial_population
from neatutils import log as lg

import pm4py
from pm4py.stats import get_variants
from pm4py.algo.conformance.tokenreplay.variants.token_replay import apply as get_replayed_traces
from pm4py.objects.petri_net.obj import PetriNet as pn

from pprint import pprint
from IPython.display import display
import pickle
from importlib import reload

params.load('../params/testing/test_params.json')
# log = pm4py.read_xes("../pm_data/running_example.xes")
log = lg.get_log_from_xes("../pm_data/running_example.xes")

# path = "E:/migrate_o/github_repos/EvolvePetriNets/results/data/bs_oops_now_actual_perc_fit_tr_06-25-2024_15-30-29/whatever/1_06-25-2024_15-30-36/reports/best_genome.pkl"
# path = "E:/migrate_o/github_repos/EvolvePetriNets/results/data/bs__perc_fit_tr_gen_prec_06-25-2024_15-59-21/whatever/1_06-25-2024_15-59-31/reports/best_genome.pkl"

def load_genome(path):
    with open(path, 'rb') as file:
        g: genome.GeneticNet = pickle.load(file)
    g.clear_cache()
    return g

def show_genome(g: genome.GeneticNet):
    g.clear_cache()
    display(g.get_gviz())

def get_aligned_traces(g: genome.GeneticNet, log, print=False):
    default_params = {"show_progress_bar": False}
    g.clear_cache()
    net, im, fm = g.build_petri()
    aligned_traces = get_replayed_traces(log, net, im, fm, default_params)
    if print:
        pprint(aligned_traces)
    return aligned_traces

def get_log_variants(log, debug=False):
    variants = get_variants(log)
    if debug:
        for variant in variants:
            print(' -> '.join(variant))
    return [list(v) for v in variants.keys()]


def eval_and_print_metrics(g: genome.GeneticNet, log):
    g.clear_cache()
    g.evaluate_fitness(log)
    pprint(g.fitness_metrics)

def reset_ga():
    neat_modules = [params, genome, initial_population]
    for module in neat_modules:
        reload(module)
    params.load('../params/testing/test_params.json')
