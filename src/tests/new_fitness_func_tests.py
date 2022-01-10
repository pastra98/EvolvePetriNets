# %%
import sys # syspath fuckery to run from same location as main
import os # set cwd to same as project cwd from vscode
from pathlib import Path

cwd = Path.cwd()

# RUN ONLY ONCE
if not os.getcwd().endswith("EvolvePetriNets"): # rename dir on laptop to repo name as well
    sys.path.append(str(cwd.parent.parent / "src")) # src from where all the relative imports work
    os.chdir(cwd.parent.parent) # workspace level from where I execute scripts

# notebook specific - autoreload modules
from IPython import get_ipython
ipython = get_ipython()
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")

# import various other shit
import pickle as pkl
from src import neat

# %%
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
from pm4py.visualization.footprints import visualizer as fp_visualizer
from pm4py import view_petri_net

from src.neat import fitnesscalc

from copy import copy
import pprint as pp

from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.alpha import algorithm as alpha_miner

def get_unpickled(picklename):
    with open(picklename, "rb") as f:
        d = pkl.load(f)
    return d

def footprints(log, visualize=True, printit=True):
    fp_log = footprints_discovery.apply(log, variant=footprints_discovery.Variants.ENTIRE_EVENT_LOG)
    if visualize:
        gviz = fp_visualizer.apply(fp_log, parameters={fp_visualizer.Variants.SINGLE.value.Parameters.FORMAT: "png"})
        fp_visualizer.view(gviz)
    if printit:
        for relation in fp_log:
            print(f"{relation}\n{fp_log[relation]}\n")
    return fp_log


# load log
lp = "pm_data/running_example.xes" # "pm_data/m1_log.xes"
log = xes_importer.apply(lp)


# %%
# reduce the log

def get_trace_str(trace):
    tr_events = []
    for event in trace:
        tr_events.append(event["concept:name"])
    return " -> ".join(tr_events)


splice_d = {
    1: (1, 3),
    250: (1, 2, 3, 5),
    750: (0, 1, 2, 3, 4, 5)
}

gen = 10
s_l = list(filter(lambda g: g <= gen, splice_d.keys()))[-1]

spliced_log = copy(log)
spliced_log._list = [spliced_log._list[i] for i in splice_d[s_l]]

for trace in spliced_log:
    print(get_trace_str(trace))

# %%

# get inductive model
a_net, a_im, a_fm = inductive_miner.apply(spliced_log)

# load best genome
best_g = get_unpickled("results/data/few_mutations_and_exec_01-09-2022_22-09-24/supersimple/4_01-09-2022_22-09-24/reports/best_genome.pkl")
g_net, g_im, g_fm = best_g.build_petri()

# get replays of spliced log
a_replayed_tr = fitnesscalc.get_aligned_traces(spliced_log, a_net, a_im, a_fm)
g_replayed_tr = fitnesscalc.get_aligned_traces(spliced_log, g_net, g_im, g_fm)

def transition_execution_quality(replay):
    t_exec_scoring = 1, 10
    total_quality = 0
    for trace in replay:
        bad = set([t.label for t in trace["transitions_with_problems"]])
        ok = set([t.label for t in trace["activated_transitions"]])
        score = (
            len(bad) * t_exec_scoring[0] +
            len(ok.difference(bad)) * t_exec_scoring[1]
        )
        total_quality += score
    return total_quality

print("inductive results")
view_petri_net(a_net, a_im, a_fm)
print(f"inductive fitness results\n{fitnesscalc.get_replay_fitness(a_replayed_tr)}")
print(f"exec quality: {transition_execution_quality(a_replayed_tr)}")
print(f"inductive replay\n{pp.pformat(a_replayed_tr)}\n")

print("genetic results")
view_petri_net(g_net, g_im, g_fm)
print(f"genetic fitness results\n{fitnesscalc.get_replay_fitness(g_replayed_tr)}")
print(f"exec quality: {transition_execution_quality(g_replayed_tr)}")
print(f"genetic replay\n{pp.pformat(g_replayed_tr)}\n")

# %%

g_replayed_tr = get_replayed_tr(spliced_log, g_net, g_im, g_fm)
fit = fitnesscalc.get_replay_fitness(g_replayed_tr)
print(fit)
view_petri_net(g_net, g_fm, g_im)
best_g.fitness
# print(best_g.average_trace_fitness)
# print(best_g.perc_fit_traces)
# print(best_g.log_fitness)

# %%
# investigate population
import pandas as pd
pop: pd.DataFrame
pop = pd.read_feather("results/data/few_mutations_and_exec_01-09-2022_22-09-24/supersimple/1_01-09-2022_22-09-24/reports/population.feather")

pop[pop["gen"]==2000].sort_values(by="fitness", ascending=False)