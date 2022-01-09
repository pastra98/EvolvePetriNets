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
from pm4py.algo.discovery.alpha import algorithm as alpha_miner 
from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
from pm4py.visualization.footprints import visualizer as fp_visualizer

def get_unpickled(picklename):
    with open(picklename, "rb") as f:
        d = pkl.load(f)
    return d

def get_replayed_tr(log, net, im, fm):
    return token_replay.apply(log, net, im, fm)

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

# get alpha model
a_net, a_im, a_fm = alpha_miner.apply(log)

# load best genome
# best_g = get_unpickled("results/data/single_thread_test_01-08-2022_21-05-59/roulette/1_01-08-2022_21-05-59/reports/best_genome.pkl")
# best_g = get_unpickled("results/data/single_thread_test_01-09-2022_00-19-58/roulette/3_01-09-2022_00-19-58/reports/best_genome.pkl")
# best_g.show_nb_graphviz()
# g_net, g_im, g_fm = best_g.build_petri()

a_replayed_tr = get_replayed_tr(log, a_net, a_im, a_fm)
# g_replayed_tr = get_replayed_tr(log, g_net, g_im, g_fm)

# for tr, r_tr in zip(log, a_replayed_tr):
#     events = [e["concept:name"] for e in tr]
#     perc_activated_tr = len(r_tr["activated_transitions"]) / len(events) 
#     # print(perc_activated_tr)
#     print(r_tr["activated_transitions"])

def transition_execution_quality(replay):
    t_exec_scoring = .2, .8
    total_quality = 0
    for trace in replay:
        bad = set([t.label for t in trace["transitions_with_problems"]])
        ok = set([t.label for t in trace["activated_transitions"]])
        score = (
            len(bad.intersection(ok)) * t_exec_scoring[0] +
            len(ok.difference(bad)) * t_exec_scoring[1]
        )
        total_quality += score
        print(score)
    print(total_quality)
    return total_quality

transition_execution_quality(a_replayed_tr)

# %%
# test mining with reduced log
from pm4py import view_petri_net
from copy import copy
from pm4py.algo.discovery.inductive import algorithm as inductive_miner 

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

gen = 262
s_l = list(filter(lambda g: g <= gen, splice_d.keys()))[-1]

# spliced_log = copy(log)
# spliced_log._list = [spliced_log._list[i] for i in splice_d[gen]]

# for trace in spliced_log:
#     print(get_trace_str(trace))

# # get alpha model
# # a_net, a_im, a_fm = alpha_miner.apply(log)
# a_net, a_im, a_fm = inductive_miner.apply(spliced_log)
# view_petri_net(a_net, a_im, a_fm)
