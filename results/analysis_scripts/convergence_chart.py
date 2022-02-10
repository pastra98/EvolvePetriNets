# %%
import sys # syspath fuckery to run from same location as main
import os # set cwd to same as project cwd from vscode
import subprocess # for sending stuff to os and printing out
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
import pm4py
import pickle as pkl
import pandas as pd
from src.tests import visualize_genome as vg
import matplotlib.pyplot as plt
import matplotlib as mpl
from statistics import fmean, stdev

from src.neat import fitnesscalc, innovs, netobj, genome, params
from pm4py.algo.analysis.woflan import algorithm as woflan
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
from pm4py.visualization.petri_net import visualizer

import re
from copy import copy

import numpy as np

# %%
target_dir = "results/data/final_round_spec_vs_roulette_simpler_vs_full_02-03-2022_17-10-40"
i = 0

fit = {}

for root, dirs, files in os.walk(target_dir):
    root = root.replace("\\", "/")
    rsplit = root.split("/")
    if len(rsplit) == 5: # right depth to analyze 
        params_name, run_name = rsplit[-2], rsplit[-1]
        
        if params_name not in fit:
            fit[params_name] = {}

        lpath = f"{root}/{files[0]}"
        vals = []
        with open(lpath) as f:
            for l in f:
                if l.startswith(" 'best genome fitness': "):
                    vals.append(float(re.search(r"\d+\.\d+", l).group()))
        fit[params_name][run_name] = vals

# %%

plt.figure(figsize=(8,8))

model = ['roulette_simpler_model', 'speciation_simpler_model']
# model = ['roulette_full_model',  'speciation_full_model']

plt.title(model[0].lstrip("roulette").replace("_", " "))

plt.plot([], label="speciation", color="red")
plt.plot([], label="roulette", color="blue")  
# plt.plot([], label="alpha" + model[0].lstrip("roulette"), color="green")
# plt.plot([], label="heuristic" + model[0].lstrip("roulette"), color="green")
# plt.plot([], label="inductive" + model[0].lstrip("roulette"), color="green")

for param_name, runs in fit.items():
    if param_name == model[0]:
        cols = "blue", "lightblue"
    elif param_name == model[1]:
        cols = "red", "orange"
    else:
        continue
    run_arr = np.zeros((1, 3000))
    for run, vals in runs.items():
        plt.plot(vals, color = cols[1], alpha = 0.3)
        run_arr = np.vstack([run_arr, vals])
    means = np.mean(run_arr[1:11,], axis=0)
    print(run_arr[:,2999])
    plt.plot(means, color = cols[0])
    # look at deviation as well?

# plt.yticks(np.arange(24, 36, 0.5))
# plt.yticks(range(24, 37, 1))
plt.yticks(range(36, 46, 1))
plt.xticks(range(0, 3250, 250))

plt.legend()
# fig, ax = plt.subplots()
# plt.savefig(f"results/other_miners/summaries/{model[0].split('_')[1]}.pdf", dpi=300)
plt.show()

# %%
# mine other models
alpha_miner = pm4py.algo.discovery.alpha.algorithm
inductive_miner = pm4py.algo.discovery.inductive.algorithm
heuristics_miner = pm4py.algo.discovery.heuristics.algorithm

logpath = "pm_data/running_example.xes"
log = pm4py.read_xes(logpath)

spliced_log = copy(log)
spliced_log._list = [spliced_log._list[i] for i in [1, 2, 3, 5]]
# log = spliced_log

params.load("configs/final_bp/speciation_full_model.json")

fp_log = footprints_discovery.apply(log, variant=footprints_discovery.Variants.ENTIRE_EVENT_LOG)
task_list = list(fp_log["activities"])

name, miner =  "alpha", alpha_miner
# name, miner =  "heuristics", heuristics_miner
# name, miner = "inductive", inductive_miner

# net, im, fm = miner.apply(log)
net, im, fm = miner.apply(log)

pm4py.view_petri_net(net, im, fm)

# eval fitness for pm4py mined petris
def evaluate_fitness(net, im, fm, log) -> None:
    # fitness eval
    aligned_traces = fitnesscalc.get_aligned_traces(log, net, im, fm)
    trace_fitness = fitnesscalc.get_replay_fitness(aligned_traces)
    perc_fit_traces = trace_fitness["perc_fit_traces"] / 100
    average_trace_fitness = trace_fitness["average_trace_fitness"]
    log_fitness = trace_fitness["log_fitness"]
    # get fraction of task trans represented in genome
    my_task_trans = [t.label for t in net.transitions if t.label]
    if my_task_trans:
        fraction_used_trans = len(my_task_trans) / len(task_list)
        fraction_tasks = len(my_task_trans) / len(net.transitions)
    else:
        fraction_used_trans = 0
        fraction_tasks = 0
    # soundness check
    is_sound = woflan.apply(net, im, fm, parameters={
        woflan.Parameters.RETURN_ASAP_WHEN_NOT_SOUND: True,
        woflan.Parameters.PRINT_DIAGNOSTICS: False,
        woflan.Parameters.RETURN_DIAGNOSTICS: False
    })
    # precision
    precision = fitnesscalc.get_precision(log, net, im, fm)
    # generealization
    generalization = fitnesscalc.get_generalization(net, aligned_traces)
    # simplicity
    simplicity = simplicity_evaluator.apply(net)
    # execution score
    execution_score = fitnesscalc.transition_execution_quality(aligned_traces)

    print(f"""
        perc_fit_traces: {perc_fit_traces}
        average_trace_fitness: {average_trace_fitness}
        is_sound: {is_sound}
        precision: {precision}
        generalization: {generalization}
        simplicity: {simplicity}
        fraction_used_trans: {fraction_used_trans}
        fraction_tasks: {fraction_tasks}
        execution_score: {execution_score}\n""")
    fitness = (
        + params.perc_fit_traces_weight * (perc_fit_traces / 100)
        + params.average_trace_fitness_weight * (average_trace_fitness**2)
        + params.soundness_weight * int(is_sound)
        + params.precision_weight * (precision**2)
        + params.generalization_weight * (generalization**2)
        + params.simplicity_weight * (simplicity**2)
        + params.fraction_used_trans_weight * fraction_used_trans
        + params.fraction_tasks_weight * fraction_tasks
        + execution_score
    )
    print("total", fitness)

evaluate_fitness(net, im, fm, log)

# %%
def save_pnet_as_pdf(net, im, fm, name, savedir):
    gviz = visualizer.apply(net, im, fm)
    with open(f"{savedir}/{name}.pdf", "wb") as f:
        f.write(gviz.pipe(format="pdf"))

# save_pnet_as_pdf(net, im, fm, name + "_spliced", "results/other_miners")

# %%
def get_trace_str(trace):
    tr_events = []
    for event in trace:
        name = event["concept:name"].replace(" ", "\,")
        tr_events.append(name)
    return " \\rightarrow ".join(tr_events)


for i, trace in enumerate(log):
    print(f"({i+1}): {get_trace_str(trace)} \\\\")

# %%
# summary stats, this is dumb lol, don't make it public
report = {
    "roulette_full": [
        38.83521381, 36.21014645, 38.69980938, 36.32953827, 37.58262828,
        38.49590034, 38.37317597, 36.96427169, 38.62251000, 38.53679298
    ],
    "roulette_sliced": [
        25.56815963, 27.81589874, 25.16351387, 26.13137864, 24.52575197,
        25.75216162, 26.04435033, 27.24550938, 28.17416262, 26.60097542
    ],
    "speciation_full": [
        43.57572201, 42.31104161, 44.65151178, 43.58150391, 43.63885203,
        44.65151178, 43.53620339, 44.19048584, 42.20671613, 44.26918472
    ],
    "speciation_sliced": [
        32.97583815, 33.20196738, 33.45306113, 33.53452074, 32.93489744,
        35.45306113, 33.20196738, 33.20196738, 35.45306113, 33.75131841
    ]
}

for param_name, vals in report.items():
    print(param_name)
    print("max:", max(vals))
    print("mean:", fmean(vals))
    print("stdev:", stdev(vals))
    print()