"""
Most of this file will probably need to be scrapped, it was last touched in 2021
while doing early testing for my bachelor thesis. Maybe there is still some useful
boilerplate for using pm4py, which is why I haven't deleted it yet.
"""
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

import pprint
from typing import final
import pm4py
from time import process_time
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.visualization.petri_net import visualizer

# conformance checking stuff
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.analysis.woflan import algorithm as woflan
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator

from pm4py.objects.petri_net.importer import importer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter

from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
from pm4py.visualization.footprints import visualizer as fp_visualizer

alpha_miner = pm4py.algo.discovery.alpha.algorithm
inductive_miner = pm4py.algo.discovery.inductive.algorithm
heuristics_miner = pm4py.algo.discovery.heuristics.algorithm

# %% load a log, mine it and show conformance
################################################################################
# logpath = "../pm_data/sepsis/sepsis_log.xes"
# logpath = "../pm_data/traffic/traffic_fines.xes"
# logpath = "../pm_data/m1_log.xes"
# logpath = "../pm_data/BPI_Challenge_2012.xes"
# logpath = "../pm_data/bpi2021/train/pdc2021_1100000.xes"
# logpath = "../pm_data/pdc_2016_6.xes"
logpath = "pm_data/running_example.xes"
# logpath = "../pm_data/simulated_running_example.xes"
# logpath = "../pm_data/simulated_running_example.xes"
log = pm4py.read_xes(logpath)

# name, miner =  "alpha", alpha_miner
# name, miner =  "heuristics", heuristics_miner
name, miner = "inductive", inductive_miner
use_alignments = False

net, initial_marking, final_marking = miner.apply(log)

# process_model = pm4py.convert.convert_to_bpmn(net, initial_marking, final_marking)
# pm4py.view_bpmn(process_model)

pm4py.view_petri_net(net, initial_marking, final_marking)

net_gviz = visualizer.apply(net, initial_marking, final_marking)
savepath = f"vis/"
visualizer.save(net_gviz, savepath + "running_example" + ".png")
net_gviz.format = "pdf"
visualizer.save(net_gviz, savepath + "running_example" + ".pdf")
print(f"saved under {savepath}")

print(f"{name}, using alignments: {use_alignments}")
print("starting to measure")
t1_start = process_time() 

# calculate alignments
if use_alignments:
    Aligner = pm4py.algo.conformance.alignments.petri_net.algorithm
    alignments = Aligner.apply_log(log, net, initial_marking, final_marking)
    total_cost = 0
    total_fit = 0
    for alignment in alignments:
        total_fit += alignment["fitness"]
        total_cost += alignment["cost"]
    print(f"total alignments: {len(alignments)}")
    print(f"fitness fraction: {total_fit / len(alignments)}")
    print(f"cost fraction {total_cost / len(alignments)}")
else: # token based
    fit_start = process_time() 
    # fitness eval
    fitness = replay_fitness_evaluator.apply(
        log, net, initial_marking, final_marking,
        variant=replay_fitness_evaluator.Variants.TOKEN_BASED
        )
    print(f"fitness:\n{pprint.pformat(fitness, indent=4)}")
    print(f"fitness check took: {process_time()-fit_start} seconds\n")
    # soundness check
    sound_start = process_time()
    is_sound = woflan.apply(net, initial_marking, final_marking, parameters={woflan.Parameters.RETURN_ASAP_WHEN_NOT_SOUND: True,
                                                 woflan.Parameters.PRINT_DIAGNOSTICS: False,
                                                 woflan.Parameters.RETURN_DIAGNOSTICS: False})
    print(f"is sound: {is_sound}")
    print(f"sound check took: {process_time()-sound_start} seconds\n")
    # precision
    precision_start = process_time()
    prec = precision_evaluator.apply(log, net, initial_marking, final_marking, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
    print(f"precision: {prec}")
    print(f"precision check took: {process_time()-precision_start} seconds\n")
    # generealization
    generalization_start = process_time()
    gen = generalization_evaluator.apply(log, net, initial_marking, final_marking)
    print(f"generalization: {gen}")
    print(f"generalization check took: {process_time()-generalization_start} seconds\n")
    # simplicity
    simplicity_start = process_time()
    simp = simplicity_evaluator.apply(net)
    print(f"simplicity: {simp}")
    print(f"simplicity check took: {process_time()-simplicity_start} seconds\n")
    # some preliminary fitness measure
    genetic_fitness = .5*fitness["perc_fit_traces"]/100 + .5*int(is_sound) + .3*prec + .3*gen + .3*simp
    print(f"prelimary genetic fitness: {genetic_fitness}\n")


t1_stop = process_time()
print("Elapsed time during conformance check:",t1_stop-t1_start) 

# %% print out all traces of the log

def print_traces(log):
    for trace in log:
        print(f"\nTrace {trace._get_attributes()['concept:name']}:")
        tr_events = []
        for event in trace:
            tr_events.append(event["concept:name"])
        print(" -> ".join(tr_events))


print_traces(log)

# %% Take a model and generate traces from it, export it

def export_sim_log(net, im, name, n_traces, extensive, maxlength):
    variant = simulator.Variants.EXTENSIVE if extensive else simulator.Variants.BASIC_PLAYOUT
    s_params = {variant.value.Parameters.MAX_TRACE_LENGTH: maxlength}
    if not extensive: s_params[variant.value.Parameters.NO_TRACES] = n_traces
    simulated_log = simulator.apply(
        net,
        im,
        variant=variant,
        parameters=s_params
        )
    savepath = f"../pm_data/simulated_{name}.xes"
    xes_exporter.apply(simulated_log, savepath)
    print(f"exported to:\n{savepath}")
    return

export_sim_log(net, initial_marking, "running_example",
                n_traces=1000, extensive=False, maxlength=12)


# %% export model to a pnml file

def export_to_pnml(net, im, name):
    savepath = f"../pm_data/pnml/{name}.pnml"
    pnml_exporter.apply(net, im, savepath)
    print(f"saved petri net at:\n{savepath}")

export_to_pnml(net, initial_marking, "running_example")

# %% Visualize log

def visualize_pnet(net, im, fm, name, display=True, save=False):
    net_gviz = visualizer.apply(net, im, fm)
    if save:
        savepath = f"../vis/{name}_petrinet.png"
        visualizer.save(net_gviz, savepath)
        print(f"saved under {savepath}")
    if display:
        pm4py.view_petri_net(net, initial_marking, final_marking)


visualize_pnet(net, initial_marking, final_marking, "sepsis", display=True, save=True)

# %% Visualize a pnml file

def visualize_pnml(pnet_path, display=True, save=False):
    net, initial_marking, final_marking = importer.apply(pnet_path)
    net_gviz = visualizer.apply(net, initial_marking, final_marking)
    if save:
        savepath = f"../vis/{pnet_path.split('/')[-1].rstrip('.pnml')}_petrinet.png"
        visualizer.save(net_gviz, savepath)
        print(f"saved under {savepath}")
    if display:
        pm4py.view_petri_net(net, initial_marking, final_marking)


modelpath = "../pm_data/bpi2021/models/pdc2021_110000.pnml"
visualize_pnml(modelpath, display=True, save=True)
# %% Footprint

def footprints(log, visualize=True, printit=True):
    fp_log = footprints_discovery.apply(log, variant=footprints_discovery.Variants.ENTIRE_EVENT_LOG)
    if visualize:
        gviz = fp_visualizer.apply(fp_log, parameters={fp_visualizer.Variants.SINGLE.value.Parameters.FORMAT: "png"})
        fp_visualizer.view(gviz)
    if printit:
        for relation in fp_log:
            if relation == "min_trace_length":
                print(f"{relation})")
            else:
                print(f"{relation} (# {len(fp_log[relation])})")
            print(fp_log[relation])
            print()


footprints(log)

