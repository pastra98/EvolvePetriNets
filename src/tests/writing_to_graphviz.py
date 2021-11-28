# %%
import pprint
from typing import final
import pm4py

from pm4py.visualization.petri_net import visualizer
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator

from pm4py.objects.petri_net.importer import importer

inductive_miner = pm4py.algo.discovery.inductive.algorithm

logpath = "./pm_data/running_example.xes"
log = pm4py.read_xes(logpath)

name, miner = "inductive", inductive_miner
use_alignments = False

# also check if I can load the parameters to make progress bar disappear
net, initial_marking, final_marking = miner.apply(log)

pm4py.view_petri_net(net, initial_marking, final_marking)

# parameters={pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: format})

# %% print out all traces of the log

def print_traces(log):
    for trace in log:
        print(f"\nTrace {trace._get_attributes()['concept:name']}:")
        tr_events = []
        for event in trace:
            tr_events.append(event["concept:name"])
        print(" -> ".join(tr_events))


# print_traces(log)

def visualize_pnet(net, im, fm, name, display=True, save=False):
    net_gviz = visualizer.apply(net, im, fm)
    if save:
        savepath = f"../vis/{name}_petrinet.png"
        visualizer.save(net_gviz, savepath)
        print(f"saved under {savepath}")
    if display:
        pm4py.view_petri_net(net, initial_marking, final_marking)


# visualize_pnet(net, initial_marking, final_marking, "sepsis", display=True, save=True)

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


# modelpath = "../pm_data/bpi2021/models/pdc2021_110000.pnml"
# visualize_pnml(modelpath, display=True, save=True)