# %%
import sys # syspath fuckery to run from same location as main
import os # set cwd to same as project cwd from vscode
from pathlib import Path

cwd = Path.cwd()

# RUN ONLY ONCE
if not os.getcwd().endswith("EvolvePetriNets"): # rename dir on laptop to repo name as well
    sys.path.append(str(cwd.parent.parent / "src")) # src from where all the relative imports work
    os.chdir(cwd.parent.parent) # workspace level from where I execute scripts

# from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to
from pm4py.objects.log.importer.xes import importer as xes_importer
<<<<<<< HEAD
from pm4py.visualization.petri_net import visualizer
=======
>>>>>>> 8db9ff5e6b65b9676940c0fbe42fa5c5eaaf36c1
from pm4py import view_petri_net

# load running example log
lp = "pm_data/running_example.xes"
log = xes_importer.apply(lp)

# %%
net = PetriNet("TestPetri")
# add start and end places
start = PetriNet.Place("start")
net.places.add(start)
end = PetriNet.Place("end")
net.places.add(end)
# add register request transition (first trans that appears in every trace)
trans = PetriNet.Transition("register request", label="register request")
net.transitions.add(trans)
add_arc_from_to(start, trans, net)
add_arc_from_to(trans, end, net)
# initial marking
im = Marking()
im[start] = 1
# final marking
fm = Marking()
fm[end] = 1


replayed_tr = token_replay.apply(
    log, net, im, fm,
)

print(replayed_tr)
view_petri_net(net, im, fm)

net_gviz = visualizer.apply(net, im, fm)
savepath = f"vis/test_petrinet.svg"
net_gviz.format = "svg"
visualizer.save(net_gviz, savepath)


# %%

for tr, r_tr in zip(log, replayed_tr):
    events = [e["concept:name"] for e in tr]
    perc_activated_tr = len(r_tr["activated_transitions"]) / len(events) 
    print(perc_activated_tr)

# %%
# make a demo pnet
net = PetriNet("TestPetri")
# add start and end places
start = PetriNet.Place("start")
net.places.add(start)
end = PetriNet.Place("end")
net.places.add(end)

# add transitions
trans1 = PetriNet.Transition("task_A", label="task_A")
trans2 = PetriNet.Transition("task_B", label="task_B")
trans3 = PetriNet.Transition("task_C", label="task_C")
trans4 = PetriNet.Transition("t1")
trans5 = PetriNet.Transition("t2")

for t in [trans1, trans2, trans3, trans4, trans5]:
    net.transitions.add(t)

places = []
for n in range(1, 6):
    new_p = PetriNet.Place(f"p{n}")
    places.append(new_p)
    net.places.add(new_p)
p1, p2, p3, p4, p5 = places[0], places[1], places[2], places[3], places[4]

add_arc_from_to(start, trans1, net)
add_arc_from_to(trans1, p1, net)
add_arc_from_to(p1, trans4, net)
add_arc_from_to(trans4, p2, net)
add_arc_from_to(trans4, p4, net)
add_arc_from_to(p2, trans2, net)
add_arc_from_to(p4, trans3, net)
add_arc_from_to(trans2, p3, net)
add_arc_from_to(trans3, p5, net)
add_arc_from_to(p3, trans5, net)
add_arc_from_to(p5, trans5, net)
add_arc_from_to(trans5, end, net)

# initial marking
im = Marking()
im[start] = 1
# final marking
fm = Marking()
fm[end] = 1

net_gviz.attr(fontsize='60')

# view_petri_net(net, im, fm)

net_gviz = visualizer.apply(net, im, fm)
savepath = f"vis/GenomePhenotypePetriNet.svg"
net_gviz.format = "svg"
visualizer.save(net_gviz, savepath)

