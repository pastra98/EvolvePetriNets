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

# %%

for tr, r_tr in zip(log, replayed_tr):
    events = [e["concept:name"] for e in tr]
    perc_activated_tr = len(r_tr["activated_transitions"]) / len(events) 
    print(perc_activated_tr)