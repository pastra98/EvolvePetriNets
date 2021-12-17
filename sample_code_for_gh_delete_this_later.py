from pm4py import fitness_alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments

# load running example log
lp = "pm_data/running_example.xes"
log = xes_importer.apply(lp)

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

from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness

p = {alignments.Parameters.SHOW_PROGRESS_BAR: False}

fit = replay_fitness.apply(
    log, net, im, fm,
    variant=replay_fitness.Variants.ALIGNMENT_BASED,
    parameters=p)

print(fit)

# fit = fitness_alignments(log, net, im, fm, parameters=p)

# fit = replay_fitness_evaluator.apply(
#     log, net, im, fm,
#     # parameters=parameters_tbr,
#     variant=replay_fitness_evaluator.Variants.TOKEN_BASED
# )

print(fit)
