# %%
import pm4py
from time import process_time
from pm4py.objects.petri_net.utils.decomposition import decompose
from pm4py.visualization.petri_net import visualizer

# log = pm4py.read_xes("running-example.xes")

alpha_miner = pm4py.algo.discovery.alpha.algorithm
inductive_miner = pm4py.algo.discovery.inductive.algorithm
heuristics_miner = pm4py.algo.discovery.heuristics.algorithm

# %%
################################################################################
# log = pm4py.read_xes("./pm_data/bpi2016/GroundTruthLogs/pdc_2016_5.xes")
log = pm4py.read_xes("./pm_data/m1_log.xes")

# name, miner =  "alpha", alpha_miner
# name, miner =  "heuristics", heuristics_miner
name, miner = "inductive", inductive_miner
use_alignments = False

net, initial_marking, final_marking = miner.apply(log)

# process_model = pm4py.convert.convert_to_bpmn(net, initial_marking, final_marking)
# pm4py.view_bpmn(process_model)

# pm4py.view_petri_net(net, initial_marking, final_marking)

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
#token based
else:
    Tokener = pm4py.algo.conformance.tokenreplay.algorithm
    TbrParams = Tokener.Variants.TOKEN_REPLAY.value.Parameters
    parameters_tbr = {
        TbrParams.DISABLE_VARIANTS: True,
        TbrParams.ENABLE_PLTR_FITNESS: True,
        TbrParams.STOP_IMMEDIATELY_UNFIT : True
        # TbrParams.CLEANING_TOKEN_FLOOD : True
        }
    tbr_results = Tokener.apply(log, net, initial_marking, final_marking, parameters=parameters_tbr)
    replayed_traces, place_fitness, trans_fitness, unwanted_activities = tbr_results 
    total_fitness = 0
    for trace in replayed_traces:
        total_fitness += trace["trace_fitness"]
    print(f"total traces: {len(replayed_traces)}")
    print(f"fitness fraction: {total_fitness/len(replayed_traces)}\n")

t1_stop = process_time()
print("Elapsed time during conformance check:",t1_stop-t1_start) 

# %%

parameters = {visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"}
net_gviz = visualizer.apply(net, initial_marking, final_marking, parameters=parameters)
visualizer.save(net_gviz, "petrinet.png")

list_nets = decompose(net, initial_marking, final_marking)
gviz = []

for index, model in enumerate(list_nets):
    subnet, s_im, s_fm = model
    gviz.append(visualizer.apply(subnet, s_im, s_fm))
    visualizer.save(gviz[-1], str(index)+".png")

# %%
