import pm4py
from time import process_time
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter


# log = pm4py.read_xes("./data/activitylog_uci_detailed_labour.xes")
log = pm4py.read_xes("./pm_data/m1_log.xes")

alpha_miner = pm4py.algo.discovery.alpha.algorithm
inductive_miner = pm4py.algo.discovery.inductive.algorithm
heuristics_miner = pm4py.algo.discovery.heuristics.algorithm

# for name, miner in [("alpha", alpha_miner),
#                     ("inductive", inductive_miner),
#                     ("heuristics", heuristics_miner)]:
for name, miner in [("inductive", inductive_miner)]:
    print(name)
    # discover petri net and convert to bpmn
    net, initial_marking, final_marking = miner.apply(log)
    process_model = pm4py.convert.convert_to_bpmn(net, initial_marking, final_marking)

    pm4py.view_bpmn(process_model)
    pm4py.view_petri_net(net, initial_marking, final_marking)
    pnml_exporter.apply(net, initial_marking, "petri.pnml")

    
    # calculate alignments
    print("starting to measure")
    t1_start = process_time() 

    # alignment fitness
    # Aligner = pm4py.algo.conformance.alignments.petri_net.algorithm
    # alignments = Aligner.apply_log(log, net, initial_marking, final_marking)
    # total_fit = 0
    # total_cost = 0
    # for alignment in alignments:
    #     total_fit += alignment["fitness"]
    #     total_cost += alignment["cost"]
    # print(f"avg_fit {total_fit / len(alignments)}")
    # print(f"avg_cost {total_cost / len(alignments)}")

    #token based
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
    trace_count, total_fitness = 0, 0
    for trace in replayed_traces:
        trace_count += 1
        total_fitness += trace["trace_fitness"]
    print(f"total traces: {trace_count}\nfitness fraction: {total_fitness/trace_count}\n")

    t1_stop = process_time()
    print("Elapsed time during conformance check:",t1_stop-t1_start) 

    print("\n")
