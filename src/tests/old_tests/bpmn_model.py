import pm4py
from pm4py.objects.bpmn.obj import BPMN
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

from models import m1
import pm4py.convert as converter


def make_model(m):
    bpmn_graph = BPMN()

    start_node = BPMN.StartEvent()
    bpmn_graph.add_node(start_node)
    m["all"]["Start"] = start_node 

    end_node = BPMN.EndEvent()
    bpmn_graph.add_node(end_node)
    m["all"]["End"] = end_node 

    for task_name in m["tasks"]:
        task = BPMN.Task(name=task_name)
        m["all"][task_name] = task
        bpmn_graph.add_node(task)

    for xor_name in m["xor"]:
        xor_g = BPMN.ExclusiveGateway(name=xor_name)
        m["all"][xor_name] = xor_g
        bpmn_graph.add_node(xor_g)

    for and_name in m["and"]:
        and_g = BPMN.ParallelGateway(name=and_name)
        m["all"][and_name] = and_g
        bpmn_graph.add_node(and_g)

    for name1, name2 in m["edges"]:
        n1, n2 = m["all"][name1], m["all"][name2]
        flow = BPMN.Flow(n1, n2)
        bpmn_graph.add_flow(flow)

    return bpmn_graph

process_model = make_model(m1)
pm4py.view_bpmn(process_model)

net, initial_marking, final_marking = pm4py.convert.convert_to_petri_net(process_model)
pm4py.view_petri_net(net, initial_marking, final_marking)
print(initial_marking)
print(final_marking)

# simulated_log = simulator.apply(net, initial_marking,
#     variant=simulator.Variants.EXTENSIVE,
#     parameters={simulator.Variants.EXTENSIVE.value.Parameters.MAX_TRACE_LENGTH: 7})

# xes_exporter.apply(simulated_log, "m1_log.xes")