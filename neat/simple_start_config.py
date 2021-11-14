# %% imports and read log
import random as rd

import pm4py
from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
from pm4py.visualization.footprints import visualizer as fp_visualizer

import netobj
import innovs
import genome

logpath = "../pm_data/m1_log.xes"
# logpath = "../pm_data/BPI_Challenge_2012.xes"
# logpath = "../pm_data/pdc_2016_6.xes"
# logpath = "../pm_data/running_example.xes"
# logpath = "../pm_data/simulated_running_example.xes"
log = pm4py.read_xes(logpath)

# %% footprint vis
def footprints(log, visualize=True, printit=True):
    fp_log = footprints_discovery.apply(log, variant=footprints_discovery.Variants.ENTIRE_EVENT_LOG)
    if visualize:
        gviz = fp_visualizer.apply(fp_log, parameters={fp_visualizer.Variants.SINGLE.value.Parameters.FORMAT: "png"})
        fp_visualizer.view(gviz)
    if printit:
        for relation in fp_log:
            print(f"{relation}\n{fp_log[relation]}\n")
    return fp_log


fp_log = footprints(log)

# %%
def generate_n_start_configs(n_genomes, n_arcs, log):
    fp_log = footprints(log, visualize=False, printit=False)
    task_list = list(fp_log["activities"])
    innovs.set_tasks(task_list)
    task_dict = dict()
    for task in task_list:
        task_dict[task] = netobj.GTrans(task, True)
    # generate genomes
    # for _g in range(n_genomes):
    #     arc_dict = dict()
    #     # start connection
    #     start_task_id = rd.choice(fp_log["start_activities"])
    #     start_arc_id = innovs.check_new_arc("start", start_task_id)
    #     arc_dict[start_arc_id] = netobj.GArc(start_arc_id, "start", start_task_id)
    #     # end connection
    #     end_task_id = rd.choice(fp_log["end_activities"])
    #     end_arc_id = innovs.check_new_arc("end", end_task_id)
    #     arc_dict[end_arc_id] = netobj.GArc(end_arc_id, "end", end_task_id)
    #     for _a in range(n_arcs):
    for trace in log:
        for event in trace:
            print()

def get_test_net():
    tasks = ["A", "B", "C", "D", "E"]
    innovs.set_tasks(tasks)
    gen_net = get_test_net()
    a_t = netobj.GTrans("A", True)
    b_t = netobj.GTrans("B", True)
    c_t = netobj.GTrans("C", True)
    d_t = netobj.GTrans("D", True)
    e_t = netobj.GTrans("E", True)
    t_d = {"A":a_t, "B":b_t, "C":c_t, "D":d_t, "E":e_t}
    ##########
    a1_id = innovs.store_new_arc("start", "A")
    a1 = netobj.GArc(a1_id, "start", "A")
    a2_id = innovs.store_new_arc("E", "end")
    a2 = netobj.GArc(a2_id, "E", "end")
    a_d = {a1_id:a1, a2_id:a2}
    ##########
    gen_net = genome.GeneticNet("test", t_d, dict(), a_d)
    return gen_net
