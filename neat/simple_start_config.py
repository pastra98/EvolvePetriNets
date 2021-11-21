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
    arc_dict = dict()
    for task in task_list:
        task_dict[task] = netobj.GTrans(task, True)
    # generate genomes
    new_genomes = []
    net_id = 0
    for trace in log:
        net_id += 1
        gen_net = genome.GeneticNet(net_id, task_dict, dict(), dict())
        for i, task in enumerate(trace):
            # hacky shit -------------------------------------------------------
            is_parallel = False
            # hacky shit -------------------------------------------------------
            curr_task = task["concept:name"]
            if i == 0:
                start_arc_id = innovs.check_arc("start", curr_task)
                start_arc = netobj.GArc(start_arc_id, "start", curr_task)
                gen_net.arcs[start_arc_id] = start_arc
            elif i == len(trace)-1:
                gen_net.trans_trans_conn(prev_task, curr_task)
                end_arc_id = innovs.check_arc(curr_task, "end")
                end_arc = netobj.GArc(end_arc_id, curr_task, "end")
                gen_net.arcs[end_arc_id] = end_arc
            else:
                # handle concurrency here!
                next_task = trace[i+1]["concept:name"]
                if is_parallel:
                    gen_net.trans_trans_conn(empty_trans, curr_task)
                    is_parallel = False
                elif (curr_task, next_task) in fp_log["parallel"]:
                    is_parallel = True
                    new_place_id = gen_net.new_place(prev_task)
                    new_empty_trans_id = gen_net.new_empty_trans(new_place_id)
                    empty_trans = gen_net.transitions[new_empty_trans_id]
                    gen_net.trans_trans_conn(empty_trans, curr_task)
                else:
                    gen_net.trans_trans_conn(prev_task, curr_task)
            prev_task = curr_task
        new_genomes.append(gen_net)
    return new_genomes

new_genomes = generate_n_start_configs(100, 10, log)

# %% 
for net in new_genomes:
    print("new net")
    net, im, fm = net.build_petri()
    pm4py.view_petri_net(net, im, fm)

# %% currently unused
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