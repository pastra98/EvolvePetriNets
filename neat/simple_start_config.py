import random as rd

import pm4py
from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
from pm4py.visualization.footprints import visualizer as fp_visualizer
import params

import netobj
import innovs
import genome
import params

# logpath = "../pm_data/m1_log.xes"
# logpath = "../pm_data/BPI_Challenge_2012.xes"
# logpath = "../pm_data/pdc_2016_6.xes"
# logpath = "../pm_data/running_example.xes"
# logpath = "../pm_data/simulated_running_example.xes"
# log = pm4py.read_xes(logpath)
# # parameters
# params.read_file("../neat/param_files/test_params.json")

def get_trace_str(trace):
    tr_events = []
    for event in trace:
        tr_events.append(event["concept:name"])
    return " -> ".join(tr_events)


def footprints(log, visualize=True, printit=True):
    fp_log = footprints_discovery.apply(log, variant=footprints_discovery.Variants.ENTIRE_EVENT_LOG)
    if visualize:
        gviz = fp_visualizer.apply(fp_log, parameters={fp_visualizer.Variants.SINGLE.value.Parameters.FORMAT: "png"})
        fp_visualizer.view(gviz)
    if printit:
        for relation in fp_log:
            print(f"{relation}\n{fp_log[relation]}\n")
    return fp_log

def generate_n_start_configs(n_genomes, n_arcs, log):
    fp_log = footprints(log, visualize=False, printit=False)
    task_list = list(fp_log["activities"])
    innovs.set_tasks(task_list)
    task_dict = dict()
    for task in task_list:
        task_dict[task] = netobj.GTrans(task, True)
    # generate genomes
    new_genomes = []
    net_id = 0
    # start traces loop --------------------------------------------------------
    for trace in log:
        net_id += 1
        gen_net = genome.GeneticNet(net_id, task_dict, dict(), dict())
        # start task loop ------------------------------------------------------
        parallels = []
        for i, task in enumerate(trace):
            curr_task_id = task["concept:name"]
            # first task
            if i == 0:
                start_arc_id = innovs.check_arc("start", curr_task_id)
                start_arc = netobj.GArc(start_arc_id, "start", curr_task_id)
                gen_net.arcs[start_arc_id] = start_arc
            # last task
            elif i == len(trace)-1:
                if parallels:
                    gen_net.trans_trans_conn(end_trans_id, curr_task_id)
                else:
                    gen_net.trans_trans_conn(prev_task_id, curr_task_id)
                end_arc_id = innovs.check_arc(curr_task_id, "end")
                end_arc = netobj.GArc(end_arc_id, curr_task_id, "end")
                gen_net.arcs[end_arc_id] = end_arc
            # middle task
            else:
                next_task_id = trace[i+1]["concept:name"]
                is_prev_pair_para = (prev_task_id, curr_task_id) in fp_log["parallel"]
                is_next_pair_para = (curr_task_id, next_task_id) in fp_log["parallel"]
                # get task before parallel
                if not is_prev_pair_para and is_next_pair_para:
                    task_before_para = prev_task_id
                # next task is parallel
                if is_next_pair_para:
                    parallels.append(curr_task_id)
                # end of parallel construct, build it
                elif is_prev_pair_para and not is_next_pair_para:
                    parallels.append(curr_task_id)
                    # take first parallel task, build parallel structure
                    first_para_task_id = parallels.pop(0)
                    #  use it to create start trans, connect to it
                    start_place_id = gen_net.new_place(task_before_para)
                    start_trans_id = gen_net.new_empty_trans(start_place_id)
                    gen_net.trans_trans_conn(start_trans_id, first_para_task_id)
                    # create end trans (it is already conn to first_para_task!)
                    end_place_id = gen_net.new_place(first_para_task_id)
                    end_trans_id = gen_net.new_empty_trans(end_place_id)
                    # build remaining parallels
                    for task_id in parallels:
                        # print(f"{task_before_para} -> {task_id}")
                        # print(f"{task_id} -> {next_task}")
                        gen_net.trans_trans_conn(start_trans_id, task_id)
                        gen_net.trans_trans_conn(task_id, end_trans_id)
                    # parallel structure over now
                # connect to empty trans at end of parallel construct
                elif parallels and not is_prev_pair_para and not is_next_pair_para:
                    gen_net.trans_trans_conn(end_trans_id, curr_task_id)
                    parallels.clear()
                # just normal connect to prev_task
                else:
                    gen_net.trans_trans_conn(prev_task_id, curr_task_id)
            prev_task_id = curr_task_id
        # end task loop --------------------------------------------------------
        new_genomes.append([gen_net, trace])
    # end traces loop ----------------------------------------------------------
    return new_genomes
