# %% imports and read log
import random as rd

import pm4py
from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
from pm4py.visualization.footprints import visualizer as fp_visualizer

import netobj
import innovs
import genome
import params

logpath = "./pm_data/m1_log.xes"
# logpath = "../pm_data/BPI_Challenge_2012.xes"
# logpath = "../pm_data/pdc_2016_6.xes"
# logpath = "../pm_data/running_example.xes"
# logpath = "../pm_data/simulated_running_example.xes"
log = pm4py.read_xes(logpath)

# parameters
params.read_file("./neat/param_files/test_params.json")

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
    # start traces loop --------------------------------------------------------
    for trace in log:
        net_id += 1
        gen_net = genome.GeneticNet(net_id, task_dict, dict(), dict())
        # start task loop ------------------------------------------------------
        for i, task in enumerate(trace):
            parallels = []
            curr_task_id = task["concept:name"]
            # first task
            if i == 0:
                start_arc_id = innovs.check_arc("start", curr_task_id)
                start_arc = netobj.GArc(start_arc_id, "start", curr_task_id)
                gen_net.arcs[start_arc_id] = start_arc
            # last task
            elif i == len(trace)-1:
                gen_net.trans_trans_conn(prev_task_id, curr_task_id)
                end_arc_id = innovs.check_arc(curr_task_id, "end")
                end_arc = netobj.GArc(end_arc_id, curr_task_id, "end")
                gen_net.arcs[end_arc_id] = end_arc
            # middle task
            else:
                next_task = trace[i+1]["concept:name"]
                # get task before parallel
                if ((prev_task_id, curr_task_id) not in fp_log["parallel"]) and \
                    ((curr_task_id, next_task) in fp_log["parallel"]):
                    task_before_para = prev_task_id
                # next task is parallel
                if (curr_task_id, next_task) in fp_log["parallel"]:
                    parallels.append(curr_task_id)
                # end of parallel construct, build it
                elif (prev_task_id, curr_task_id) in fp_log["parallel"]:
                    parallels.append(curr_task_id)
                    # take first parallel task, build parallel structure
                    first_para_task_id = parallels.pop()
                    #  use it to create end trans
                    start_place_id = gen_net.new_place(task_before_para)
                    start_trans_id = gen_net.new_empty_trans(start_place_id)
                    start_trans = gen_net.transitions[start_trans_id]
                    gen_net.trans_trans_conn(start_trans_id, first_para_task_id)
                    #  use it to create end trans (it is already conn to first_para_task!)
                    end_place_id = gen_net.new_place(first_para_task_id)
                    end_trans_id = gen_net.new_empty_trans(end_place_id)
                    end_trans = gen_net.transitions[end_trans_id]
                    # build remaining parallels
                    for task_id in parallels:
                        gen_net.trans_trans_conn(start_trans_id, task_id)
                        gen_net.trans_trans_conn(task_id, end_trans_id)
                    # parallel structure over now
                # connect to empty trans at end of parallel construct
                elif parallels:
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

new_genomes = generate_n_start_configs(100, 10, log)

# %% 
def get_trace_str(trace):
    tr_events = []
    for event in trace:
        tr_events.append(event["concept:name"])
    return " -> ".join(tr_events)


for net, trace in new_genomes:
    print(get_trace_str(trace))
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
################################################################################
                    # if (prev_task, curr_task) in fp_log["parallel"]: