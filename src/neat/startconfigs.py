from random import gauss
from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
from pm4py.visualization.footprints import visualizer as fp_visualizer

from . import netobj, innovs, genome, params

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

def generate_n_traces_with_concurrency(n_genomes, log):
    base_genomes = traces_with_concurrency(log)
    new_genomes = []
    while len(new_genomes) < n_genomes:
        for new_genome in base_genomes:
            new_genomes.append(new_genome.clone())
            if len(new_genomes) == n_genomes:
                break
    return new_genomes

def traces_with_concurrency(log):
    fp_log = footprints(log, visualize=False, printit=False)
    task_list = list(fp_log["activities"])
    innovs.set_tasks(task_list)
    # generate genomes
    new_genomes = []
    # start traces loop --------------------------------------------------------
    for trace in log:
        # task dict with fresh genes for each genome
        gen_net = genome.GeneticNet(dict(), dict(), dict())
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
                    start_place_id = gen_net.extend_new_place(task_before_para)
                    start_trans_id = gen_net.extend_new_trans(start_place_id)
                    gen_net.trans_trans_conn(start_trans_id, first_para_task_id)
                    # create end trans (it is already conn to first_para_task!)
                    end_place_id = gen_net.extend_new_place(first_para_task_id)
                    end_trans_id = gen_net.extend_new_trans(end_place_id)
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
        new_genomes.append(gen_net)
    # end traces loop ----------------------------------------------------------
    return new_genomes

def generate_n_random_genomes(n_genomes, log):
    #
    fp_log = footprints(log, visualize=False, printit=False)
    task_list = list(fp_log["activities"])
    innovs.set_tasks(task_list)
    #
    new_genomes = []
    for _ in range(n_genomes):
        gen_net = genome.GeneticNet(dict(), dict(), dict())
        for _ in range(int(abs(gauss(*params.initial_tp_gauss_dist)))):
            gen_net.trans_place_arc()
        for _ in range(int(abs(gauss(*params.initial_pt_gauss_dist)))):
            gen_net.place_trans_arc()
        for _ in range(int(abs(gauss(*params.initial_tt_gauss_dist)))):
            gen_net.trans_trans_conn()
        for _ in range(int(abs(gauss(*params.initial_pe_gauss_dist)))):
            gen_net.extend_new_place()
        for _ in range(int(abs(gauss(*params.initial_te_gauss_dist)))):
            gen_net.extend_new_trans()
        for _ in range(int(abs(gauss(*params.initial_as_gauss_dist)))):
            gen_net.split_arc()
        new_genomes.append(gen_net)
        ## hacky shit
        for sa in list(fp_log["start_activities"]):
            gen_net.place_trans_arc("start", sa)
        # for ea in list(fp_log["end_activities"]):
        #     gen_net.trans_place_arc(ea, "end")

        ### even hackier shit
        gen_net.trans_place_arc("pay compensation", "end")
        ### even hackier shit

        ## hacky shit

    return new_genomes


# logpath = "../pm_data/m1_log.xes"
# logpath = "../pm_data/BPI_Challenge_2012.xes"
# logpath = "../pm_data/pdc_2016_6.xes"
# logpath = "../pm_data/running_example.xes"
# logpath = "../pm_data/simulated_running_example.xes"
# log = pm4py.read_xes(logpath)
# # parameters
# params.load("../neat/param_files/test_params.json")

def test_startconf():
    trans_d = {}
    innovs.set_tasks(["A", "B", "C", "D"])
    for t in innovs.tasks:
        trans_d[t] = netobj.GTrans(t, True)
    gen_net = genome.GeneticNet(trans_d, dict(), dict())

    start_arc = netobj.GArc(0, "start", "A")
    gen_net.arcs[0] = start_arc

    end_arc = netobj.GArc(1, "D", "end")
    gen_net.arcs[1] = end_arc

    return gen_net
