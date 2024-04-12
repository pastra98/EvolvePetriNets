from random import gauss
from pm4py.algo.discovery.footprints.algorithm import apply as footprints
from neat import innovs, genome, params

# TODO - this can be improved
def generate_n_random_genomes(n_genomes, log):
    # get footprints needed to get the task list
    fp_log = footprints(log, visualize=False, printit=False)
    task_list = list(fp_log["activities"])
    innovs.set_tasks(task_list)
    # generate n random genomes
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
        # connect all start and end activities to start and end - debatable
        for sa in list(fp_log["start_activities"]):
            gen_net.place_trans_arc("start", sa)
        for ea in list(fp_log["end_activities"]):
            gen_net.trans_place_arc(ea, "end")

    return new_genomes
