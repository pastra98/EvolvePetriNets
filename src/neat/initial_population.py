from random import gauss
from neat import genome, params
from pm4py.algo.discovery.footprints.algorithm import apply as footprints
from pm4py.objects.conversion.log import converter as log_converter

def get_log_footprints(log) -> list:
    log = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
    return footprints(log)


# TODO - this can be improved
def generate_n_random_genomes(n_genomes, log, component_tracker):
    fp_log = get_log_footprints(log)
    tl = [a for a in fp_log["activities"]]
    # generate n random genomes
    new_genomes = []
    for _ in range(n_genomes):
        gen_net = genome.GeneticNet(
            transitions = dict(),
            places = dict(),
            arcs = dict(),
            task_list=tl,
            pop_component_tracker = component_tracker
            )

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
        # TODO: remove this cheating later
        # connect all start and end activities to start and end - debateable
        has_start_conn, has_end_conn = False, False
        sa = list(fp_log["start_activities"])[0]
        ea = list(fp_log["end_activities"])[0]
        for a in gen_net.arcs.values():
            if a.source_id == "start" and a.target_id == sa:
                has_start_conn = True
            elif a.source_id == ea and a.target_id == "end":
                has_end_conn = True
        if not has_start_conn:
            gen_net.place_trans_arc("start", sa)
        if not has_end_conn:
            gen_net.trans_place_arc(ea, "end")

    return new_genomes
