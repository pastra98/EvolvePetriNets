from random import gauss
from neat import genome, params
from pm4py.objects.petri_net.obj import PetriNet as pn
from pm4py.algo.discovery.footprints.algorithm import apply as footprints
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.discovery import (
    discover_petri_net_alpha as alpha,
    discover_petri_net_inductive as inductive,
    discover_petri_net_heuristics as heuristics,
    discover_petri_net_ilp as ilp
)


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


def get_bootstrapped_population(n_genomes, log, component_tracker):
    """This is just the simplest implementation to test how the fitness func
    will deal with mined nets
    """
    fp_log = get_log_footprints(log)
    tl = [a for a in fp_log["activities"]]
    mined_nets = []
    # miners = [alpha, inductive, heuristics, ilp]
    miners = [alpha]
    for miner in miners:
        net, im, fm = miner(log)
        g = construct_genome_from_mined_net(net, im, fm, tl, component_tracker)
        for _ in range(int(n_genomes/len(miners))):
            mined_nets.append(g.clone(self_is_parent=False))
    # if rounding errors lead to len(mined_nets) != n_genomes
    delta = n_genomes - len(mined_nets)
    if delta > 0:
        mined_nets += [g.clone(self_is_parent=False) for _ in range(delta)]
    elif delta < 0:
        mined_nets = mined_nets[:n_genomes]
    return mined_nets


def construct_genome_from_mined_net(net, im, fm, tl, ct):
    g = genome.GeneticNet(dict(), dict(), dict(), task_list=tl, pop_component_tracker=ct)
    place_dict = {"source":"start", "start":"start", "sink":"end", "end":"end"}
    trans_dict = {t:t for t in tl} # map t.label to genome id
    
    for p in net.places:
        # if there are multiple start/end places, they will all be treated like one
        # meaning all their connections are just in one place
        if p.name not in place_dict.keys():
            new_id = g.add_new_place()
            place_dict[p.name] = new_id

    for t in net.transitions:
        if t.label not in tl:
            new_id = g.add_new_trans()
            trans_dict[t.label] = new_id

    for a in net.arcs:
        if type(a.source) == pn.Place:
            p_id = place_dict[a.source.name]
            t_id = trans_dict[a.target.label]
            g.add_new_arc(p_id, t_id)
        else:
            t_id = trans_dict[a.source.label]
            p_id = place_dict[a.target.name]
            g.add_new_arc(t_id, p_id)
    
    return g
