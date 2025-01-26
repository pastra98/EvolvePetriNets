from random import gauss
from neat import genome, params

from pm4py.objects.petri_net.obj import PetriNet as pn
from pm4py.discovery import (
    discover_petri_net_alpha as alpha,
    discover_petri_net_alpha_plus as alpha_plus,
    discover_petri_net_inductive as inductive,
    discover_petri_net_heuristics as heuristics,
    discover_petri_net_ilp as ilp
)


def create_initial_pop(log, component_tracker) -> list:
    """Creates an initial population consisting of random genomes and bootstrapped
    genomes (if specified).
    """
    bootstrap_g = get_bootstrap_genomes(log, component_tracker)
    n_random_g = params.popsize - len(bootstrap_g)

    if n_random_g < 0:
        raise Exception("Number of bootstrap genomes should not exceed population size")

    random_g = get_random_genomes(n_random_g, log, component_tracker)
    return bootstrap_g + random_g


def get_random_genomes(n_genomes, log, component_tracker):
    # generate n random genomes
    new_genomes = []
    for _ in range(n_genomes):
        gen_net = genome.GeneticNet(
            transitions = dict(),
            places = dict(),
            arcs = dict(),
            task_list=log["task_list"],
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

        # connect start/end activities to source/sink
        if params.connect_sa_ea:
            # sets of all start and end activities
            start_act = set(log["footprints"]["start_activities"])
            end_act = set(log["footprints"]["end_activities"])
            # filter out all start/end activities that are already connected to source/sink
            for a in gen_net.arcs.values():
                if a.source_id == "start" and a.target_id in start_act:
                    start_act.remove(a.target_id)
                elif a.source_id in end_act and a.target_id == "end":
                    end_act.remove(a.source_id)
            # if there are still unconnected sa/ea, connect them to source and sink
            for sa in start_act:
                gen_net.place_trans_arc("start", sa)
            for ea in end_act:
                gen_net.trans_place_arc(ea, "end")

        gen_net.my_mutation = ""
        new_genomes.append(gen_net)

    return new_genomes


def get_bootstrap_genomes(log, component_tracker):
    bootstrap_setup = {
        alpha: params.n_alpha_genomes,
        # alpha_plus: params.n_alpha_genomes,
        inductive: params.n_inductive_genomes,
        heuristics: params.n_heuristics_genomes,
        ilp: params.n_ilp_genomes
    }
    mined_nets = []
    for miner, count in bootstrap_setup.items():
        if not count:
            continue
        net, im, fm = miner(log["dataframe"])
        g = construct_genome_from_mined_net(net, im, fm, log["task_list"], component_tracker)
        for _ in range(count):
            mined_nets.append(g.clone(self_is_parent=False))
    return mined_nets


def construct_genome_from_mined_net(net, im, fm, tl, ct):
    g = genome.GeneticNet(dict(), dict(), dict(), task_list=tl, pop_component_tracker=ct)

    # different miners call source/sink by different names - this is hacky crap
    place_dict = {
        "source":"start", "start":"start", "source0":"start",
        "sink":"end", "end":"end", "sink0":"end"
        }
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
            t.label = t.name
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
