import sys, importlib # used for reset

import numpy as np
from numba import njit

from typing import List
from neat.netobj import GArc, GTrans, GPlace
from neat import params

from pm4py.algo.discovery.footprints.algorithm import apply as footprints
from pm4py.objects.conversion.log import converter as log_converter

is_tasks_set = False
fp_log = None

nodes = {"start":GPlace, "end":GPlace}
arcs = {}

# dict containing extended places and trans that are not yet connected to any other nodes
extensions = {} 
extensions_inverted = {}

splits = {}

trans_to_trans = {}

component_dict = {}

curr_genome_id = 0

curr_arc_id = 0
curr_node_id = 0


def reset():
    importlib.reload(sys.modules[__name__])
    return


def get_new_genome_id():
    global curr_genome_id
    curr_genome_id += 1
    return curr_genome_id


def get_trans_to_trans(source_id, target_id):
    """connects a transition to another transition via a place
    """
    global trans_to_trans
    # check if connection already exists
    if (source_id, target_id) in trans_to_trans:
        return trans_to_trans[(source_id, target_id)]
    # if not, create new place and arcs
    else:
        new_place_id = store_new_node(GPlace)
        # should do check arc instead of directly adding
        arc1_id = get_arc(source_id, new_place_id)
        arc2_id = get_arc(new_place_id, target_id)
        trans_to_trans[(source_id, target_id)] = (arc1_id, new_place_id, arc2_id)
        return trans_to_trans[(source_id, target_id)]


def get_extension(extend_from_id: str) -> dict:
    """returns the extension info for a node, if it exists.
    If not, creates a new node"""
    global extensions, extensions_inverted, nodes
    if extend_from_id in extensions:
        return extensions[extend_from_id]
    else:
        # based on nodetype, create new node
        ntype = GPlace if nodes[extend_from_id][0] == GTrans else GTrans
        new_node_id = store_new_node(ntype)
        # in case of extending extension (e.g. in startconfigs), check arc makes sure
        # to delete from_id from extensions. the returned innov id will still be new
        get_arc(extend_from_id, new_node_id)
        new_arc_id = store_new_arc(extend_from_id, new_node_id)
        ext_info = {"arc": new_arc_id, "node": new_node_id, "ntype": ntype}
        # save extension info
        extensions[extend_from_id] = ext_info
        extensions_inverted[new_node_id] = extend_from_id
        return ext_info


def get_split(source, target):
    global splits
    check_tasks_set()
    split_name = f"{source.id}-x->{target.id}"
    if split_name in splits:
        return splits[split_name]
    else:
        new_place_id = store_new_node(GPlace)
        new_trans_id = store_new_node(GTrans)
        # trans -> place: TRANS -ARC1-> NEW_PLACE -ARC2-> NEW_TRANS -ARC3-> PLACE
        if isinstance(source, GTrans) and isinstance(target, GPlace):
            arc1_id = store_new_arc(source.id, new_place_id)
            arc2_id = store_new_arc(new_place_id, new_trans_id)
            arc3_id = store_new_arc(new_trans_id, target.id)
        # place -> trans: PLACE -ARC1-> NEW_TRANS -ARC2-> NEW_PLACE -ARC3-> TRANS
        elif isinstance(source, GPlace) and isinstance(target, GTrans):
            arc1_id = store_new_arc(source.id, new_trans_id)
            arc2_id = store_new_arc(new_trans_id, new_place_id)
            arc3_id = store_new_arc(new_place_id, target.id)
        # store new ids
        split_d = {"p":new_place_id, "t":new_trans_id, "a1":arc1_id, "a2":arc2_id, "a3":arc3_id}
        splits[split_name] = split_d
        return split_d


def get_arc(source_id: int, target_id: int) -> int:
    global arcs, extensions
    check_tasks_set()
    # in case the source is a (so far) unconnected extension, remove it from extensions
    # before connecting to it to target
    if source_id in extensions_inverted:
        ext_key = extensions_inverted[source_id] # get the orig. node that source was extended from
        del extensions[ext_key] # delete extension such that new extensions -> new keys
        del extensions_inverted[source_id] # delete reverse extension
    arc_name = f"{source_id}--->{target_id}"
    if arc_name in arcs:
        return arcs[arc_name]
    else:
        new_id = store_new_arc(source_id, target_id)
        return new_id


def store_new_arc(source_id: int, target_id: int)-> int:
    global curr_arc_id, arcs
    check_tasks_set()
    arc_name = f"{source_id}--->{target_id}"
    curr_arc_id += 1
    arcs[arc_name] = curr_arc_id
    return curr_arc_id


def store_new_node(node_type: type)-> str:
    global curr_node_id, nodes
    # can only store empty transitions and places
    check_tasks_set()
    curr_node_id += 1
    prefix = "t" if node_type == GTrans else "p"
    node_name = prefix + str(curr_node_id)
    nodes[node_name] = (node_type, False)
    return node_name


def set_tasks(log):
    global fp_log, nodes, is_tasks_set
    is_tasks_set = True
    # need to convert to df for this to get the normal fp log
    log = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
    fp_log = footprints(log)
    
    for name in get_task_list():
        nodes[name] = (GTrans, True)


def check_tasks_set():
    global is_tasks_set
    if not is_tasks_set:
        raise Exception("Initial task list must be set!!")


def get_task_list() -> List[str]:
    global fp_log
    return list(fp_log["activities"])


def update_component_fitnesses(population):
    global component_dict
    fitness_components = []
    for g in population:
        # TODO: can later test different fitness metrics here, e.g. just perc_fit_traces
        fitness_components.append((g.fitness, set(g.get_component_set().keys())))
    if params.use_t_vals:
        component_dict = calculate_t_vals(fitness_components)
    # TODO: can do other stuff that influences the probability map here
    # like check the best improvements
    # TODO: maybe use info from more generations later
        

# @njit(parallel=True)
def calculate_t_vals(fitness_components: list) -> dict:
    comp_dict = {}

    # TODO: this logic could also be handled in caller, especially if other
    # keys are added to the dict for other prob_map influences
    pop_fit, comp_dict  = [], {}
    for fc in fitness_components: 
        pop_fit.append(fc[0])
        for component in fc[1]: # assort fitness val with every component
                comp_dict.setdefault(component,
                                        {"all_fitnesses": [],
                                         "t_val": None}
                                    ).get("all_fitnesses").append(fc[0])
    pop_fit = np.array(pop_fit)
    pop_sum, pop_len = pop_fit.sum(), len(pop_fit)
    pop_df = pop_len - 2

    for component in comp_dict:
        included = np.array(comp_dict[component]["all_fitnesses"])
        comp_dict[component]["t_val"] = compute_t(included, pop_fit, pop_len, pop_sum, pop_df)
        # this should only happen in the first gen, when the start and end connections
        # yield components, shared by everyone, making t value comparison impossible
        if len(included) == params.popsize:
            comp_dict[component]["t_val"] = 1 # TODO: revisit this number maybe later
    return comp_dict

# @njit(parallel=True)
def compute_t(inc, pop, pop_len, pop_sum, pop_df):
    inc_sum, inc_len = inc.sum(), len(inc)
    inc_avg_fit = inc_sum / inc_len
    exc_len = pop_len - inc_len
    exc_avg_fit = (pop_sum - inc_sum) / exc_len
    mean_diff = inc_avg_fit - exc_avg_fit

    inc_df, exc_df = inc_len-1, exc_len-1

    inc_ss = ((inc - inc_avg_fit)**2).sum()
    inc_var = inc_ss / inc_len
    
    exc_ss = ((pop - exc_avg_fit)**2).sum() - inc_ss - inc_len*mean_diff**2
    exc_var = exc_ss / exc_len

    pool_var = (inc_var*inc_df + exc_var*exc_df) / pop_df
    se = (pool_var/inc_len + pool_var/exc_len)**0.5
    return float(mean_diff/se)
