import sys, importlib # used for reset
import numpy as np

from numba import njit
from copy import copy
from typing import List

from neat.genome import GTrans, GPlace
from neat import params

from pm4py.algo.discovery.footprints.algorithm import apply as footprints
from pm4py.objects.conversion.log import converter as log_converter

is_tasks_set = False
fp_log = None

nodes = {"start":GPlace, "end":GPlace}
arcs = dict()

# dict containing extended places and trans that are not yet connected to any other nodes
component_dict = dict()
component_history = dict()

curr_genome_id = 0


def reset():
    importlib.reload(sys.modules[__name__])
    return


def get_new_genome_id():
    global curr_genome_id
    curr_genome_id += 1
    return curr_genome_id


def set_tasks(log):
    global fp_log, nodes, is_tasks_set
    is_tasks_set = True
    # need to convert to df for this to get the normal fp log
    log = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
    fp_log = footprints(log)
    
    for name in get_task_list():
        nodes[name] = (GTrans, True)


def get_task_list() -> List[str]:
    global fp_log
    return list(fp_log["activities"])


def update_component_fitnesses(population: list, generation: int):
    global component_dict
    # for the old generation, add the components to the history
    for c in component_dict:
        if c not in component_history:
            component_history[c] = generation
    # pair the components of the new population with their fitness values
    fitness_components = []
    for g in population:
        # TODO: can later test different fitness metrics here, e.g. just perc_fit_traces
        fitness_components.append((g.fitness, g.get_unique_component_set()))
    if params.use_t_vals:
        component_dict = calculate_t_vals(fitness_components)
    # TODO: can do other stuff that influences the probability map here
    # like check the best improvements
    # TODO: maybe use info from more generations later
        

# @njit(parallel=True)
def calculate_t_vals(fitness_components: list) -> dict:
    comp_fitness_dict = {}

    # TODO: this logic could also be handled in caller, especially if other
    # keys are added to the dict for other prob_map influences
    pop_fit, comp_fitness_dict  = [], {}
    for fc in fitness_components: 
        pop_fit.append(fc[0])
        for component in fc[1]: # assort fitness val with every component
                comp_fitness_dict.setdefault(component,
                                        {"all_fitnesses": [],
                                         "t_val": None}
                                    ).get("all_fitnesses").append(fc[0])
    pop_fit = np.array(pop_fit)
    pop_sum, pop_len = pop_fit.sum(), len(pop_fit)
    pop_df = pop_len - 2

    for component in comp_fitness_dict:
        included = np.array(comp_fitness_dict[component]["all_fitnesses"])
        comp_fitness_dict[component]["t_val"] = compute_t(included, pop_fit, pop_len, pop_sum, pop_df)
        # this should only happen in the first gen, when the start and end connections
        # yield components, shared by everyone, making t value comparison impossible
        if len(included) == params.popsize:
            comp_fitness_dict[component]["t_val"] = 1 # TODO: revisit this number maybe later
    return comp_fitness_dict

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
