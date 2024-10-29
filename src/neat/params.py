import json
from typing import Dict, List, Union, TypedDict

name = None

def load(fpath: str):
    """Read in a params json, assign global variables to it
    """
    global name
    name = fpath
    with open(fpath) as f:
        new_params = json.load(f)
    for k in globals()["__annotations__"]:
        if k in new_params:
            globals()[k] = new_params[k]
        else:
            print(f"Parameter '{k}' missing in json")


def new_param_json(fpath: str, save_current=False):
    """Save list of variables, along with a type hint
    """
    vars = globals()["__annotations__"]
    export_vars = {}
    for k, v in list(vars.items()):
        if save_current and k in globals():
            export_vars[k] = globals()[k]
        else:
            export_vars[k] = str(v)
    with open(fpath, "w") as f:
        json.dump(export_vars, f, indent=4)
        print(f"saved params at:\n{fpath}")


def get_curr_curr_dict() -> dict:
    export_vars = {}
    for k in globals()["__annotations__"]:
        export_vars[k] = globals()[k]
    return export_vars


# ---------- GENERAL GA SETTINGS
selection_strategy: str # "speciation" / "roulette" / "truncation"

# -------------------- population setup
popsize: int
n_alpha_genomes: int
n_inductive_genomes: int
n_heuristics_genomes: int
n_ilp_genomes: int

# -------------------- start config: random
connect_sa_ea: bool # should start/end activities be connected to source/sink
initial_tp_gauss_dist: list[float, float]
initial_pt_gauss_dist: list[float, float]
initial_tt_gauss_dist: list[float, float]
initial_pe_gauss_dist: list[float, float]
initial_te_gauss_dist: list[float, float]
initial_as_gauss_dist: list[float, float]


# -------------------- selection strategies: SHARED PARAMS
# used by all 3 selection strategies
pop_perc_crossover: float # percentage of population spawns that will be crossover
start_crossover: int # [0-1], after what generation start crossover

# used by roulette and truncation selection
pop_perc_elite: float # percentage of population spawns that will be crossover

# used by speciation and trunctation
spawn_cutoff: float # either determines cutoff within species mating pool or entire pop

# -------------------- selection strategy: SPECIATION
species_boundary: float

species_component_pool_size: int # how many (randomly chosen) species membrs contrib their comp to species comp pool
tournament_size: int # for crossover, how big should tournament be that selects 2 parents

# species relevant stuff
enough_gens_to_change_things: int
update_species_rep: bool
leader_is_rep: bool
elitism: bool

allowed_gens_no_improvement: int
old_age: int
old_penalty: float
youth_bonus: float

# ---------- FITNESS CHECK
# token replay mult/penalties
replay_mult: float # for every successive full points transition execution, mult doubles
missing_penal: float # is subtracted from replay pts at ratio (consumed / missing)

# for min token use metric, value is user-defined but can probably be calculated in a smart way
min_tokens_for_replay: int # 51 for simple running_example log

# other fitness measures
class MetricParams(TypedDict):
    weight: float # weight to multiply metric with
    raise_by: float # metric is raised by that, default should be 1
    active_gen: int # in which generation start adding that metric
    # anchor_to[0]: key to metric, default "". If specified, only add fitness if other metric reaches anchor_to[1]
    anchor_to: List[Union[str, float]] 

metric_dict: Dict[str, MetricParams] # metric: MetricParams


# ---------- MUTATIONS GENERAL
# -------------------- guiding the mutations
use_t_vals: bool

"""
Parameters in list depend on the Mutation Rate, which is either 0 (normal) or 1 (high)
When using atomic mutations, the same probabilities are used, however they are now
weights in a singular random choice. This changes their influence and should be considered
when atomic mutations are used.
"""
# arc mutations
prob_remove_arc: list[float, float]
# Make a new arc
prob_t_p_arc: list[float, float]
prob_p_t_arc: list[float, float]
# connect trans to trans
prob_t_t_conn: list[float, float]
# extend to new node or trans
prob_new_p: list[float, float]
prob_new_empty_t: list[float, float]
# split an arc
prob_split_arc: list[float, float]
# prune extensions
prob_prune_leafs: list[float, float]
# flip arc
prob_flip_arc: list[float, float]

is_no_preference_for_tasks: bool # if this is True, prob pick_tasks_trans is ignored
prob_pick_empty_trans: float # probability of picking a task transition

# ------------------------------------------------------------------------------