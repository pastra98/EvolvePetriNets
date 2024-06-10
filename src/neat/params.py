import json

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
start_config: str
popsize: int
selection_strategy: str # "speciation" / "roulette" / "truncation"
log_splices: dict

# -------------------- start config: random
initial_tp_gauss_dist: list[float, float]
initial_pt_gauss_dist: list[float, float]
initial_tt_gauss_dist: list[float, float]
initial_pe_gauss_dist: list[float, float]
initial_te_gauss_dist: list[float, float]
initial_as_gauss_dist: list[float, float]

# -------------------- selection strategy: SPECIATION
distance_metric: str # "innovs" / "behavior" / "components"
species_boundary: float

# component multiplier, if distance_metric == "components" increase importance of shared components
component_mult: float

# species relevant stuff
enough_gens_to_change_things: int
update_species_rep: bool
leader_is_rep: bool
selection_threshold: int
spawn_cutoff: float # actually also used by truncation_pop_update
elitism: bool

allowed_gens_no_improvement: int
old_age: int
old_penalty: float
youth_bonus: float
# -------------------- selection strategy: ROULETTE 
# -------------------- selection strategy: TRUNCATION

# ---------- FITNESS CHECK
# stuff that is extracted from alignment info
perc_fit_traces_weight: float
average_trace_fitness_weight: float
log_fitness_weight: float
# other fitness measures
soundness_weight: float
precision_weight: float
generalization_weight: float
simplicity_weight: float
fraction_used_trans_weight: float
fraction_tasks_weight: float
t_exec_scoring_weight: list

# ---------- MUTATIONS GENERAL
mutation_type: str # "multi" / "atomic"

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
max_arcs_removed: int # in case of atomic mutation, this is set automatically to 1
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

is_no_preference_for_tasks: bool # if this is True, prob pick_tasks_trans is ignored
prob_pick_empty_trans: float # probability of picking a task transition

# ------------------------------------------------------------------------------