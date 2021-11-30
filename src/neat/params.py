import json

name: str
vars_not_in_config = ["name"]

def load(fname: str, savepath="param_files/"):
    """Read in a params json, assign global variables to it
    """
    savepath = savepath + fname + ".json"
    with open(savepath) as f:
        global name
        name = fname
        new_params = json.load(f)
    for k in globals()["__annotations__"]:
        if k in new_params:
            globals()[k] = new_params[k]
        elif k not in vars_not_in_config:
            print(f"Parameter '{k}' missing in json")

def new_param_json(fname: str, save_current=False, savepath="param_files/"):
    """Save list of variables, along with a type hint
    """
    savepath = savepath + fname + ".json"
    vars = globals()["__annotations__"]
    export_vars = {}
    for k, v in list(vars.items()):
        if k not in vars_not_in_config:
            if save_current and k in globals():
                export_vars[k] = globals()[k]
            else:
                export_vars[k] = str(v)
    with open(savepath, "w") as f:
        json.dump(export_vars, f, indent=4)
        print(f"saved params at:\n{savepath}")

# ---------- GENERAL GA SETTINGS
start_config: str
popsize: int
selection_strategy: str # "speciation" / "roulette" / "truncation"

# -------------------- selection strategy: SPECIATION
species_boundary: float
coeff_matched: float
coeff_disjoint: float
coeff_excess: float

# species relevant stuff
enough_gens_to_change_things: int
update_species_rep: bool
leader_is_rep: bool
selection_threshold: int
spawn_cutoff: float

old_age: int
old_penalty: float
youth_bonus: float
random_mating: bool
# -------------------- selection strategy: ROULETTE 
# -------------------- selection strategy: TRUNCATION

# ---------- FITNESS CHECK
perc_fit_traces_weight: float
soundness_weight: float
precision_weight: float
generalization_weight: float
simplicity_weight: float

# ---------- MUTATIONS GENERAL
"""
Parameters in list depend on the Mutation Rate, which is either 0 (normal) or 1 (high)
"""
prob_t_p = [float, float]
prob_t_t = [float, float]
prob_p_p = [float, float]
prob_new_place = [float, float]
prob_split_arc = [float, float]
prob_increase_arcs = [float, float]
prob_disable_arc = [float, float]

num_trys_make_conn: int
# -------------------- connect trans to place
prob_connect_nontask_trans = [float, float]
prob_trans_to_place = [float, float]

# -------------------- connect trans to trans

# -------------------- split arc
prevent_chaining: bool
num_trys_split_arc: int

# -------------------- connect trans -> new place
prob_pick_dead_trans: float

# ------------------------------------------------------------------------------