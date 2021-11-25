import json

name: str
vars_not_in_config = ["name"]

def read_file(fname: str):
    """Read in a params json, assign global variables to it
    """
    with open(fname) as f:
        global name
        name = fname
        new_params = json.load(f)
    for k in globals()["__annotations__"]:
        if k in new_params:
            globals()[k] = new_params[k]
        elif k not in vars_not_in_config:
            print(f"Parameter '{k}' missing in json")


def new_param_json(fname: str, save_current=False, savepath="/param_files"):
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

# ---------- GENOME MUTATIONS
prob_t_p: float
prob_t_t: float
prob_p_p: float
prob_new_place: float
prob_split_arc: float
prob_increase_arcs: float
prob_disable_arc: float

num_trys_make_conn: int

# -------------------- connect trans to place
prob_connect_nontask_trans: float
prob_trans_to_place: float

# -------------------- connect trans to trans

# -------------------- split arc
prevent_chaining: bool
num_trys_split_arc: int

# -------------------- connect trans -> new place
prob_pick_dead_trans: float

# ------------------------------------------------------------------------------