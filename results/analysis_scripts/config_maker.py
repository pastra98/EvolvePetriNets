# %%
import sys # syspath fuckery to run from same location as main
import os # set cwd to same as project cwd from vscode
from pathlib import Path

from pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star import PARAM_SYNC_COST_FUNCTION

cwd = Path.cwd()

# RUN ONLY ONCE
if not os.getcwd().endswith("EvolvePetriNets"): # rename dir on laptop to repo name as well
    sys.path.append(str(cwd.parent.parent / "src")) # src from where all the relative imports work
    os.chdir(cwd.parent.parent) # workspace level from where I execute scripts

# notebook specific - autoreload modules
from IPython import get_ipython
ipython = get_ipython()
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")

# import various other shit
import pickle as pkl
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

from src import neat
from copy import copy
import pprint as pp
import json


# %%
# make a new config folder
class Config():
    def __init__(self, name: str, save_to: str):
        self.name = name
        self.save_to = save_to
        self.base_params = {}
        self.base_cfg = {
            "logpath": "pm_data/running_example.xes",
            "ga_kwargs": {
                "is_pop_serialized": True,
                "is_minimal_serialization": True,
                "is_timed": True
            },
            "stop_cond": {
                "var": "gen",
                "val": 1000
            },
            "send_gen_info_to_console": False,
            "is_profiled": False,
            "save_reduced_history_df": True,
            "save_reports": True,
            "save_params": True,
            "dump_pickle": False
        }
    
    def set_base_params(self, fp_list):
        for fp in fp_list:
            self.add_base_param(fp)

    def add_base_param(self, fp: str):
        with open(fp) as f:
            name = fp.split("/")[-1].rstrip(".json")
            self.base_params[name] = {"base": json.load(f)}

    def get_overview(self):
        for base_param_name in self.base_params:
            print(f"\n{base_param_name}\nchanging values:")
            for change_val in self.base_params[base_param_name]:
                if change_val != "base":
                    print(f"{change_val} - new values:")
                    print(self.base_params[base_param_name][change_val]["values"])
    
    def new_param_list(self, base_params_name: str, param_to_change: str, new_values: list) -> list:
        if base_params_name in self.base_params:
            base = self.base_params[base_params_name]
        else:
            print(f"{base_params_name} base params not found!")

        param_list = []
        for val in new_values:
            param_copy = copy(base["base"])
            param_copy[param_to_change] = val
            param_list.append(param_copy)

        self.base_params[base_params_name][param_to_change] = {
            "values": new_values,
            "params": param_list
        }

    def param_list_for_all_bases(self, param_to_change: str, new_values: list):
        for bp_name in self.base_params:
            self.new_param_list(bp_name, param_to_change, new_values)

    def save_config(self, n_runs: int):
        saves = {}
        for bp_name in self.base_params:
            bp_dict = self.base_params[bp_name]
            for p in bp_dict:
                if p == "base":
                    saves[bp_name] = bp_dict[p]
                else:
                    cp_dict = bp_dict[p]
                    for i, cp in enumerate(cp_dict["params"]):
                        val = cp_dict["values"][i]
                        saves[f"{bp_name}_{p}_{val}"] = cp
        final_cfg = {}
        final_cfg["name"] = self.name
        final_cfg["setups"] = []
        for savename, params in saves.items():
            ppath = f"{self.save_to}/params/{savename}.json"
            with open(ppath, "w") as f:
                json.dump(params, f, indent=4)
            new_setup = copy(self.base_cfg)
            new_setup["setupname"] = savename
            new_setup["parampath"] = ppath
            new_setup["n_runs"] = n_runs
            final_cfg["setups"].append(new_setup)
        with open(f"{self.save_to}/config.json", "w") as f:
            json.dump(final_cfg, f, indent=4)

# def compare_params(p1: dict, p2: dict):
#     changed = {}
#     for key in p1:
#         if p1[key] != p2[key]:
#             changed[key] = {"p1": p1[key], "p2": p2[key]}
#     pp.pprint(changed, indent=4, depth=1)

# setup config for 2x, 3x, 4x
x_cfg = Config("improve_fitness_2", "configs/improve_fitness_2")
bp_paths = [
    "configs/improve_fitness_2/base1x.json",
    "configs/improve_fitness_2/base2x.json",
    "configs/improve_fitness_2/base3x.json",
    "configs/improve_fitness_2/base4x.json"
]
x_cfg.set_base_params(bp_paths)

# set param lists for all
x_cfg.param_list_for_all_bases("perc_fit_traces_weight", [.5, 1, 1.5])
x_cfg.param_list_for_all_bases("soundness_weight", [.5, 1, 1.5])
x_cfg.param_list_for_all_bases("precision_weight", [.5, 1, 1.5])
x_cfg.param_list_for_all_bases("generalization_weight", [.5, 1, 1.5])
x_cfg.param_list_for_all_bases("simplicity_weight", [.5, 1, 1.5])
x_cfg.param_list_for_all_bases("fraction_tasks_weight", [.5, 1, 1.5])
# save with two runs (should be 152 total, for 76 configs)
x_cfg.save_config(2)

# %%
# fitness func on 5 variations

from src.neat import params
import importlib
import json

importlib.reload(params)

base_fp = "configs/improve_fitness/base_config.json"

with open(base_fp, "r") as f:
    base = json.load(f)

base

all_weights = [
    "perc_fit_traces_weight", "average_trace_fitness_weight", "log_fitness_weight",
    "soundness_weight", "precision_weight", "generalization_weight", "simplicity_weight",
    "fraction_used_trans_weight", "fraction_tasks_weight"
]

# "perc_fit_traces_weight", 
# "average_trace_fitness_weight", 
# "log_fitness_weight", 
# "soundness_weight", 
# "fraction_used_trans_weight", 

# %%
import itertools
import json
from copy import copy

possible = [0, 0.5, 1]
all_combis = [list(p) for p in itertools.product(possible, repeat=5)]
final = []
for l in all_combis:
    append = True
    if not l.count(0) in [4,5] and not l.count(1) == 5 and not l.count(0.5) == 5:
        final.append(l)

changeparams = [
    "perc_fit_traces_weight", 
    "average_trace_fitness_weight", 
    "log_fitness_weight", 
    "soundness_weight", 
    "fraction_used_trans_weight", 
]

with open("configs/improve_fitness/base_config.json", "r") as f:
    base = json.load(f)

savedir = "configs/improve_fitness/params"
final_params = []
for values in final:
    new_params = copy(base)
    for ppath, value in zip(changeparams, values):
        new_params[ppath] = value
    fname = savedir + "/" + "_".join([str(v).replace(".", "") for v in values]) + ".json"
    final_params.append(fname)
    with open(fname, "w") as f:
        json.dump(new_params, f, indent=4)

# %%
base_config = {
    "setupname": "",
    "parampath": "",
    "logpath": "pm_data/running_example.xes",
    "ga_kwargs": {
        "is_pop_serialized": True,
        "is_minimal_serialization": True,
        "is_timed": True
    },
    "stop_cond": {
        "var": "gen",
        "val": 1000
    },
    "n_runs": 2,
    "send_gen_info_to_console": False,
    "is_profiled": False,
    "save_reduced_history_df": True,
    "save_reports": True,
    "save_params": True,
    "dump_pickle": False
}

plist = []
for ppath in final_params:
    cfg = copy(base_config)
    cfg["setupname"] = ppath.split("/")[-1].rstrip(".json")
    cfg["parampath"] = ppath
    plist.append(cfg)

final_cfg = {"name": "test_top5_fitness", "setups": plist}
with open("configs/improve_fitness/crazy_config.json", "w") as f:
    json.dump(final_cfg, f, indent=4)