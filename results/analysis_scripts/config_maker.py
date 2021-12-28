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

# %%
# read a params file into memory
from src.neat import params
import importlib
importlib.reload(params)

def save_new_params(old_name, new_name, save_current=True):
    params.load(old_name)
    print(params.popsize)
    params.new_param_json(new_name, save_current=True)


# %%
# make a new config folder
class config():
    def __init__(self, name: str):
        self.name = name
        self.base_params = {}
    
    def set_base_params():
        pass

    def add_base_param():
        pass

    def get_overview(self):
        for base_param_name in self.base_params:
            pass
    
    def new_param_list(self, base_params_name: str, param_to_change: str, new_values: list) -> list:
        if base_params_name in self.base_params:
            base = self.base_params[base_params_name]
        else:
            print(f"{base_params_name} base params not found!")

        param_list = []
        for val in new_values:
            param_copy = copy(base)
            param_copy[param_to_change] = val
            param_list.append(param_copy)

        self.base_params[base_params_name][param_to_change] = {
            "values": new_values,
            "params": param_list
        }

def compare_params(p1: dict, p2: dict):
    changed = {}
    for key in p1:
        if p1[key] != p2[key]:
            changed[key] = {"p1": p1[key], "p2": p2[key]}
    pp.pprint(changed, indent=4, depth=1)
