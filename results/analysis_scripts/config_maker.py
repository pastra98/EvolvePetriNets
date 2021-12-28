# %%
import sys # syspath fuckery to run from same location as main
import os # set cwd to same as project cwd from vscode
from pathlib import Path

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

# %%
# read a params file into memory
from src.neat import params
import importlib
importlib.reload(params)

def save_new_params(old_name, new_name, save_current=True):
    params.load(old_name)
    print(params.popsize)
    params.new_param_json(new_name, save_current=True)