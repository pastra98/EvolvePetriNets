# %%
import sys # syspath fuckery to run from same location as main
import os # set cwd to same as project cwd from vscode
import subprocess # for sending stuff to os and printing out
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
from src.tests import visualize_genome as vg
import matplotlib.pyplot as plt
import matplotlib as mpl
from statistics import fmean
from src import neat

# %%
# list available files
files = [f"results/data/{f}" for f in os.listdir("results/data")]
print(f"files in data:")
print('\n'.join(files))


# %% test loading from one run
# # tree_out = subprocess.getoutput(f"tree /F {analysis_fp}")

# tree_out = subprocess.getoutput("tree /F results/data/test_multiproc_12-26-2021_09-20-04")
# print(tree_out)

# print(f"tree /F {analysis_fp}")

# %%
hist = pd.read_feather('results/data/test_multiproc_12-26-2021_10-02-00/t1/1_12-26-2021_10-02-00/reports/history.feather')
pop = pd.read_feather('results/data/test_multiproc_12-26-2021_10-02-00/t1/1_12-26-2021_10-02-00/reports/population.feather')
species = pd.read_feather('results/data/test_multiproc_12-26-2021_10-02-00/t1/1_12-26-2021_10-02-00/reports/species.feather')

# %%