# %%
import sys # syspath fuckery to run from same location as main
import os # set cwd to same as project cwd from vscode
from pathlib import Path
from matplotlib import pyplot as plt

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

import pickle
import pandas as pd

# %%
def load_pkl(fp: str) -> pd.DataFrame:
    with open(fp, "rb") as f:
        return pickle.load(f)

r0_mem = load_pkl("results/data/test_muppy_12-24-2021_17-27-59/speciation_test_0___12-24-2021_17-27-59/speciation_test_0___12-24-2021_17-27-59_mem.pkl")
r0_mem.sort_values(by="memory", ascending=False, inplace=True)

r1_mem = load_pkl("results/data/test_muppy_12-24-2021_17-27-59/speciation_test_1___12-24-2021_17-29-48/speciation_test_1___12-24-2021_17-29-48_mem.pkl")
r1_mem.sort_values(by="memory", ascending=False, inplace=True)

r2_mem = load_pkl("results/data/test_muppy_12-24-2021_17-27-59/speciation_test_2___12-24-2021_17-31-39/speciation_test_2___12-24-2021_17-31-39_mem.pkl")
r2_mem.sort_values(by="memory", ascending=False, inplace=True)

r3_mem = load_pkl("results/data/test_muppy_12-24-2021_17-27-59/speciation_test_3___12-24-2021_17-33-29/speciation_test_3___12-24-2021_17-33-29_mem.pkl")
r3_mem.sort_values(by="memory", ascending=False, inplace=True)

r4_mem = load_pkl("results/data/test_muppy_12-24-2021_17-27-59/speciation_test_4___12-24-2021_17-35-23/speciation_test_4___12-24-2021_17-35-23_mem.pkl")
r4_mem.sort_values(by="memory", ascending=False, inplace=True)

end_mem = load_pkl("results/data/test_muppy_12-24-2021_17-27-59/overall_mem.pkl")
end_mem.sort_values(by="memory", ascending=False, inplace=True)

all_reps = [r0_mem, r1_mem, r2_mem, r3_mem, r4_mem, end_mem]

totals = [rep["memory"].sum() / 1_000 for rep in all_reps]
deltas = [totals[i] - totals[i-1] for i in range(len(totals))][1:]

top1_mems = [rep.iloc[0]["memory"] for rep in all_reps]
top1_deltas = [top1_mems[i] - top1_mems[i-1] for i in range(len(top1_mems))][1:]

# %%
def print_biggest_changes(reps):
    pass

print_biggest_changes(all_reps)
# %%
plt.plot(totals)

# %%
end_mem
