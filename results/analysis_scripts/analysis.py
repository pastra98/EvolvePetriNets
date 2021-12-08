# %%
import sys # syspath fuckery to run from same location as main
import os # set cwd to same as project cwd from vscode

srcpath = "C:\\Users\\pauls\\OneDrive\\Dokumente\\Uni\\Fächer\\BSc\\GeneticProcessMining\\genetic_miner\\src"
rootpath = "C:\\Users\\pauls\\OneDrive\\Dokumente\\Uni\\Fächer\\BSc\\GeneticProcessMining\\genetic_miner"

sys.path.append(srcpath)
os.chdir(rootpath)

# %%
# list available files
files = [f"results/data/{f}" for f in os.listdir("results/data")]
print(f"files in data:")
print('\n'.join(files))

# %%
# load pickle from fname, assign name d
import pickle as pkl

fname = files[-1]

with open(fname, "rb") as f:
    print(f"open file:\n{fname}")
    d = pkl.load(f)

ga_instance = d["speciation_params_ga_params"]

# %%
gv = ga_instance[0][1][0].get_graphviz()

gv.render()