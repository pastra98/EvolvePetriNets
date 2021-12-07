# %%
import sys # syspath fuckery to run from same location as main
import os # set cwd to same as project cwd from vscode
import pprint as pp
from pm4py import view_petri_net

# TODO do this using python lol
# laptop
# srcpath = "C:\\Users\\pauls\\OneDrive\\Dokumente\\Uni\\Fächer\\BSc\\GeneticProcessMining\\genetic_miner\\src"
# rootpath = "C:\\Users\\pauls\\OneDrive\\Dokumente\\Uni\\Fächer\\BSc\\GeneticProcessMining\\genetic_miner"
# desktop
srcpath = "D:\\Bibliotheken\\OneDrive\\Dokumente\\Uni\\Fächer\\BSc\\GeneticProcessMining\\genetic_miner\\src"
rootpath = "D:\\Bibliotheken\\OneDrive\\Dokumente\\Uni\\Fächer\\BSc\\GeneticProcessMining\\genetic_miner"

sys.path.append(srcpath)
os.chdir(rootpath)

from IPython import get_ipython
ipython = get_ipython()
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")

# %%
from neat import startconfigs, innovs, params
from pm4py.objects.log.importer.xes import importer as xes_importer
from IPython.core.display import display, HTML, Image

def show_graphviz(g):
    gviz = g.get_graphviz()
    display(Image(data=gviz.pipe(format="png"), unconfined=True, retina=True))

lp = "pm_data/running_example.xes" # "pm_data/m1_log.xes"
log = xes_importer.apply(lp)
innovs.reset()
params.load("speciation_params")

test_genomes = startconfigs.traces_with_concurrency(log)

# %%
# check cloning
target_g = test_genomes[1]
g2 = target_g.clone()
for _ in range(3):
    print()

    # pp.pprint(g.transitions)
    show_graphviz(target_g)

    print("clone:")
    # pp.pprint(g2.transitions)
    show_graphviz(g2)
    g2.trans_place_arc()

# %%
# visualize all test genomes
for g in test_genomes:
    print(g.id)
    show_graphviz(g)

# %%
# perform mutations on target g
target_g = test_genomes[1]

show_graphviz(target_g)

# target_g.trans_trans_conn()

# target_g.place_trans_arc()
# target_g.trans_place_arc()

# target_g.extend_new_place()
# target_g.extend_new_trans()

show_graphviz(target_g)
