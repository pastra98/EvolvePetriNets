# %%
import sys # syspath fuckery to run from same location as main
import os # set cwd to same as project cwd from vscode
import pprint as pp
from pm4py import view_petri_net

srcpath = "C:\\Users\\pauls\\OneDrive\\Dokumente\\Uni\\Fächer\\BSc\\GeneticProcessMining\\genetic_miner\\src"
rootpath = "C:\\Users\\pauls\\OneDrive\\Dokumente\\Uni\\Fächer\\BSc\\GeneticProcessMining\\genetic_miner"

sys.path.append(srcpath)
os.chdir(rootpath)

from IPython import get_ipython
ipython = get_ipython()
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")

# %%
from neat import startconfigs, innovs, params
from pm4py.objects.log.importer.xes import importer as xes_importer

lp = "pm_data/running_example.xes" # "pm_data/m1_log.xes"
log = xes_importer.apply(lp)
innovs.reset()
params.load("speciation_params")

test_genomes = startconfigs.traces_with_concurrency(log)
target_g = test_genomes[1]

# %%
# define show_graphviz(), show it for all test genomes
from pm4py.visualization.petri_net import visualizer
from IPython.core.display import display, HTML, Image

def show_graphviz(g):
    gviz = g.get_graphviz()
    display(Image(data=gviz.pipe(format="png"), unconfined=True, retina=True))

for g in test_genomes:
    print(g.id)
    show_graphviz(g)

# %%
# perform mutations on target g
show_graphviz(target_g)
# target_g.place_trans_arc()
target_g.trans_place_arc()
show_graphviz(target_g)

# %%
# check uniqueness of genes
for g in test_genomes:
    print()
    pp.pprint(g.transitions)
    g.build_petri()
    view_petri_net(g.net, g.im, g.fm)
    # # show_graphviz(g)
    # print("clone:")
    # g2 = g.clone()
    # pp.pprint(g2.transitions)
    # # show_graphviz(g2)
    # view_petri_net(g2.net, g2.im, g2.fm)