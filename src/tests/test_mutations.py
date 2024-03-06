# %%
from pty import slave_open
import sys, os, pprint
from pathlib import Path
from pm4py import view_petri_net
from src.neatutils import startconfigs



cwd = Path.cwd()

# RUN ONLY ONCE
if not os.getcwd().endswith("genetic_miner"):
    sys.path.append(str(cwd.parent.parent / "src")) # src from where all the relative imports work
    os.chdir(cwd.parent.parent) # workspace level from where I execute scripts

from IPython import get_ipython
ipython = get_ipython()
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")

from src.tests import visualize_genome as vg
from src.neat.genome import GeneticNet

# %%
from neat import innovs, params
from pm4py.objects.log.importer.xes import importer as xes_importer
from IPython.core.display import display, HTML, Image

from pm4py import view_petri_net
from pm4py.visualization.petri_net import visualizer
from IPython.core.display import display, HTML, Image

def vis_genome(g, view=True, save=False, sname=""):
    net, im, fm = g.build_petri()
    if save:
        net_gviz = visualizer.apply(net, im, fm)
        savepath = f"vis/show_mutations/"
        visualizer.save(net_gviz, savepath + sname + ".png")
        net_gviz.format = "svg"
        visualizer.save(net_gviz, savepath + sname + ".svg")
        print(f"saved under {savepath}")
    if view:
        print(g.id)
        view_petri_net(net, im, fm)

params.load("params/testing/default_speciation_params.json")

# %%
# check cloning
innovs.reset()
test_genome = startconfigs.test_startconf()
target_g = test_genome.clone()

target_g.trans_trans_conn("A", "B")
target_g.trans_trans_conn("B", "D")
target_g.trans_place_arc("D", "end")
vis_genome(target_g, save=True, sname="1_base")

target_g.extend_new_trans("p1")
vis_genome(target_g, save=True, sname="2_extend_trans")

target_g.split_arc(4)
target_g.split_arc(5)
vis_genome(target_g, save=True, sname="3_split_arc")

target_g.extend_new_place("t5")
vis_genome(target_g, save=True, sname="4_extend_place")

target_g.trans_place_arc("t3", "p6")
target_g.trans_place_arc("t3", "p2")
vis_genome(target_g, save=True, sname="5_trans_place")

del target_g.arcs[16]
vis_genome(target_g, save=True, sname="6_remove_arc")

target_g.place_trans_arc("p8", "C")
vis_genome(target_g, save=True, sname="7_place_trans")

target_g.trans_trans_conn("C", "t7")
vis_genome(target_g, save=True, sname="8_trans_trans")