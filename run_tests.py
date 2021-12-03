# %%
from pm4py.objects.log.importer.xes import importer as xes_importer

lp = "pm_data/running_example.xes" # "pm_data/m1_log.xes"
log = xes_importer.apply(lp)

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


# depending on if file is run as notebook or script, do different things (for debugging)
if isnotebook():
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
else:
    print("probably a debugging session, go to this function:")
    from src.tests import print_verbose_fitness as pv
    from src.tests import test_startconfigs as ts
    from sys import exit
    tg = ts.get_genomes(log)
    print("normal\n")
    pv.print_fitness(tg[1], log, False)
    print("monkey patch\n")
    pv.mpatch_test(tg[1], log, False)
    exit()

# %% ---------------------------------------------------------------------------
# ---------- TEST STARTCONFIGS ----------
# ------------------------------------------------------------------------------
from src.tests import initial_pop_speciation, test_startconfigs as ts
from src.tests import visualize_genome as vg

tg = ts.get_genomes(log)

for g in tg:
    vg.show_graphviz(g)
    g.evaluate_fitness(log)
    print(f"{10*' '}{g.fitness}")

# %% ---------------------------------------------------------------------------
from pm4py.algo.evaluation.replay_fitness.variants import token_replay
t = tg[1]

from src.tests import monkeypatch_tbr
token_replay.apply = monkeypatch_tbr.apply

token_replay.apply(log, t.net, t.im, t.fm)

# %% ---------------------------------------------------------------------------
# ---------- TEST COMPAT SCORE ----------
# ------------------------------------------------------------------------------
from src.tests import initial_pop_speciation as ips

tg = ips.run(log)
mom, dad = tg[0], tg[1] # try wiht more similar ones

t = mom.get_compatibility_score(dad)

# %% ---------------------------------------------------------------------------
# ---------- RELOAD AND SAVE PARAMS ----------
# ------------------------------------------------------------------------------
from src.tests import read_and_save_new_params as rsp

# overwrite the existing test params
rsp.save_new_params(old_name="speciation_params_new", new_name="speciation_params")

# %% ---------------------------------------------------------------------------
# ---------- PRINTING VERBOSE FITNESS ----------
# ------------------------------------------------------------------------------
from src.tests import print_verbose_fitness as pf

g1 = tg[0][0]
pf.print_fitness(g1, log, True)

# %% ---------------------------------------------------------------------------
# ---------- VISUALIZE PETRI ----------
# ------------------------------------------------------------------------------
from IPython.core.display import display, HTML, Image

# vg.vis_genome(g1)
gv1 = mom.get_graphviz()
gv2 = dad.get_graphviz()

display(Image(data=gv1.pipe(format="png"), unconfined=True, retina=True))
display(Image(data=gv2.pipe(format="png"), unconfined=True, retina=True))