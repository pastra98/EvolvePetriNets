# %%
from pm4py.objects.log.importer.xes import importer as xes_importer

lp = "pm_data/running_example.xes" # "pm_data/m1_log.xes"
log = xes_importer.apply(lp)

# from IPython import get_ipython
# ipython = get_ipython()
# ipython.magic("load_ext autoreload")
# ipython.magic("autoreload 2")

# %% ---------------------------------------------------------------------------
# ---------- TEST STARTCONFIGS ----------
# ------------------------------------------------------------------------------
from src.tests import initial_pop_speciation, test_startconfigs as ts
from src.tests import visualize_genome as vg

tg = ts.get_genomes(log)

for g in tg:
    vg.vis_genome(g)
    g.evaluate_fitness(log)
    print(f"{10*' '}{g.fitness}")

# %% ---------------------------------------------------------------------------
# ---------- TEST COMPAT SCORE ----------
# ------------------------------------------------------------------------------
from src.tests import initial_pop_speciation as ips

tg = ips.run(log)
mom, dad = tg[0], tg[1] # try wiht more similar ones
t = mom.get_compatibility_score(dad)
# TODO
# shits broken because matched, disjoint, excess add up to more than
# len(longest) which should be impossible. step thru to find out!
# then fix innov finding for places and transitions

# mom_set = set(mom.arcs.keys())
# dad_set = set(dad.arcs.keys())
# print(mom_set)
# print(dad_set)
# print(mom_set.intersection(dad_set))

# # print arcs for mom and dad
# for a_id, a in mom.arcs.items():
#     print(f"{a_id}: {a.source_id} -> {a.target_id}")
# for a_id, a in dad.arcs.items():
#     print(f"{a_id}: {a.source_id} -> {a.target_id}")

# %% ---------------------------------------------------------------------------
# ---------- RELOAD AND SAVE PARAMS ----------
# ------------------------------------------------------------------------------
from src.tests import read_and_save_new_params as rsp

# overwrite the existing test params
rsp.save_new_params(old_name="speciation_params", new_name="speciation_params")

# %% ---------------------------------------------------------------------------
# ---------- PRINTING VERBOSE FITNESS ----------
# ------------------------------------------------------------------------------
from src.tests import print_verbose_fitness as pf

g1 = tg[0][0]
pf.print_fitness(g1, log, True)

# %% ---------------------------------------------------------------------------
# ---------- VISUALIZE PETRI ----------
# ------------------------------------------------------------------------------

vg.vis_genome(g1)