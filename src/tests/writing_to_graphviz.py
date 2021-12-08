# %%
from pm4py.objects.log.importer.xes import importer as xes_importer

lp = "pm_data/running_example.xes" # "pm_data/m1_log.xes"
log = xes_importer.apply(lp)

# %%