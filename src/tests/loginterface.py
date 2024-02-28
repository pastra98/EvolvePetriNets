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

# %%
import pandas as pd
import pm4py
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.algo.filtering.pandas.start_activities import start_activities_filter
from pm4py.objects.petri_net import importer

class LogInterface:
    def __init__(self, logpath:str)-> None:
        self.log = pm4py.read_xes(logpath)
        # self.start_task = start_t
        # self.end_task = end_t
        # convert log to dataframe
        self.df = self.get_dataframe()
        # self.tasks = self.log_df["concept:name"].unique()
    
    def get_dataframe(self) -> pd.DataFrame:
        # convert to dataframe
        log_converter = pm4py.objects.conversion.log.converter
        df = log_converter.apply(
                self.log,
                variant=log_converter.Variants.TO_DATA_FRAME
            )
        # rename cols and keep only task, time and case cols
        df = df.rename(columns={
                    "concept:name": "task",
                    "time:timestamp": "time",
                    "case:concept:name": "case",
                    "org:resource": "resource"
                })[["task", "resource", "time", "case"]]
        # df.sort_values("time", inplace=True)
        df["time"] = df["time"].dt.tz_localize(None)
        return df


lp = "pm_data/running_example.xes"
lf = LogInterface(lp)
df = lf.df
df.to_latex(buf="vis/log_table.tex", caption="A simple event log")
# logpath = "../pm_data/bpi2016/GroundTruthLogs/pdc_2016_5.xes"
# logpath = "../pm_data/m1_log.xes"
# logpath = "../pm_data/BPI_Challenge_2012.xes"
# logpath = "../pm_data/pdc_2016_6.xes"

# log = pm4py.read_xes(logpath)
# log_converter = pm4py.objects.conversion.log.converter

# df = log_converter.apply(
#     log,
#     variant=log_converter.Variants.TO_DATA_FRAME
# )

# %%
variants = variants_filter.get_variants(log)
print(f"there are {len(variants)} variants")
print(f"there are {len(log)} traces in total\n")

for i, variant in enumerate(variants):
    ev_list = [ev["concept:name"] for ev in variants[variant][0]]
    print(f"{i}:\n{' -> '.join(ev_list)}\n")

# df = df.rename(columns={
#             "concept:name": "task",
#             "time:timestamp": "time",
#             "case:concept:name": "case"
#         })[["task", "time", "case"]]

# %%
task_counts = df.value_counts("concept:name").reset_index(name='count')
task_counts_dict = df.value_counts("concept:name").to_dict()
case_counts = df.value_counts("case:concept:name").reset_index(name='count')
# print(task_counts.to_dict("index"))
# print(df.value_counts("case:concept:name").to_dict("index"))

# %%
for trace in log:
    print(f"\nTrace {trace._get_attributes()['concept:name']}:")
    tr_events = []
    for event in trace:
        tr_events.append(event["concept:name"])
    print(" -> ".join(tr_events))
