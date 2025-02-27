# %%
"""
Based on the log eval thing
"""
import pm4py
from neatutils.log import get_log_from_xes
import os
from typing import Optional
from scripts.helper_scripts import setup_analysis as sa
import timeit
from statistics import median



def print_net_stats(net):
    """Prints num places, transitions and arcs for a net
    """
    print("num places:", len(net.places))
    print("num transitions:", len(net.transitions))
    print("num arcs:", len(net.arcs))


def load_pnml_show_stats(fp):
    """reads a pnml file and then calls show_net_and_stats()
    """
    net, im, fm = pm4py.read_pnml(fp)
    show_net_and_stats(net, im, fm)


def show_net_and_stats(net, im, fm):
    """wrapper for visualizing a petri net and printing some size and order info about the net
    """
    pm4py.view_petri_net(net, im, fm)
    print_net_stats(net)

def get_log_and_stats(lp):
    """prints some stats about a log
    """
    ldict = get_log_from_xes(lp)
    df = ldict["dataframe"]
    return ldict, {
        "log unique activities": len(df["concept:name"].unique()),
        "log total events": len(df),
        "log total variants": len(ldict["variants"])
    }

def get_bpic_train_logpath_and_modelpath(
    year: int,
    pre_2020: int = 1,
    dependent_tasks: bool = True,
    has_loops: bool = True,
    loop_complexity: str = "simple",
    or_constructs: bool = True,
    routing_constructs: bool = False,
    optional_tasks: bool = False,
    duplicate_tasks: bool = False,
    noise: bool = False
    ) -> str:
    """
    Returns the filepath for an XES file based on the year and configuration parameters.
    
    Args:
        year (int): The year of the contest (2016-2024)
        pre_2020 (int): For years before 2020, specifies which log number to use (1-10)
        dependent_tasks (bool): Long-term dependencies between tasks
        has_loops (bool): Enable loops in the model
        loop_complexity (str): "simple" or "complex" (only used if has_loops is True)
        or_constructs (bool): Include OR splits/joins
        routing_constructs (bool): Include invisible routing tasks
        optional_tasks (bool): Allow skipping some tasks
        duplicate_tasks (bool): Include recurrent activities
        noise (bool): Include noise in traces
    
    Returns:
        str: The complete filepath to the XES file
    """
    base_dir = r"I:/EvolvePetriNets/pm_data/pdc_logs"
    
    if year not in range(2016, 2025) or year == 2018:
        raise ValueError(f"Invalid year {year}. Must be between 2016-2024, excluding 2018.")
    
    if year < 2020:
        if not 1 <= pre_2020 <= 10:
            raise ValueError("pre_2020 must be between 1 and 10")
        filename = f"pdc_{year}_{pre_2020}.xes"
        print(f"Selected log number {pre_2020} for year {year}")
        
    else:
        # Convert to A-G notation for filename
        config = "".join([
            "1" if dependent_tasks else "0",
            "2" if has_loops and loop_complexity.lower() == "complex" else
            "1" if has_loops else "0",
            "1" if or_constructs else "0",
            "1" if routing_constructs else "0",
            "1" if optional_tasks else "0",
            "1" if duplicate_tasks else "0",
            "1" if noise else "0"
        ])
        filename = f"pdc_{year}_{config}.xes" if year < 2021 else f"pdc{year}_{config}.xes"
        # set the isclassified whatever to False
        if year >= 2023:
            filename = filename.rstrip(".xes") + "0" + ".xes"
    
    if year < 2021:
        modelname = filename.replace(".xes", ".pnml")
    elif year < 2023:
        modelname = filename.replace("0.xes", ".pnml")
    else:
        modelname = filename.replace("00.xes", ".pnml") 
    modelpath = os.path.join(base_dir, str(year), "Models", modelname)
    logpath = os.path.join(base_dir, str(year), "Training Logs", filename)
    return modelpath, logpath


# iterate through different pdc years, using the defaults I set (kind of arbitrary, e.g. using simple loops only)
# probably do this loop multiple times for stat significance

data = []
# wow this code suuuucks
for y in [2020, 2021, 2022, 2023, 2024]:
    # fp = get_bpic_train_logpath(y)
    for dt in [True, False]:
        for hl in [True, False]:
            mp, lp = get_bpic_train_logpath_and_modelpath(y, dependent_tasks=dt, has_loops=hl, loop_complexity="complex", or_constructs=False)
            log, stats = get_log_and_stats(lp)
            g = sa.load_genome_from_pnml(mp)
            t = []
            for _ in range(10):
                start_time = timeit.default_timer()
                g.evaluate_fitness(log)
                time_taken = timeit.default_timer() - start_time
                t.append(time_taken * 1_000)
            data.append({
                "time to eval (ms)": median(t),
                "model transitions": len(g.transitions),
                "model arcs": len(g.arcs),
                "model places": len(g.places),
                "logpath": lp,
                "is GWFM result": False,
            } | stats)


# %%
from pathlib import Path
from scripts.analysis_scripts import useful_functions as uf

p = Path("I:/EvolvePetriNets/analysis/data/performance/mined_models")

data=[]
for setup in p.iterdir():
    log, stats = None, None
    models = []
    # load the data
    for f in setup.iterdir():
        if f.name.endswith(".xes"):
            log, stats = get_log_and_stats(lp)
            lp = str(f)
        elif f.name.endswith(".gz"):
            models.append(uf.load_pickle(str(f)))
    # eval the models 
    for g in models:
        for _ in range(10):
            start_time = timeit.default_timer()
            g.evaluate_fitness(log)
            time_taken = timeit.default_timer() - start_time
            t.append(time_taken * 1_000)
        # add data to data
        data.append({
            "time to eval (ms)": median(t),
            "model transitions": len(g.transitions),
            "model arcs": len(g.arcs),
            "model places": len(g.places),
            "logpath": lp,
            "is GWFM result": True,
        } | stats)

# %%
import pandas as pd
df = pd.DataFrame(data)
# df.sort_values("log total events")
df.to_feather("../analysis/data/performance/eval_times_just_mined.feather")

# TODO: look at correlations between data here in the df

# %%
from ydata_profiling import ProfileReport
import pandas as pd
df = pd.read_feather("../analysis/data/performance/eval_times_just_mined.feather")
profile = ProfileReport(df)
profile.to_file("report_just_mined.html")


# %%
# import matplotlib.pyplot as plt

# df.plot(kind="scatter", x="time to eval", y="log total events", figsize=(6,4), title="Time to evaluate ~ Events in Log")

