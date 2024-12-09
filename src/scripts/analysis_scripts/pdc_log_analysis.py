# %%
"""
I made this script to help choose which logs to use when evaluating the GWFM.
Because logs come in different sizes with different nums of variants, this
script prints that stuff.

Also because I will most likely go with logs for the process discovery challenges (PDC),
I let claude write a function to pick a specified pdc log based on arguments. the pdc
logs are all saved on my local hard drive

it does
* print size info about logs
* print size info about petri nets
* load petri nets from pnml using pm4py
* Iterate through different years of pdc challenges, can filter for any specific log
* Can also iterate through pm4py mining algos and apply those
"""

import pm4py
from neatutils.log import get_log_from_xes
import os
from typing import Optional


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

def get_log_stats(lp):
    """prints some stats about a log
    """
    ldict = get_log_from_xes(lp)
    df = ldict["dataframe"]
    print("unique activities:", len(df["concept:name"].unique()))
    print("total events", len(df))
    print("total variants", len(ldict["variants"]))

def get_bpic_train_logpath(
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
        
        print(f"Selected configuration for year {year}:")
        print(f"Dependent tasks: {dependent_tasks}")
        print(f"Loops: {'No' if not has_loops else loop_complexity.capitalize()}")
        print(f"OR constructs: {or_constructs}")
        print(f"Routing constructs: {routing_constructs}")
        print(f"Optional tasks: {optional_tasks}")
        print(f"Duplicate tasks: {duplicate_tasks}")
        print(f"Noise: {noise}")
    
    return os.path.join(base_dir, str(year), "Training Logs", filename)


print("running example log stats")
get_log_stats("../pm_data/running_example.xes")

# iterate through different pdc years, using the defaults I set (kind of arbitrary, e.g. using simple loops only)
for y in [2016, 2017, 2019, 2020, 2021, 2022, 2023, 2024]:
    fp = get_bpic_train_logpath(y)
    print("\nlog statistics")
    get_log_stats(fp)
    print(80*"-", "\n")

# # commented out selecting a specific year and argument, can reuse this
# fp = get_xes_filepath(2022, loop_complexity="complex")
# print("\nlog statistics")
# get_log_stats(fp)
# print(80*"-", "\n")

# %%
# iterate through mining algos

from pm4py.discovery import (
    discover_petri_net_alpha as alpha,
    discover_petri_net_alpha_plus as alpha_plus,
    discover_petri_net_inductive as inductive,
    discover_petri_net_heuristics as heuristics,
    discover_petri_net_ilp as ilp
)


miners = [alpha, alpha_plus, inductive, heuristics, ilp]

for miner in miners:
    print(miner)
    net, im, fm = miner(log)
    show_net_and_stats(net, im, fm)
