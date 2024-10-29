from pm4py.stats import get_variants
from pm4py.algo.discovery.footprints.algorithm import apply as get_footprints

import pm4py

def get_log_from_xes(logpath: str) -> dict:
    log = pm4py.read_xes(logpath, show_progress_bar=False)
    footprints = get_footprints(log)
    return {
        "dataframe": log,
        "footprints": footprints,
        "variants": get_variants(log),
        "task_list": [a for a in footprints["activities"]]
    }