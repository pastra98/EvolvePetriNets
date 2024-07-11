from pm4py.stats import get_variants
from pm4py.algo.discovery.footprints.algorithm import apply as get_footprints

import pm4py

def get_log_from_xes(logpath: str) -> dict:
    log = pm4py.read_xes(logpath)
    return {
        "variants": get_variants(log),
        "footprints": get_footprints(log)
    }