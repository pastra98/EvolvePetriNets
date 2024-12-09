from pm4py.stats import get_variants
from datetime import datetime, timedelta
from pm4py.algo.discovery.footprints.algorithm import apply as get_footprints

import pm4py

def get_log_from_xes(logpath: str) -> dict:
    log = pm4py.read_xes(logpath, show_progress_bar=False)
    # Add a dummy timestamp column because pm4py requires it
    if "time:timestamp" not in log.columns:
        start_date = datetime(1970, 1, 1)  # Unix epoch start
        timestamps = [start_date + timedelta(minutes=i) for i in range(len(log))]
        log['time:timestamp'] = timestamps
    # compute footprits and variants, return dict
    footprints = get_footprints(log)
    variants = get_variants(log)
    return {
        "dataframe": log,
        "footprints": footprints,
        "variants": variants,
        "task_list": [a for a in footprints["activities"]]
    }