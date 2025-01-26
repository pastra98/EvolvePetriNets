from pm4py.stats import get_variants
from datetime import datetime, timedelta
from pm4py.algo.discovery.footprints.algorithm import apply as get_footprints
from pprint import pprint
from pm4py.objects.log.util import pandas_numpy_variants
from copy import copy

import pm4py

def get_log_from_xes(logpath: str, is_pdc_log=False) -> dict:
    log = pm4py.read_xes(logpath, show_progress_bar=False)
    # Add a dummy timestamp column because pm4py requires it
    if "time:timestamp" not in log.columns:
        start_date = datetime(1970, 1, 1)  # Unix epoch start
        timestamps = [start_date + timedelta(minutes=i) for i in range(len(log))]
        log['time:timestamp'] = timestamps
    # compute footprits and variants, return dict
    footprints = get_footprints(log)
    # keys of the fragments will be just the index of a variant (i.e. their order).
    # therefore to map fragments to cases, the case_variant_map maps the id (order) of
    # variants to case ids. it checks the case-variant map returned by pm4py to do this
    # this procedure is only implemented for pdc logs currently
    variants, case_variant = pandas_numpy_variants.apply(log)
    if is_pdc_log:
        case_variant_reversed = {v: k for k, v in case_variant.items()}
        case_variant_map = {}
        for v_id, v in enumerate(variants.keys()):
            case_id = case_variant_reversed[v]
            case_variant_map.setdefault(case_id, []).append(v_id)
    else:
        case_variant_map = None

    fragments, shortened_variants, unique_prefix = split_log_fragments(variants)

    return {
        "dataframe": log,
        "footprints": footprints,
        "variants": variants,
        "task_list": [a for a in footprints["activities"]],
        "prefixes": fragments,
        "unique_prefixes": unique_prefix,
        "shortened_variants": shortened_variants,
        "case_variant_map": case_variant_map
    }


def split_log_fragments(variants_d: dict) -> dict:
    """For the dumbest, ugliest, probably-incorrect-but-working-well-enough computation
    of a prefix tree, look no further! 100% certified organic programmer spaghetti,
    this is what happens when I hack until it kinda works and then be done with it.
    """
    variants = list(variants_d.keys())
    longest_v = max([len(v) for v in variants])

    # first get the branches of the task tree
    task_tree = {}
    for task_nr in range(longest_v):
        curr_tasks = {}
        for var_i, var in enumerate(variants):
            if task_nr+1 > len(var):
                continue # shorter variants
            task = var[task_nr]
            curr_tasks.setdefault(task, set()).add(var_i)
        task_tree[task_nr] = curr_tasks

    # then construct fragments/prefixes
    final_fragments = {}
    fragments = {tuple([k]): v for k, v in task_tree[0].items()}
    for task_nr, branches in list(task_tree.items())[1:]:
        new_fragments = {}
        for t, vids in branches.items():
        
            for fk, fv in fragments.items():
                if overlap := vids.intersection(fv):
                    if len(overlap) == len(fv): # extend current fragment
                        new_fragments[fk + tuple([t])] = overlap
                    else: # fragment branches of
                        if fk in final_fragments:
                            final_fragments[fk] = final_fragments[fk].union(fv)
                        else:
                            final_fragments[fk] = fv # keep the old fragment in final fragments
                        new_fragments[tuple([t])] = overlap # start a new one
        fragments = new_fragments

    # a dict to give each fragment/prefix an id
    tagged_fragments = {}
    for i, item in enumerate(final_fragments.items()):
        k, v = item
        tagged_fragments[i] = [k, v]

    # assign prefixes to traces, save the rest of the trace
    shortened_variants = {}
    prefix_predecessors = {}
    unique_prefixes = set()
    for i, var in enumerate(variants):
        match_frags = []
        rest_of_var = copy(var)
        for j, f in tagged_fragments.items():
            tasks, matching_vars = f
            if i in matching_vars and rest_of_var[:len(tasks)] == tasks:
                if len(match_frags) <= 1:
                    match_frags.append(j)
                    rest_of_var = rest_of_var[len(tasks):]
                    prefix_predecessors[j] = match_frags[0]
                # check if this is the one predecessor
                else:
                    prefix_pred = prefix_predecessors.get(j)
                    if prefix_pred == None or prefix_pred == match_frags[-1]:
                        prefix_predecessors[j] = match_frags[-1]
                        match_frags.append(j)
                        rest_of_var = rest_of_var[len(tasks):]
        shortened_variants[i] = [match_frags, rest_of_var]
        unique_prefixes.add(tuple(match_frags))

    prefix_map = {k: v[0] for k, v in tagged_fragments.items()}
    return prefix_map, shortened_variants, unique_prefixes
