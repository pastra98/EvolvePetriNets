from collections import defaultdict
from copy import copy

from neat import params

from pm4py import filter_variants, get_variants

# -------------------- For successively adding more splices during run

def get_spliced_log_on_gen(curr_gen, log):
    """gets the key indicating the splices in params if curr gen is equal or higher
    """
    unlocked_generation  = list(filter(lambda g: int(g) <= curr_gen, params.log_splices.keys()))[-1]
    splice_list = params.log_splices[unlocked_generation]
    return get_spliced_log(log, splice_list)

def get_spliced_log(log, splice_list: list[int]):
    """copy log and filter for variants indicated by the splices
    """
    log = copy(log) # TODO: check if this is necessary, might eat memory
    log._list = [log._list[int(i)] for i in splice_list]
    return log

# -------------------- Splicing methods
# ---------- for splitting a log into splices for bootstrapping

def balanced_splice(log, n_splices):
    """Splits variants into n evenly distributed splices based on variant lengths.
    Returns filtered logs for the variants
    """
    # TODO:
    # x needs to transform it back to a log
    # - develop a more balanced algo
    # - make it more explicit that we need a log
    # - maybe abstract the variant transformation??
    # - fix docstring

    variants = list(get_variants(log))

    # Group the log into sets based on the length of variants
    length_groups = defaultdict(list)
    for variant in variants:
        length_groups[len(variant)].append(variant)

    # Initialize splices
    splices = [[] for s in range(n_splices)]

    # Distribute variants into splices
    for _length, group in length_groups.items():
        num_variants = len(group)
        base_variants_per_splice = num_variants // n_splices
        extra_variants = num_variants % n_splices

        start_index = 0
        for i in range(n_splices):
            # Distribute an extra variant to the first extra_variants splices
            if i < extra_variants:
                end_index = start_index + base_variants_per_splice + 1
            else:
                end_index = start_index + base_variants_per_splice
            
            splices[i].extend(group[start_index:end_index])
            start_index = end_index

    # filter out empty splices, get full log for each variant in the splice
    return [filter_variants(log, s) for s in splices if s]
