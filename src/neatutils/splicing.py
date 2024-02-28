from neat import params
from copy import copy

def get_spliced_log_on_gen(curr_gen, log):
    # gets the key indicating the splices in params if curr gen is equal or higher
    unlocked_generation  = list(filter(lambda g: int(g) <= curr_gen, params.log_splices.keys()))[-1]
    splice_list = params.log_splices[unlocked_generation]
    return get_spliced_log(log, splice_list)

def get_spliced_log(log, splice_list: list[int]):
    # copy log and filter for traces indicated by the splices
    log = copy(log) # TODO: check if this is necessary, might eat memory
    log._list = [log._list[int(i)] for i in splice_list]
    return log