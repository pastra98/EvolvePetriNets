from src.neat import startconfigs, innovs
from pm4py import view_petri_net

from src.neat import genome
from src.neat import params

params.load("test")

def get_fp_log(log):
    return startconfigs.footprints(log)

def get_genomes(log):
    return startconfigs.traces_with_concurrency(log)