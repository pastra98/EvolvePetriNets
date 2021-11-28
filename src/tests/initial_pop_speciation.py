from src.tests import initial_pop_speciation, test_startconfigs as ts
from src.tests import visualize_genome as vg
from src.neat import params


def run(log):
    params.load("speciation_params")
    tg = ts.get_genomes(log)
    return tg