import importlib
from src.neat import genome

importlib.reload(genome)

def get_ga(param_name, log):
    t_ga = ga.GeneticAlgorithm(param_name, log)
