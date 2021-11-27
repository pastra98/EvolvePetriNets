from src.neat import ga

def get_ga(param_name, log):
    t_ga = ga.GeneticAlgorithm(param_name, log)