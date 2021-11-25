from src.neat import ga
from pm4py.objects.log.importer.xes import importer as xes_importer


def run():
    log_path = "pm_data/m1_log.xes"
    log = xes_importer.apply(log_path)
    param_name = "param_files/test.json"
    new_ga = ga.GeneticAlgorithm(param_name, log)