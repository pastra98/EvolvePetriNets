from src.neat import ga
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.visualization.petri_net import visualizer


def run():
    log_path = "pm_data/m1_log.xes"
    log = xes_importer.apply(log_path)
    param_name = "param_files/test.json"
    genetic_algo = ga.GeneticAlgorithm(param_name, log)

    print(genetic_algo.population)
    print(len(genetic_algo.population))
    # why is dis a nested list? lots of stuff to fix here
    for g in genetic_algo.population:
        # implement genome.copy() method
        print(g.id)
        net, im, fm = g.build_petri()
        net_gviz = visualizer.apply(net, im, fm)
        savepath = f"vis/t1/{g.id}_petrinet.png"
        visualizer.save(net_gviz, savepath)
        print(f"saved under {savepath}")
