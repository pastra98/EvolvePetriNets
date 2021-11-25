import pprint
from time import process_time

# conformance checking stuff
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.analysis.woflan import algorithm as woflan
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator

from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
from pm4py.visualization.footprints import visualizer as fp_visualizer

from time import process_time

from src.neat import startconfigs
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.visualization.petri_net import visualizer
from pm4py import view_petri_net

def get_test_genomes(save, view):
    log_path = "pm_data/m1_log.xes"
    log = xes_importer.apply(log_path)
    initial_pop = startconfigs.traces_with_concurrency(log)
    # why is dis a nested list? lots of stuff to fix here
    for g in initial_pop:
        # implement genome.copy() method
        print(g.id)
        net, im, fm = g.build_petri()
        if save:
            net_gviz = visualizer.apply(net, im, fm)
            savepath = f"vis/t1/{g.id}_petrinet.png"
            visualizer.save(net_gviz, savepath)
            print(f"saved under {savepath}")
        if view:
            view_petri_net(net, im, fm)
    return initial_pop

def get_fitness(genome):
    fit_start = process_time() 
    # fitness eval
    fitness = replay_fitness_evaluator.apply(
        log, net, initial_marking, final_marking,
        variant=replay_fitness_evaluator.Variants.TOKEN_BASED
        )
    print(f"fitness:\n{pprint.pformat(fitness, indent=4)}")
    print(f"fitness check took: {process_time()-fit_start} seconds\n")
    # soundness check
    sound_start = process_time()
    is_sound = woflan.apply(net, initial_marking, final_marking, parameters={woflan.Parameters.RETURN_ASAP_WHEN_NOT_SOUND: True,
                                                 woflan.Parameters.PRINT_DIAGNOSTICS: False,
                                                 woflan.Parameters.RETURN_DIAGNOSTICS: False})
    print(f"is sound: {is_sound}")
    print(f"sound check took: {process_time()-sound_start} seconds\n")
    # precision
    precision_start = process_time()
    prec = precision_evaluator.apply(log, net, initial_marking, final_marking, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
    print(f"precision: {prec}")
    print(f"precision check took: {process_time()-precision_start} seconds\n")
    # generealization
    generalization_start = process_time()
    gen = generalization_evaluator.apply(log, net, initial_marking, final_marking)
    print(f"generalization: {gen}")
    print(f"generalization check took: {process_time()-generalization_start} seconds\n")
    # simplicity
    simplicity_start = process_time()
    simp = simplicity_evaluator.apply(net)
    print(f"simplicity: {simp}")
    print(f"simplicity check took: {process_time()-simplicity_start} seconds\n")
    # some preliminary fitness measure
    genetic_fitness = .5*fitness["perc_fit_traces"]/100 + .5*int(is_sound) + .3*prec + .3*gen + .3*simp
    print(f"prelimary genetic fitness: {genetic_fitness}\n")