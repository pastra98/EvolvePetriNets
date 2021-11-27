import pprint
from time import process_time

# conformance checking stuff
from pm4py import view_petri_net
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.analysis.woflan import algorithm as woflan
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator

from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
from pm4py.visualization.footprints import visualizer as fp_visualizer

from pm4py.algo.conformance.tokenreplay import algorithm as token_based_replay

def print_fitness(genome, log, view):
    genome.build_petri()
    net, im, fm = genome.net, genome.im, genome.fm
    if view:
        print(genome.id)
        view_petri_net(net, im, fm)
    fit_start = process_time() 

    # set tbr params
    tbr = token_based_replay.Variants.TOKEN_REPLAY.value.Parameters
    tbr_params = {tbr.SHOW_PROGRESS_BAR: False}

    # fitness eval
    fitness = replay_fitness_evaluator.apply(
        log, net, im, fm,
        parameters=tbr_params,
        variant=replay_fitness_evaluator.Variants.TOKEN_BASED
        )
    print(f"fitness:\n{pprint.pformat(fitness, indent=4)}")
    print(f"fitness check took: {process_time()-fit_start} seconds\n")

    # soundness check
    sound_start = process_time()
    is_sound = woflan.apply(net, im, fm, parameters={
        woflan.Parameters.RETURN_ASAP_WHEN_NOT_SOUND: True,
        woflan.Parameters.PRINT_DIAGNOSTICS: False,
        woflan.Parameters.RETURN_DIAGNOSTICS: False
        })
    print(f"is sound: {is_sound}")
    print(f"sound check took: {process_time()-sound_start} seconds\n")

    # precision
    precision_start = process_time()
    prec = precision_evaluator.apply(
        log, net, im, fm,
        parameters=tbr_params,
        variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
        )
    print(f"precision: {prec}")
    print(f"precision check took: {process_time()-precision_start} seconds\n")

    # generealization
    generalization_start = process_time()
    gen = generalization_evaluator.apply(log, net, im, fm)
    print(f"generalization: {gen}")
    print(f"generalization check took: {process_time()-generalization_start} seconds\n")

    # simplicity
    simplicity_start = process_time()
    simp = simplicity_evaluator.apply(net)
    print(f"simplicity: {simp}")
    print(f"simplicity check took: {process_time()-simplicity_start} seconds\n")

    # some preliminary fitness measure
    genetic_fitness = (
        + 1.0 * (fitness["perc_fit_traces"] / 100)
        + 0.5 * int(is_sound)
        + 0.3 * prec
        + 0.3 * gen
        + 0.3 * simp
    )
    print(f"prelimary genetic fitness: {genetic_fitness}\n")

    print(f"overall time for fitness calc:\n{process_time()-fit_start}")
    print(f"{80*'-'}\n")
