'''
    This file is part of PM4Py (More Info: https://pm4py.fit.fraunhofer.de).

    PM4Py is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PM4Py is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PM4Py.  If not, see <https://www.gnu.org/licenses/>.
'''

from pm4py.algo.conformance.tokenreplay import algorithm as executor
from pm4py.objects import log as log_lib
from pm4py.algo.evaluation.precision import utils as precision_utils
from pm4py.statistics.start_activities.log.get import get_start_activities
from pm4py.objects.petri_net.utils.align_utils import get_visible_transitions_eventually_enabled_by_marking
from enum import Enum
from pm4py.objects.log.obj import EventLog
from pm4py.algo.conformance.tokenreplay.variants import token_replay
from pm4py.util.xes_constants import DEFAULT_NAME_KEY
from pm4py.util import constants
from pm4py.objects.petri_net.obj import PetriNet, Marking

from collections import Counter
from math import sqrt

from . import params

class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    ATTRIBUTE_KEY = constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY
    TOKEN_REPLAY_VARIANT = "token_replay_variant"
    CLEANING_TOKEN_FLOOD = "cleaning_token_flood"
    MULTIPROCESSING = "multiprocessing"

################################################################################
# monkeypatch pulp shit
from pm4py.algo.analysis.woflan.place_invariants import utility
from pm4py.algo.analysis.woflan import algorithm as woflan
import neatutils.monkeypatch_pulp as mp_pulp
utility.transform_basis = mp_pulp.transform_basis
woflan.transform_basis = mp_pulp.transform_basis
################################################################################

def get_replay_fitness(aligned_traces):
    """
    Gets a dictionary expressing fitness in a synthetic way from the list of boolean values
    saying if a trace in the log is fit, and the float values of fitness associated to each trace

    Parameters
    ------------
    aligned_traces
        Result of the token-based replayer
    parameters
        Possible parameters of the evaluation

    Returns
    -----------
    dictionary
        Containing two keys (percFitTraces and averageFitness)
    """
    no_traces = len(aligned_traces)
    fit_traces = len([x for x in aligned_traces if x["trace_is_fit"]])
    sum_of_fitness = sum([x["trace_fitness"] for x in aligned_traces])
    perc_fit_traces = 0.0
    average_fitness = 0.0
    log_fitness = 0
    total_m = sum([x["missing_tokens"] for x in aligned_traces])
    total_c = sum([x["consumed_tokens"] for x in aligned_traces])
    total_r = sum([x["remaining_tokens"] for x in aligned_traces])
    total_p = sum([x["produced_tokens"] for x in aligned_traces])
    if no_traces > 0 and total_c > 0 and total_p > 0:
        perc_fit_traces = float(100.0 * fit_traces) / float(no_traces)
        average_fitness = float(sum_of_fitness) / float(no_traces)
        log_fitness = 0.5 * (1 - total_m / total_c) + 0.5 * (1 - total_r / total_p)
    return {"perc_fit_traces": perc_fit_traces, "average_trace_fitness": average_fitness, "log_fitness": log_fitness,
            "percentage_of_fitting_traces": perc_fit_traces }


def get_aligned_traces(log, petri_net, initial_marking, final_marking):
    parameters_tr = {token_replay.Parameters.ACTIVITY_KEY: DEFAULT_NAME_KEY,
                     token_replay.Parameters.CONSIDER_REMAINING_IN_FITNESS: True,
                     token_replay.Parameters.CLEANING_TOKEN_FLOOD: False,
                     token_replay.Parameters.TRY_TO_REACH_FINAL_MARKING_THROUGH_HIDDEN: True,
                     token_replay.Parameters.WALK_THROUGH_HIDDEN_TRANS: True,
                     token_replay.Parameters.SHOW_PROGRESS_BAR: False}

    aligned_traces = executor.apply(
        log, petri_net, initial_marking, final_marking,
        variant=executor.Variants.TOKEN_REPLAY, parameters=parameters_tr
        )
    return aligned_traces


def get_generalization(petri_net, aligned_traces):
    """
    Gets the generalization from the Petri net and the list of activated transitions
    during the replay

    The approach has been suggested by the paper
    Buijs, Joos CAM, Boudewijn F. van Dongen, and Wil MP van der Aalst. "Quality dimensions in process discovery:
    The importance of fitness, precision, generalization and simplicity."
    International Journal of Cooperative Information Systems 23.01 (2014): 1440001.

    A token replay is applied and, for each transition, we can measure the number of occurrences
    in the replay. The following formula is applied for generalization

           \sum_{t \in transitions} (math.sqrt(1.0/(n_occ_replay(t)))
    1 -    ----------------------------------------------------------
                             # transitions

    Parameters
    -----------
    petri_net
        Petri net
    aligned_traces
        Result of the token-replay

    Returns
    -----------
    generalization
        Generalization measure
    """
    trans_occ_map = Counter()
    for trace in aligned_traces:
        for trans in trace["activated_transitions"]:
            trans_occ_map[trans] += 1
    inv_sq_occ_sum = 0.0
    for trans in trans_occ_map:
        this_term = 1.0 / sqrt(trans_occ_map[trans])
        inv_sq_occ_sum = inv_sq_occ_sum + this_term
    for trans in petri_net.transitions:
        if trans not in trans_occ_map:
            inv_sq_occ_sum = inv_sq_occ_sum + 1
    generalization = 1.0
    if len(petri_net.transitions) > 0:
        generalization = 1.0 - inv_sq_occ_sum / float(len(petri_net.transitions))
    return generalization


def get_precision(log: EventLog, net: PetriNet, marking: Marking, final_marking: Marking):
    token_replay_variant = executor.Variants.TOKEN_REPLAY
    activity_key = log_lib.util.xes.DEFAULT_NAME_KEY

    # default value for precision, when no activated transitions (not even by looking at the initial marking) are found
    precision = 1.0
    sum_ee = 0
    sum_at = 0

    parameters_tr = {
        token_replay.Parameters.CONSIDER_REMAINING_IN_FITNESS: False,
        token_replay.Parameters.TRY_TO_REACH_FINAL_MARKING_THROUGH_HIDDEN: False,
        token_replay.Parameters.STOP_IMMEDIATELY_UNFIT: True,
        token_replay.Parameters.WALK_THROUGH_HIDDEN_TRANS: True,
        token_replay.Parameters.CLEANING_TOKEN_FLOOD: False,
        token_replay.Parameters.ACTIVITY_KEY: activity_key,
        token_replay.Parameters.SHOW_PROGRESS_BAR: False
    }

    prefixes, prefix_count = precision_utils.get_log_prefixes(log, activity_key=activity_key)
    prefixes_keys = list(prefixes.keys())
    fake_log = precision_utils.form_fake_log(prefixes_keys, activity_key=activity_key)

    aligned_traces = executor.apply(fake_log, net, marking, final_marking, variant=token_replay_variant,
                                        parameters=parameters_tr)

    # fix: also the empty prefix should be counted!
    start_activities = set(get_start_activities(log))
    trans_en_ini_marking = set([x.label for x in get_visible_transitions_eventually_enabled_by_marking(net, marking)])
    diff = trans_en_ini_marking.difference(start_activities)
    sum_at += len(log) * len(trans_en_ini_marking)
    sum_ee += len(log) * len(diff)
    # end fix

    for i in range(len(aligned_traces)):
        if aligned_traces[i]["trace_is_fit"]:
            log_transitions = set(prefixes[prefixes_keys[i]])
            activated_transitions_labels = set(
                [x.label for x in aligned_traces[i]["enabled_transitions_in_marking"] if x.label is not None])
            sum_at += len(activated_transitions_labels) * prefix_count[prefixes_keys[i]]
            escaping_edges = activated_transitions_labels.difference(log_transitions)
            sum_ee += len(escaping_edges) * prefix_count[prefixes_keys[i]]

    if sum_at > 0:
        precision = 1 - float(sum_ee) / float(sum_at)

    return precision


def transition_execution_quality(replay):
    # import pprint as pp
    # pp.pprint(replay)
    total_quality = 0
    for trace in replay:
        bad = set([t.label for t in trace["transitions_with_problems"]])
        ok = set([t.label for t in trace["activated_transitions"]])
        score = (
            len(bad) * params.t_exec_scoring_weight[0] +
            len(ok.difference(bad)) * params.t_exec_scoring_weight[1]
        )
        total_quality += score
    return total_quality