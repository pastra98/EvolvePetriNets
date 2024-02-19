from neat import params

################################################################################
# monkeypatch pulp shit
from pm4py.algo.analysis.woflan.place_invariants import utility
from pm4py.algo.analysis.woflan import algorithm as woflan
import neatutils.monkeypatch_pulp as mp_pulp
utility.transform_basis = mp_pulp.transform_basis
woflan.transform_basis = mp_pulp.transform_basis
################################################################################

def transition_execution_quality(aligned_traces):
    total_quality = 0
    for trace in aligned_traces:
        bad = set([t.label for t in trace["transitions_with_problems"]])
        ok = set([t.label for t in trace["activated_transitions"]])
        score = (
            len(bad) * params.t_exec_scoring_weight[0] +
            len(ok.difference(bad)) * params.t_exec_scoring_weight[1]
        )
        total_quality += score
    return total_quality