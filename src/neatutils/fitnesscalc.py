from neat import params

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