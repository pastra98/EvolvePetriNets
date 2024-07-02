from neat import params
from pm4py.stats import get_variants

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


def transition_execution_quality_multiplier(aligned_traces, log):
    """Very hacky function that applies multiplier for successively executed trans
    without problems. Aligning with variants however is very hacky (relies on log
    being sorted the same always), and variants shouldn't need to be calculated
    every time this is invoked. I.E.: NOT READY FOR USE.
    """
    def align_replay_to_variant(replay: list, variant: list):
        aligned_replay = []
        for item in variant:
            if replay and item == replay[0]:
                aligned_replay.append(replay.pop(0))
            else:
                aligned_replay.append(None)
        return aligned_replay

    total_quality = 0
    variants = list(get_variants(log).keys())
    for replay, variant in zip(aligned_traces, variants):
        activated_t = [t.label for t in replay["activated_transitions"]]
        bad_t = [t.label for t in replay["transitions_with_problems"]]
        ok_t = [t for t in activated_t if t not in bad_t]
        aligned = align_replay_to_variant(ok_t, variant)

        # Step 2: Iterate and score
        ok_score = 0
        multiplier = 1
        for item in aligned:
            if item is not None:
                ok_score += 1 * multiplier
                multiplier += 1
            else:
                multiplier = 1

        score = (
            len(bad_t) * params.t_exec_scoring_weight[0] +
            ok_score  * params.t_exec_scoring_weight[1]
        )
        total_quality += score
        
    return total_quality
