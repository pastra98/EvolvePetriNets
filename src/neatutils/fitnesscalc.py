from neat import params

from functools import cache
from typing import Tuple, Dict, List
from statistics import mean
import random as rd

class Place:
    def __init__(self):
        self.n_tokens = 0

    def remove_t(self):
        self.n_tokens -= 1

    def insert_t(self):
        self.n_tokens += 1

    def has_tokens(self) -> bool:
        return self.n_tokens > 0


class Transition:
    def __init__(self, is_task: bool):
        self.is_task = is_task
        self.inputs: Dict[str, Place] = dict()
        self.outputs: Dict[str, Place] = dict()
    
    def __str__(self):
        inputs_str = ', '.join(self.inputs.keys())
        outputs_str = ', '.join(self.outputs.keys())
        return f"Transition(is_task={self.is_task}, inputs=[{inputs_str}], outputs=[{outputs_str}])"

    def add_place(self, place_id: str, place: Place, is_input: bool):
        if is_input:
            self.inputs[place_id] = place
        else:
            self.outputs[place_id] = place
    
    def fire_and_get_quality(self):
        # enable trans if necessary, assign missing tokens
        missing = 0 if self.is_enabled() else self._enable()
        quality = (len(self.inputs), len(self.outputs), missing)
        self._fire()
        return quality


    def _fire(self):
        for p in self.inputs.values():
            p.remove_t()
        for p in self.outputs.values():
            p.insert_t()
    

    def is_enabled(self):
        """Checks if all places have tokens, trans are also enabled if they have no inputs,
        but a missing penalty will be added
        """
        for p in self.inputs.values():
            if not p.has_tokens():
                return False
        return True


    def _enable(self):
        # maybe bad design to have a fruitful function that mutates state, but this is practical
        produced = 0
        for p in self.inputs.values():
            if not p.has_tokens():
                p.insert_t()
                produced += 1
        return produced

# constants for replay
MAX_PTS = 1 #
NO_INPUTS_PENAL = 0.5
NO_OUTPUTS_PENAL = 0.5

@cache
def max_replay_fitness(len_and_card: tuple):
    """Uses closed form for sum of multipliers, distributes multiplier across them
    works under the assumption that MAX_PTS == 1
    """
    if MAX_PTS != 1:
        raise Exception("this function was built under the assumptio MAX_PTS == 1")
    fit = 0
    for tlen, cardinality in len_and_card:
        trace_fit = 1 + params.replay_mult * ((tlen * (tlen-1)) / 2)
        fit += trace_fit * cardinality
    return fit


class Petri:
    """A Petri net implementation specifically for performing token replay and
    calculating fitness metrics
    """
    def __init__(
            self,
            places: Dict[str, Place],
            transitions: Dict[str, Transition],
            log: Dict[str, dict]):
        
        self.transitions = transitions
        self.places = places
        self.hidden_transitions = {
            t_id: t for t_id, t in transitions.items() if not t.is_task
            }
        self.log = log
        # compute the prefixes, update marking dict
        self.marking_dict: Dict[int, List[dict, List[List[Place, int]]]] = {}
        self.compute_prefixes()


    def evaluate(self):
        """Get all fitness metrics
        """
        # get replay & node degrees
        replay = self.replay_log()
        # do the prefix optimized replay
        node_degrees = self._get_node_degrees()
        # simplicity / generalization metrics
        metrics = {
            "aggregated_replay_fitness": self._aggregate_trace_fitness(replay),
            "io_connectedness": self._io_connectedness(node_degrees),
            "mean_by_max": self._mean_by_max_simplicity(node_degrees),
            "trans_by_tasks": self._fraction_task_trans(),
            "precision": self._precision(replay),
            "token_usage": self._token_usage(replay),
            "over_enabled_trans": self._over_enabled_transitions(replay),
            "num_arcs": self._num_arcs(),
            "trans_by_places": self._trans_place_ratio(),
            "remaining_score": self._remaining_score(replay),
            "sink_score": self._sink_score(replay)
        }
        return {
            "replay": replay,
            "metrics": metrics
        }

# -------------------- REPLAY -------------------- 

    def replay_log(self) -> dict:
        """Using the prefix optimization, this will replace the other replay func
        """
        # the markings need to be calculated before this
        log_replay: List[dict] = []
        for var_i in range(len(self.log["variants"])):
            prefixes, rest_of_trace = self.log["shortened_variants"][var_i]
            initial_replay = self.load_marking(prefixes[-1])
            trace_replay = self._replay_trace(rest_of_trace)
            full_replay = self.merge_replay_stats(initial_replay, trace_replay)
            trace_fitness = self._get_trace_fitness(full_replay)
            log_replay.append(full_replay | {"fitness": trace_fitness})
        return log_replay


    def _replay_trace(self, trace: Tuple[str]):
        """Replay a trace, return cpmr counts. Continues from the current marking
        remember to reset it when it should start from initial marking
        """
        replay: List[Tuple[str, List[int]]] = []
        c, p, m, r = 0, 0, 0, 0 # consumed, produced, missing, remaining
        for task in trace:
            if task not in self.transitions:
                replay.append((task, (0, 0, 0), []))
                continue
            trans = self.transitions[task]
            # if there are hidden transitions and the trans is disabled, try enabling it
            if self.hidden_transitions and not trans.is_enabled():
                fired_hiddens = self._try_enable_trans_through_hidden(trans)
                if fired_hiddens:
                    for fh in fired_hiddens:
                        quality = fh[1]
                        c += quality[0]; p += quality[1] # no missings, ht is only fired if is enabled
                        replay.append(fh)
            # fire trans (if it cannot be enabled, add missing tokens
            quality = trans.fire_and_get_quality()
            c += quality[0]; p += quality[1]; m += quality[2]
            enabled_t = [t_id for t_id, t in self.transitions.items() if t.is_enabled()]
            replay.append((task, quality, enabled_t))
        # at the end of the replay, count the remaining as well as tokens in the sink
        r = sum([p.n_tokens for p_id, p in self.places.items() if p_id != "end"])
        s = self.places["end"].n_tokens
        return {"replay": replay, "consumed": c, "produced": p,
                "missing": m, "remaining": r, "sink": s}


    def _try_enable_trans_through_hidden(self, trans: Transition):
        """In the current version of this method, I do not recurse into ht further back.
        Returns the list [(ht_id, quality), ...], as well as total tokens
        consumed and produced by firing all hidden trans required to enable trans.
        May return an empty list if enabling through hidden not possible.
        """
        fired_hiddens = []
        # find the places that miss tokens, then find potential hidden trans
        places_missing_token = {id for id, p in trans.inputs.items() if not p.has_tokens()}
        hidden_trans = list(self.hidden_transitions.items())
        rd.shuffle(hidden_trans) # shuffled to not give pref to any ht
        for t_id, ht in hidden_trans:
            overlap = set(ht.outputs.keys()).intersection(places_missing_token)
            if overlap and ht.is_enabled(): # overlap contains the place(s) that connect to ht
                quality = ht.fire_and_get_quality()
                enabled_t = [t_id for t_id, t in self.transitions.items() if t.is_enabled()]
                fired_hiddens.append((t_id, quality, enabled_t))
                places_missing_token -= overlap
                if not places_missing_token:
                    break # stop if there are no more places missing a token
        return fired_hiddens


    def _get_trace_fitness(self, trace_replay: dict):
        """Use replay stats to calculate the fitness of trace, reward successive
        good executions (none missing, trans has output places)
        """
        fitness, n_flawless = 0, 0

        for firing_info in trace_replay["replay"]:
            trans, quality = firing_info[0], firing_info[1]
            # do not add pts for firing hidden trans
            if trans in self.transitions and not self.transitions[firing_info[0]].is_task:
                continue
            # if a task trans was fired, evaluate how many pts it generated
            pts = MAX_PTS
            consumed, produced, missing = quality[0], quality[1], quality[2]
            # subtract input/output penalties
            if not consumed: pts -= NO_INPUTS_PENAL
            if not produced: pts -= NO_OUTPUTS_PENAL
            # missing token penalty
            if missing:
                pts -= params.missing_penal * (consumed/missing)
            # update multiplier
            if pts == MAX_PTS:
                n_flawless += 1
            else:
                n_flawless = 0
            fitness += pts * max(1, params.replay_mult * (n_flawless-1))
        # penalize for remaining tokens (set to 0 if using the remaining score)
        return fitness

# -------------------- MARKINGS -------------------- 

    def set_initial_marking(self):
        """Clear net of tokens, set 1 in token source
        """
        for p in self.places.values():
            p.n_tokens = 0
        self.places["start"].n_tokens = 1


    def load_marking(self, prefix_id: int):
        # Again super ugly, a method that changes state of the object and is fruitful
        # return the replay state at this point
        # load the marking, i.e. token-place configuration
        replay_info, placelist = self.marking_dict[prefix_id]
        for p, num_tokens in placelist:
            p.n_tokens = num_tokens
        return replay_info


    @staticmethod
    def merge_replay_stats(r1: dict, r2: dict):
        # remaining and sink are only calculated at end of replay (their state is stored in places anyways)
        merged = {}
        for stat in r1:
            if stat in ["remaining", "sink"]:
                merged[stat] = r2[stat]
            else:
                merged[stat] = r1[stat] + r2[stat]
        return merged


    def compute_prefixes(self):

        def get_marking_and_replay(prefix: int):
            # previous state is taken care of by outer loop
            # load the prefix tasks, replay the trace, save the place info, return it
            replay_info = self._replay_trace(prefix_map[prefix])
            placelist = []
            for p in self.places.values():
                placelist.append([p, p.n_tokens])
            # big problem: need to also store the replay info when then replaying the tasks
            return [replay_info, placelist]

        prefix_map = self.log["prefixes"]

        for pf_list in self.log["unique_prefixes"]:
            # if all prefixes are already in the dict, continue
            initial_pf = pf_list[0]
            if initial_pf not in self.marking_dict:
                # need initial marking for the first prefix, no need to merge it
                self.set_initial_marking()
                self.marking_dict[initial_pf] = get_marking_and_replay(initial_pf)

            prev_pf = initial_pf
            for pf in pf_list[1:]:
                if pf not in self.marking_dict:
                    # load the state from the marking, along with replay info
                    prev_replay_info = self.load_marking(prev_pf)
                    # continue replay from the previous marking
                    replay_info, placelist = get_marking_and_replay(pf)
                    # merge the replay infos
                    merged_replay = self.merge_replay_stats(prev_replay_info, replay_info)
                    self.marking_dict[pf] = [merged_replay, placelist]
                prev_pf = pf

# -------------------- METRICS -------------------- 
# ---------- REPLAY FITNESS

    def _aggregate_trace_fitness(self, replay: list):
        """Aggregate fitness from all traces of replay, divides by max achievable fitness.
        """
        agg_fitness = 0
        # assumes the order of replay is the same as the order of variants in the log
        for replay, cardinality in zip(replay, self.log["variants"].values()):
            agg_fitness += replay["fitness"] * cardinality
        max_fit = max_replay_fitness( # tuples len of trace with cardinality
            tuple([(len(t), cardinality) for t, cardinality in self.log["variants"].items()])
            )
        fraction = agg_fitness / max_fit
        return max(fraction, 0)


    def _io_connectedness(self, node_degrees):
        """Fraction of places that have both inputs and outputs
        """
        # could do similar thing for transitions, but for now just disable
        io_connected = [p for p in node_degrees["places"].values() if p[0]>0 and p[1]>0]
        return len(io_connected) / len(self.places)


    def _remaining_score(self, replay):
        """Calculated based on the fraction of remaining tokens after replay vs.
        produced tokens
        """
        tot_produced, tot_remaining = 0, 0
        for trace in replay:
            tot_produced += trace["produced"]
            tot_remaining += trace["remaining"]

        if tot_produced == 0:
            return 1
        return 1 - (tot_remaining / tot_produced)

# ---------- SIMPLICITY

    def _get_node_degrees(self):
        """Helper func to calculate degrees of nodes and places
        """
        p_degrees = {p: [0, 0] for p in self.places}
        t_degrees = {t: [0, 0] for t in self.transitions}
        for t in self.transitions.values():
            t_degrees[t] = [len(t.inputs), len(t.outputs)]
            for p in list(t.inputs.keys()):
                p_degrees[p][0] += 1
            for p in list(t.outputs.keys()):
                p_degrees[p][1] += 1
        return {"places": p_degrees, "transitions": t_degrees}
    

    def _mean_by_max_simplicity(self, node_degrees):
        """Idea is to penalize the max degree being further away from the mean
        Could be seen as a sort of simplicity metric
        """
        # FUTUREIMPROVEMENT: could also use variance maybe?
        p_degrees = [sum(p) for p in list(node_degrees["places"].values())]
        t_degrees = [sum(t) for t in list(node_degrees["transitions"].values())]
        all_degrees = [d for d in t_degrees + p_degrees if d > 0]
        return mean(all_degrees) / max(all_degrees)
    

    def _token_usage(self, replay):
        """Penalize the model for producing too many tokens
        """
        token_sum = 0
        for trace in replay:
            token_sum += trace["produced"]
        token_ratio = params.min_tokens_for_replay / max(token_sum, 1)
        return min(token_ratio, 1) # cap it to 1


    def _fraction_task_trans(self):
        """get fraction of task trans from overall trans, i.e. penalize for every hidden trans
        """
        my_task_trans = [t for t in self.transitions.values() if t.is_task]
        if my_task_trans:
            return len(my_task_trans) / len(self.transitions)
        else:
            return 0 # if a model has 0 task trans its fitness is 0


    def _num_arcs(self):
        """Just the number of transitions divided by the number of arcs, capped to 0-1
        """
        n_arcs = sum([len(t.inputs) + len(t.outputs) for t in self.transitions.values()])
        return min(len(self.transitions) / n_arcs, 1)


    def _trans_place_ratio(self):
        """ratio of transitions to places, capped to [0-1]
        """
        return min(len(self.transitions) / len(self.places), 1)

# ---------- PRECISION

    def _precision(self, replay):
        """Precision formula from ProDiGen miner
        """
        all_enabled = 0
        for trace in replay:
            for q in trace["replay"]:
                all_enabled += len(q[2])
        return 1 / max(all_enabled, 1)


    def _over_enabled_transitions(self, replay):
        """Whenever more trans are enabled than indicated in dfg, increase denominator
        Basically an improved version of the _precision() metric above.
        """
        s_dict = {t: [] for t in self.log["footprints"]["activities"]}
        for s in self.log["footprints"]["dfg"]:
            s_dict[s[0]].append(s[1])

        enabled_too_much = 0
        for trace in replay:
            # if a fired trans is hidden, count its enabled trans towards next task
            enabled_by_hiddens = []
            for firing_info in trace["replay"]:
                trans = firing_info[0]
                enables = firing_info[2]
                # hidden trans
                if trans not in s_dict:
                    enabled_by_hiddens += enables
                # task trans
                else:
                    should_enable = s_dict[trans]
                    enabled_by_task = enables + enabled_by_hiddens
                    if len(enabled_by_task) > len(should_enable):
                        enabled_too_much += len(enabled_by_task) - len(should_enable)
                        # print(f"{trans} enabled too much:\n{enabled_by_task}\n")
                    enabled_by_hiddens.clear() # reset enabled_by_hiddens

        return 1 / max(enabled_too_much, 1) # if none were enabled too much, perfect score


    def _sink_score(self, replay):
        """Only considers traces where tokens even reached the sink. For these
        traces, there is a penalty for every token -1 in the sink. Idea is to
        penalize models for moving too many tokens into the sink (as they don't
        count towards the remaining score)
        """
        in_sink, n_valid = 0, 0
        for trace in replay:
            s = trace["sink"]
            if s > 0:
                n_valid += 1
                in_sink += s

        if n_valid > 0:
            perc_valid = n_valid / len(replay)
            sink_score = n_valid / in_sink
            return sink_score * perc_valid
        return 0

# ---------- GENERALIZATION

    def _generalization(self, replay):
        """Adapted metric by Buijs, vanDongen & vanDerAalst (2014)
        https://doi.org/10.1142/S0218843014400012
        punishes highly uneven amount transition activations

        not used because this only really makes sense with more hidden transitions..
        it should look the same for every replay anyways
        """
        task_trans_cts = {name: 0 for name, t in self.transitions.items() if t.is_task}
        for trace in replay:
            for firing_info in trace["replay"]:
                task_trans_cts[firing_info[0]] += 1
        cnt_sum = 0
        for cnt in task_trans_cts.values():
            cnt_sum += 1 / (cnt**0.5)
        gen_score = 1 - (len(task_trans_cts) / cnt_sum)
        return max(gen_score, 0)