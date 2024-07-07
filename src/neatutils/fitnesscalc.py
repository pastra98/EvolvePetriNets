from neat import params
from pm4py.stats import get_variants

from typing import Tuple, Dict, List
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
        return all([p.has_tokens() for p in self.inputs.values()])


    def _enable(self):
        # maybe bad design to have a fruitful function that mutates state, but this is practical
        produced = 0
        for p in self.inputs.values():
            if not p.has_tokens():
                p.insert_t()
                produced += 1
        return produced


class Petri:
    def __init__(self, places: Dict[str, Place], transitions: Dict[str, Transition]):
        self.places = places
        self.transitions = transitions
        self.hidden_transitions = {
            t_id: t for t_id, t in transitions.items() if not t.is_task
            }


    def _replay_trace(self, trace: Tuple[str]):
        self.set_initial_marking()
        replay: List[Tuple[str, List[int]]] = []
        c, p, m, r = 0, 0, 0, 0 # consumed, produced, missing, remaining
        for task in trace:
            if task not in self.transitions:
                replay.append((task, (0, 0, 0))) 
                continue
            # if trans exists, fire it (enable if necessary), save quality
            trans = self.transitions[task]
            if not trans.is_enabled():
                fired_hiddens, ht_c, ht_p = self._try_enable_trans_through_hidden(trans)
                if fired_hiddens:
                    replay += fired_hiddens
                    c += ht_c; p += ht_p
            quality = trans.fire_and_get_quality()
            c += quality[0]; p += quality[1]; m += quality[2]
            replay.append((task, quality))
        # at the end of the replay, count the missing
        r = sum([p.n_tokens for p_id, p in self.places.items() if p_id != "end"])
        return {"replay": replay, "consumed": c, "produced": p,
                "missing": m, "remaining": r}


    def _try_enable_trans_through_hidden(self, trans: Transition):
        """In the current version of this method, I do not recurse into ht further back.
        Returns the list [(ht_id, quality), ...], as well as total tokens
        consumed and produced by firing all hidden trans required to enable trans.
        May return an empty list if enabling through hidden not possible.
        """
        fired_hiddens = []
        c, p = 0, 0 # consumed, produced (no missing bc. ht only fires if enabled)
        # find the places that miss tokens, then find potential hidden trans
        places_missing_token = {id for id, p in trans.inputs.items() if not p.has_tokens()}
        hidden_trans = list(self.hidden_transitions.items()); rd.shuffle(hidden_trans)
        for t_id, ht in hidden_trans: # shuffled to not give pref to any ht
            overlap = set(ht.outputs.keys()).intersection(places_missing_token)
            if overlap and ht.is_enabled(): # overlap contains the place(s) that connect to ht
                quality = ht.fire_and_get_quality()
                c += quality[0]; p += quality[1]
                fired_hiddens.append((t_id, quality))
                places_missing_token -= overlap
                if not places_missing_token:
                    break # stop if there are no more places missing a token
        return fired_hiddens, c, p


    def set_initial_marking(self):
        for p in self.places.values():
            p.n_tokens = 0
        self.places["start"].n_tokens = 1


    def replay_log(self, log):
        # TODO: factor in cardinalities of how many traces per variant
        variants = [list(v) for v in get_variants(log).keys()]
        log_replay: List[Dict] = []
        for trace in variants:
            trace_replay = self._replay_trace(trace)
            trace_fitness = self._get_trace_fitness(trace_replay, True)
            log_replay.append(trace_replay | {"fitness": trace_fitness})
        agg_fitness = self._aggregate_trace_fitness(log_replay)
        return {"log_replay": log_replay, "fitness": agg_fitness}


    def _get_trace_fitness(self, trace_replay: dict, use_mult: bool):
        MULT = 1.5
        MAX_PTS = 1
        NO_INPUTS_PENAL = 0.5
        NO_OUTPUTS_PENAL = 0.5
        MISSING_PENAL = 0.25
        REMAINING_PENAL = 0.25
        agg_fit, n_flawless = 0, 0
        execution_qualities = [e[1] for e in trace_replay["replay"]] 
        for q in execution_qualities:
            pts = MAX_PTS
            consumed, produced, missing = q[0], q[1], q[2]
            # subtract input/output penalties
            if not consumed: pts -= NO_INPUTS_PENAL
            if not produced: pts -= NO_OUTPUTS_PENAL
            # missing token penalty
            if missing:
                pts -= MISSING_PENAL * (consumed/missing)
            # update multiplier
            if pts == MAX_PTS:
                n_flawless += 1
            else:
                n_flawless = 0
            agg_fit += pts * max(1, MULT * (n_flawless-1))
        # penalize for remaining tokens
        agg_fit -= REMAINING_PENAL * trace_replay["remaining"]
        return agg_fit / len(execution_qualities)


    def _aggregate_trace_fitness(self, log_replay: list):
        agg_fitness = 0
        for trace in log_replay:
            agg_fitness += trace["fitness"]
        return agg_fitness
