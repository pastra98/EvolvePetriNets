from typing import Dict, List, Tuple
from functools import cache
from statistics import mean
import random as rd
from neat import params

import numpy as np
from numba import jit, boolean, int32, float64

@jit(nopython=True)
def custom_setdiff1d(ar1, ar2):
    mask = np.ones(len(ar1), dtype=np.bool_)
    for element in ar2:
        mask &= (ar1 != element)
    return ar1[mask]


@jit(nopython=True)
def update_enabled_mask_numba(marking, input_matrix):
    result = np.empty(input_matrix.shape[1], dtype=np.bool_)
    for i in range(input_matrix.shape[1]):
        result[i] = np.all(marking >= input_matrix[:, i])
    return result


@jit(nopython=True)
def fire_transition_numba(marking, input_matrix, output_matrix, change_matrix, t_idx):
    consumed = np.sum(input_matrix[:, t_idx])
    produced = np.sum(output_matrix[:, t_idx])
    
    if not np.all(marking >= input_matrix[:, t_idx]):
        missing = np.sum(np.maximum(input_matrix[:, t_idx] - marking, 0))
        marking = np.maximum(marking, input_matrix[:, t_idx])
    else:
        missing = 0
    
    marking += change_matrix[:, t_idx]
    
    return marking, consumed, produced, missing


@jit(nopython=True)
def try_enable_trans_through_hidden_numba(marking, input_matrix, output_matrix, change_matrix, t_idx, hidden_transitions, enabled_mask):
    fired_hiddens = []
    places_missing_token = np.where(marking < input_matrix[:, t_idx])[0]
    
    for ht_idx in hidden_transitions:
        overlap = np.where((output_matrix[:, ht_idx] > 0) & (marking < input_matrix[:, t_idx]))[0]
        
        if len(overlap) > 0 and enabled_mask[ht_idx]:
            marking, consumed, produced, missing = fire_transition_numba(marking, input_matrix, output_matrix, change_matrix, ht_idx)
            fired_hiddens.append((ht_idx, (int(consumed), int(produced), int(missing))))
            places_missing_token = custom_setdiff1d(places_missing_token, overlap)
            if len(places_missing_token) == 0:
                break

    return marking, fired_hiddens


class PetriNetNP:
    def __init__(self, places, transitions, log):
        self.places = places
        self.transitions = transitions
        self.log = log
        
        self.num_places = len(places)
        self.num_transitions = len(transitions)
        
        self.input_matrix = np.zeros((self.num_places, self.num_transitions), dtype=np.int32)
        self.output_matrix = np.zeros((self.num_places, self.num_transitions), dtype=np.int32)
        self.marking = np.zeros(self.num_places, dtype=np.int32)
        
        self.place_to_index = {place: i for i, place in enumerate(places)}
        self.transition_to_index = {trans: i for i, trans in enumerate(transitions)}
        self.index_to_transition = {i: trans for trans, i in self.transition_to_index.items()}
        
        self.task_transition_ids = set()
        self.task_transitions = set()
        self.hidden_transitions = []
        self.change_matrix = None
        self.enabled_mask = None


    def add_arc(self, place, transition, is_input, is_task=True):
        p_idx, t_idx = self.place_to_index[place], self.transition_to_index[transition]
        if is_input:
            self.input_matrix[p_idx, t_idx] = 1
        else:
            self.output_matrix[p_idx, t_idx] = 1
        if is_task:
            self.task_transitions.add(t_idx)
            self.task_transition_ids.add(transition)
        else:
            self.hidden_transitions.append(t_idx)


    def finalize_setup(self):
        self.change_matrix = self.output_matrix - self.input_matrix
        self.enabled_mask = np.zeros(self.num_transitions, dtype=np.bool_)
        self.hidden_transitions = np.array(self.hidden_transitions, dtype=np.int32)


    def set_initial_marking(self, initial_place):
        self.marking.fill(0)
        self.marking[self.place_to_index[initial_place]] = 1
        self._update_enabled_mask()


    def _update_enabled_mask(self):
        self.enabled_mask = update_enabled_mask_numba(self.marking, self.input_matrix)


    def fire_transition(self, t_idx):
        self.marking, consumed, produced, missing = fire_transition_numba(
            self.marking, self.input_matrix, self.output_matrix, self.change_matrix, t_idx
        )
        self._update_enabled_mask()
        return int(consumed), int(produced), int(missing)


    def _try_enable_trans_through_hidden(self, t_idx):
        self.marking, fired_hiddens = try_enable_trans_through_hidden_numba(
            self.marking, self.input_matrix, self.output_matrix, self.change_matrix,
            t_idx, self.hidden_transitions, self.enabled_mask
        )
        self._update_enabled_mask()
        return fired_hiddens


    def replay_log(self) -> List[dict]:
        log_replay = []
        for trace in self.log["variants"]:
            self.finalize_setup()
            trace_replay = self.replay_trace(trace)
            trace_fitness = self._get_trace_fitness(trace_replay)
            log_replay.append({**trace_replay, "fitness": trace_fitness})
        return log_replay


    def replay_trace(self, trace: Tuple[str]) -> dict:
        self.set_initial_marking("start")
        replay = []
        c, p, m, r = 0, 0, 0, 0

        for task in trace:
            if task not in self.transition_to_index:
                replay.append((task, (0, 0, 0), []))
                continue

            t_idx = self.transition_to_index[task]

            if not self.enabled_mask[t_idx]:
                fired_hiddens = self._try_enable_trans_through_hidden(t_idx)
                for ht_idx, quality in fired_hiddens:
                    c += quality[0]
                    p += quality[1]
                    replay.append((self.index_to_transition[ht_idx], quality, 
                                   [self.index_to_transition[i] for i, e in enumerate(self.enabled_mask) if e]))

            quality = self.fire_transition(t_idx)
            c += quality[0]
            p += quality[1]
            m += quality[2]

            replay.append((task, quality, 
                           [self.index_to_transition[i] for i, e in enumerate(self.enabled_mask) if e]))

        r = np.sum(self.marking) - self.marking[self.place_to_index["end"]]
        return {"replay": replay, "consumed": c, "produced": p, "missing": m, "remaining": int(r)}


    def _get_trace_fitness(self, trace_replay: dict) -> float:
        # Constants
        MULT = 1.5
        MAX_PTS = 1
        NO_INPUTS_PENAL = 0.5
        NO_OUTPUTS_PENAL = 0.5
        MISSING_PENAL = 0.4
        REMAINING_PENAL = 1

        fitness, n_flawless = 0, 0

        for firing_info in trace_replay["replay"]:
            trans, quality = firing_info[0], firing_info[1]
            if trans not in self.task_transition_ids:
                continue

            pts = MAX_PTS
            consumed, produced, missing = quality

            if not consumed:
                pts -= NO_INPUTS_PENAL
            if not produced:
                pts -= NO_OUTPUTS_PENAL
            if missing:
                pts -= MISSING_PENAL * (consumed / missing)

            if pts == MAX_PTS:
                n_flawless += 1
            else:
                n_flawless = 0
            fitness += pts * max(1, MULT * (n_flawless - 1))

        fitness -= REMAINING_PENAL * trace_replay["remaining"]
        return fitness


# -------------------- METRICS -------------------- 

    def _aggregate_trace_fitness(self, log_replay: list):
        """Aggregate fitness from all traces of replay, divides by max achievable fitness.
        """
        agg_fitness = 0
        for replay, cardinality in zip(log_replay, self.log["variants"].values()):
            agg_fitness += replay["fitness"] * cardinality
        max_fit = max_replay_fitness( # tuples len of trace with cardinality
            tuple([(len(t), cardinality) for t, cardinality in self.log["variants"].items()])
            )
        return agg_fitness / max_fit


    def _get_node_degrees(self) -> Dict[str, Dict[str, List[int]]]:
        p_degrees = {p: [np.sum(self.input_matrix[i, :]), np.sum(self.output_matrix[i, :])] 
                     for i, p in enumerate(self.places)}
        t_degrees = {t: [np.sum(self.input_matrix[:, i]), np.sum(self.output_matrix[:, i])] 
                     for i, t in enumerate(self.transitions)}
        return {"places": p_degrees, "transitions": t_degrees}


    def _mean_by_max_simplicity(self, node_degrees: Dict[str, Dict[str, List[int]]]) -> float:
        p_degrees = [sum(p) for p in node_degrees["places"].values()]
        t_degrees = [sum(t) for t in node_degrees["transitions"].values()]
        all_degrees = [d for d in p_degrees + t_degrees if d > 0]
        return mean(all_degrees) / max(all_degrees) if all_degrees else 0


    def _io_connectedness_simplicity(self, node_degrees: Dict[str, Dict[str, List[int]]]) -> float:
        io_connected = [p for p in node_degrees["places"].values() if p[0] > 0 and p[1] > 0]
        return len(io_connected) / self.num_places if self.num_places > 0 else 0


    def _precision(self, replay: List[Dict]) -> float:
        all_enabled = sum(len(q[2]) for trace in replay for q in trace["replay"])
        return 1 / max(all_enabled, 1)


    def _transitions_by_tokens(self, replay: List[Dict]) -> float:
        all_produced = sum(q[1][1] for trace in replay for q in trace["replay"])
        tbt = self.num_transitions / max(all_produced, 1)
        return min(tbt, 1)


    def _fraction_task_trans(self) -> float:
        if self.task_transitions:
            return len(self.task_transitions) / self.num_transitions
        else:
            return 0


    def _num_arcs(self) -> float:
        total_arcs = np.sum(self.input_matrix) + np.sum(self.output_matrix)
        return 1 / max(total_arcs, 1)


    def _trans_place_ratio(self) -> float:
        return min(self.num_transitions / self.num_places, 1) if self.num_places > 0 else 0


    def _over_enabled_transitions(self, log_replay):
        """Whenever more trans are enabled than indicated in dfg, increase denominator
        """
        s_dict = {t: [] for t in self.log["footprints"]["activities"]}
        for s in self.log["footprints"]["dfg"]:
            s_dict[s[0]].append(s[1])

        enabled_too_much = 0
        for trace_replay in log_replay:
            # if a fired trans is hidden, count its enabled trans towards next task
            enabled_by_hiddens = []
            for firing_info in trace_replay["replay"]:
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


    def evaluate(self):
        """Get all fitness metrics
        """
        # get replay & node degrees
        replay = self.replay_log()
        node_degrees = self._get_node_degrees()
        # simplicity / generalization metrics
        metrics = {
            "aggregated_replay_fitness": self._aggregate_trace_fitness(replay),
            "io_connectedness": self._io_connectedness_simplicity(node_degrees),
            "mean_by_max": self._mean_by_max_simplicity(node_degrees),
            "trans_by_tasks": self._fraction_task_trans(),
            "precision": self._precision(replay),
            "trans_by_tokens": self._transitions_by_tokens(replay),
            "over_enabled_trans": self._over_enabled_transitions(replay),
            "num_arcs": self._num_arcs(),
            "trans_by_places": self._trans_place_ratio()
        }
        return {
            "replay": replay,
            "metrics": metrics
        }


def max_replay_fitness(len_and_card: tuple):
    """Uses closed form for sum of multipliers, distributes multiplier across them
    works under the assumption that MAX_PTS == 1
    """
    fit = 0
    for tlen, cardinality in len_and_card:
        trace_fit = 1 + params.replay_mult * ((tlen * (tlen-1)) / 2)
        fit += trace_fit * cardinality
    return fit
